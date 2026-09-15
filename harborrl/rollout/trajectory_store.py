from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from slime.utils.types import Sample

from harborrl.types import RunContext, TaskSpec

logger = logging.getLogger(__name__)


from harborrl.env import env_bool as _env_bool, env_int as _env_int
from harborrl.environments.registry import (
    interval_candidates_for_slug as _interval_candidates_for_slug,
)
from harborrl.environments.registry import slug_for as _slug_for


# ─── Trajectory export ───
# Toggle via env var TERMINAL_SAVE_TRAJ_DIR (empty=disabled).
# Output layout (one dir per rollout sample):
#   {save_dir}/t{task}_r{rollout_id}_st{train_step}_g{group}_s{sample}_{uid}_{ts}/
#       meta.json       # task spec + sampling params + reward breakdown
#       traj.json       # per-turn dialogue + tool calls + reward metadata


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def _get_terminal_save_dir() -> Path | None:
    save_dir = os.getenv("TERMINAL_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("TERMINAL_SAVE_TRAJ_DIR=%s mkdir failed: %s", save_dir, exc)
        return None
    return path


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_or_env_int(sample: Sample, key: str, env_name: str) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key)
    if value is None:
        value = os.getenv(env_name)
    return _optional_int(value)


def _trajectory_dataset_slug(data_source: str | None) -> str:
    return _slug_for(data_source)


def _interval_candidates_for_dataset(
    dataset_slug: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _interval_candidates_for_slug(dataset_slug)


def _trajectory_save_interval(args, data_source: str | None = None) -> int:
    dataset_slug = _trajectory_dataset_slug(data_source)
    arg_names, env_names = _interval_candidates_for_dataset(dataset_slug)
    raw = None
    raw_source = None
    for name in arg_names:
        value = getattr(args, name, None)
        if value is not None and value != "":
            raw = value
            raw_source = name
            break
    if raw is None:
        for name in env_names:
            value = os.getenv(name)
            if value is not None and value != "":
                raw = value
                raw_source = name
                break
    if raw is None:
        raw = getattr(args, "trajectory_save_interval", None)
        raw_source = "trajectory_save_interval"
    if raw is None or raw == "":
        raw = os.getenv("TRAJECTORY_SAVE_INTERVAL", "1")
        raw_source = "TRAJECTORY_SAVE_INTERVAL"
    value = _optional_int(raw)
    if value is None:
        logger.warning(
            "Invalid trajectory save interval %s=%r for dataset=%s; falling back to 1",
            raw_source,
            raw,
            dataset_slug,
        )
        return 1
    return value


def _should_save_trajectory(run_ctx: RunContext, interval: int) -> bool:
    if interval <= 0:
        return False
    if interval == 1:
        return True
    step = run_ctx.train_step
    if step is None:
        step = run_ctx.rollout_id
    if step is None:
        # No rollout metadata is available, so preserve old save-all behavior.
        return True
    return int(step) % interval == 0


def _trajectory_save_policy() -> str:
    raw = os.getenv("TRAJECTORY_SAVE_POLICY", "step_interval").strip().lower()
    if raw in {"", "legacy", "interval"}:
        return "step_interval"
    if raw in {"task_timeseries", "task-time-series", "task_step", "task-step"}:
        return "task_timeseries"
    logger.warning("Unknown TRAJECTORY_SAVE_POLICY=%r; using step_interval", raw)
    return "step_interval"


def _trajectory_env_int(name: str, default: int) -> int:
    return _env_int(name, default)


def _trajectory_task_save_interval(default_interval: int) -> int:
    raw = os.getenv("TRAJECTORY_TASK_SAVE_INTERVAL", "").strip()
    if not raw:
        return default_interval
    value = _optional_int(raw)
    if value is None:
        logger.warning(
            "Invalid TRAJECTORY_TASK_SAVE_INTERVAL=%r; using %d",
            raw,
            default_interval,
        )
        return default_interval
    return value


def _trajectory_reward_strata() -> set[str]:
    raw = os.getenv("TRAJECTORY_SAVE_REWARD_STRATA", "best,worst")
    values = {
        part.strip().lower()
        for part in raw.split(",")
        if part.strip()
    }
    allowed = {"best", "worst", "latest"}
    unknown = values - allowed
    if unknown:
        logger.warning(
            "Ignoring unknown TRAJECTORY_SAVE_REWARD_STRATA entries: %s",
            sorted(unknown),
        )
    values &= allowed
    return values or {"best", "worst"}


def _trajectory_step_value(run_ctx: RunContext) -> int | None:
    for value in (run_ctx.train_step, run_ctx.rollout_step, run_ctx.rollout_id):
        step = _optional_int(value)
        if step is not None:
            return step
    return None


def _trajectory_task_id(task_spec: TaskSpec) -> str:
    name = str(task_spec.task_name or "unknown")
    path = str(task_spec.task_path or "")
    digest = hashlib.sha1(f"{name}\n{path}".encode("utf-8")).hexdigest()[:8]
    slug = _sanitize_filename(name)[:96].strip("._-") or "unknown"
    return f"{slug}-{digest}"


# Canonical priority when guessing "the" reward value out of a reward dict or
# an index record.  The writer side (shared._sync_reward_aliases) makes
# total_reward the explicit alias of score; keep every reader on this order.
_REWARD_VALUE_KEYS = ("total_reward", "score", "raw_reward", "raw_score", "accuracy")
# Index records carry a numeric "reward" column that wins over the aliases.
_RECORD_REWARD_KEYS = ("reward", "total_reward", "raw_reward", "raw_score")


def _first_finite_reward(mapping: Dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        try:
            if value is None or value == "":
                continue
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def _trajectory_reward_value(reward: Dict[str, Any]) -> float | None:
    return _first_finite_reward(reward, _REWARD_VALUE_KEYS)


def _format_reward_for_filename(value: float | None) -> str:
    if value is None:
        return "na"
    text = f"{value:+.3f}"
    return (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )


def _trajectory_index_path(save_dir: Path) -> Path:
    return save_dir / "index.jsonl"


@contextmanager
def _trajectory_index_lock(save_dir: Path):
    lock_path = save_dir / ".index.lock"
    fh = None
    locked = False
    try:
        fh = lock_path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            locked = True
        except Exception as exc:
            logger.warning("[traj-save] could not lock %s: %s", lock_path, exc)
        yield
    finally:
        if fh is not None:
            if locked:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            fh.close()


def _trajectory_record_dir(save_dir: Path, record: dict[str, Any]) -> Path | None:
    rel_path = record.get("rel_path")
    if rel_path:
        path = save_dir / str(rel_path)
    else:
        raw_path = record.get("path")
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = save_dir / path
    try:
        resolved_root = save_dir.resolve()
        resolved_path = path.resolve()
    except Exception:
        return None
    if resolved_path.parent != resolved_root:
        return None
    return resolved_path


def _trajectory_load_index(save_dir: Path) -> list[dict[str, Any]]:
    index_path = _trajectory_index_path(save_dir)
    if not index_path.exists():
        return []
    active: dict[str, dict[str, Any]] = {}
    try:
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            rel_path = str(record.get("rel_path") or "")
            if not rel_path:
                continue
            event = str(record.get("event") or "save")
            if event == "delete":
                active.pop(rel_path, None)
                continue
            if event == "save":
                record_dir = _trajectory_record_dir(save_dir, record)
                if record_dir is not None and (record_dir / "traj.json").exists():
                    active[rel_path] = record
    except Exception as exc:
        logger.warning("[traj-save] failed reading %s: %s", index_path, exc)
        return []
    return list(active.values())


def _trajectory_append_index(save_dir: Path, record: dict[str, Any]) -> None:
    index_path = _trajectory_index_path(save_dir)
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(record), ensure_ascii=False, default=str))
        fh.write("\n")
    _trajectory_index_cache_update(save_dir, record)


# In-process view of index.jsonl.  Every save used to re-read and re-stat the
# whole index (O(total_records) disk work per rollout sample, thousands of
# entries); within a training run this process is the only writer, so the
# parsed active set is cached and updated incrementally on append.  External
# edits to index.jsonl only take effect after a process restart.
_INDEX_CACHE: dict[str, list[dict[str, Any]]] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def _trajectory_index_cache_update(save_dir: Path, record: dict[str, Any]) -> None:
    key = str(save_dir)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is None:
            return
        event = str(record.get("event") or "save")
        rel_path = str(record.get("rel_path") or "")
        if event == "delete":
            cached[:] = [
                existing
                for existing in cached
                if str(existing.get("rel_path") or "") != rel_path
            ]
        elif event == "save" and rel_path:
            cached.append(dict(record))


def _trajectory_load_index_cached(save_dir: Path) -> list[dict[str, Any]]:
    key = str(save_dir)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return [dict(record) for record in cached]
    records = _trajectory_load_index(save_dir)
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.setdefault(key, [dict(record) for record in records])
    return records


def _trajectory_record_reward(record: dict[str, Any]) -> float | None:
    return _first_finite_reward(record, _RECORD_REWARD_KEYS)


def _trajectory_record_ts(record: dict[str, Any]) -> int:
    value = _optional_int(record.get("ts_ns"))
    if value is not None:
        return value
    value = _optional_int(record.get("created_ts_ns"))
    if value is not None:
        return value
    return 0


def _trajectory_keep_subset(
    records: list[dict[str, Any]],
    limit: int,
    strata: set[str],
) -> set[str]:
    if limit <= 0 or len(records) <= limit:
        return {str(r.get("rel_path")) for r in records if r.get("rel_path")}

    chosen: list[dict[str, Any]] = []

    def add(record: dict[str, Any] | None) -> None:
        if not record or len(chosen) >= limit:
            return
        rel_path = str(record.get("rel_path") or "")
        if rel_path and all(str(r.get("rel_path") or "") != rel_path for r in chosen):
            chosen.append(record)

    latest = max(records, key=_trajectory_record_ts, default=None)
    add(latest)
    reward_records = [
        record for record in records
        if _trajectory_record_reward(record) is not None
    ]
    if "best" in strata and reward_records:
        add(max(reward_records, key=lambda r: _trajectory_record_reward(r) or 0.0))
    if "worst" in strata and reward_records:
        add(min(reward_records, key=lambda r: _trajectory_record_reward(r) or 0.0))
    if "latest" in strata:
        add(latest)
    for record in sorted(records, key=_trajectory_record_ts, reverse=True):
        add(record)
        if len(chosen) >= limit:
            break
    return {str(r.get("rel_path")) for r in chosen if r.get("rel_path")}


def _trajectory_cleanup(
    save_dir: Path,
    active_records: list[dict[str, Any]],
    *,
    task_max_per_step: int,
    task_max_per_task: int,
    max_total: int,
    strata: set[str],
) -> int:
    to_delete: set[str] = set()

    if task_max_per_step > 0:
        by_task_step: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in active_records:
            key = (
                str(record.get("task_id") or record.get("task_name") or "unknown"),
                str(record.get("train_step") if record.get("train_step") is not None else "na"),
            )
            by_task_step.setdefault(key, []).append(record)
        for records in by_task_step.values():
            keep = _trajectory_keep_subset(records, task_max_per_step, strata)
            for record in records:
                rel_path = str(record.get("rel_path") or "")
                if rel_path and rel_path not in keep:
                    to_delete.add(rel_path)

    remaining = [
        record for record in active_records
        if str(record.get("rel_path") or "") not in to_delete
    ]
    if task_max_per_task > 0:
        by_task: dict[str, list[dict[str, Any]]] = {}
        for record in remaining:
            key = str(record.get("task_id") or record.get("task_name") or "unknown")
            by_task.setdefault(key, []).append(record)
        for records in by_task.values():
            keep = _trajectory_keep_subset(records, task_max_per_task, strata)
            for record in records:
                rel_path = str(record.get("rel_path") or "")
                if rel_path and rel_path not in keep:
                    to_delete.add(rel_path)

    remaining = [
        record for record in active_records
        if str(record.get("rel_path") or "") not in to_delete
    ]
    if max_total > 0 and len(remaining) > max_total:
        keep = _trajectory_keep_subset(remaining, max_total, strata | {"latest"})
        for record in remaining:
            rel_path = str(record.get("rel_path") or "")
            if rel_path and rel_path not in keep:
                to_delete.add(rel_path)

    deleted = 0
    for rel_path in sorted(to_delete):
        record = next(
            (r for r in active_records if str(r.get("rel_path") or "") == rel_path),
            None,
        )
        if record is None:
            continue
        target = _trajectory_record_dir(save_dir, record)
        if target is None or not target.exists():
            continue
        try:
            shutil.rmtree(target)
            deleted += 1
            _trajectory_append_index(
                save_dir,
                {
                    "event": "delete",
                    "schema_version": 1,
                    "rel_path": rel_path,
                    "path": str(target),
                    "deleted_ts_ns": time.time_ns(),
                    "reason": "retention_limit",
                },
            )
        except Exception as exc:
            logger.warning("[traj-save] cleanup failed for %s: %s", target, exc)
    return deleted


def _trajectory_save_decision(
    *,
    policy: str,
    run_ctx: RunContext,
    task_id: str,
    reward: float | None,
    interval: int,
    active_records: list[dict[str, Any]],
) -> dict[str, Any]:
    step = _trajectory_step_value(run_ctx)
    decision: dict[str, Any] = {
        "policy": policy,
        "saved": False,
        "reason": "skipped",
        "train_step": step,
        "task_id": task_id,
        "reward": reward,
        "legacy_interval": interval,
    }

    if policy == "step_interval":
        should_save = _should_save_trajectory(run_ctx, interval)
        decision.update(
            {
                "saved": bool(should_save),
                "reason": "legacy_interval" if should_save else "legacy_interval_skip",
            }
        )
        return decision

    if policy != "task_timeseries":
        decision["reason"] = "unknown_policy"
        return decision

    task_interval = _trajectory_task_save_interval(interval)
    max_per_step = _trajectory_env_int("TRAJECTORY_TASK_MAX_PER_STEP", 2)
    max_per_task = _trajectory_env_int("TRAJECTORY_TASK_MAX_PER_TASK", 24)
    max_total = _trajectory_env_int("TRAJECTORY_MAX_TOTAL", 5000)
    strata = _trajectory_reward_strata()
    decision.update(
        {
            "task_save_interval": task_interval,
            "task_max_per_step": max_per_step,
            "task_max_per_task": max_per_task,
            "max_total": max_total,
            "reward_strata": sorted(strata),
        }
    )

    if task_interval <= 0:
        decision["reason"] = "task_interval_disabled"
        return decision
    if step is not None and int(step) % task_interval != 0:
        decision["reason"] = "task_interval_skip"
        return decision

    same_task_step = [
        record for record in active_records
        if str(record.get("task_id") or record.get("task_name") or "unknown") == task_id
        and str(record.get("train_step") if record.get("train_step") is not None else "na")
        == str(step if step is not None else "na")
    ]
    decision["existing_task_step_count"] = len(same_task_step)
    if max_per_step <= 0 or len(same_task_step) < max_per_step:
        decision.update({"saved": True, "reason": "task_step_slot"})
        return decision

    reward_records = [
        record for record in same_task_step
        if _trajectory_record_reward(record) is not None
    ]
    if reward is not None and reward_records:
        rewards = [_trajectory_record_reward(record) for record in reward_records]
        rewards = [value for value in rewards if value is not None]
        if "best" in strata and rewards and reward > max(rewards):
            decision.update({"saved": True, "reason": "task_step_best"})
            return decision
        if "worst" in strata and rewards and reward < min(rewards):
            decision.update({"saved": True, "reason": "task_step_worst"})
            return decision

    decision["reason"] = "task_step_quota"
    return decision


def _attach_trajectory_save_metadata(
    samples: list[Sample],
    sample: Sample,
    metadata: dict[str, Any],
) -> None:
    targets = samples if samples else [sample]
    for target in targets:
        if not isinstance(target.metadata, dict):
            target.metadata = {}
        target.metadata["trajectory_save"] = _jsonable(metadata)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    return str(obj)


def _exploration_audit_from_reward(reward: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(reward, dict):
        return {}
    keys = (
        "explore_mood",
        "explore_mood_code",
        "explore_total_bonus",
        "explore_base_score_before_bonus",
        "explore_bonus_to_base_abs_ratio",
        "explore_curiosity_pressure",
        "explore_tool_intrinsic_pressure",
        "explore_safety_pressure",
        "explore_reward_hacking_risk",
        "explore_over_exploration_risk",
        "explore_safety_tension",
        "explore_action_count",
        "explore_tool_call_count",
        "explore_danger_command_count",
        "explore_parse_error_count",
        "explore_intrinsic_scaled",
        "explore_intrinsic_in_total",
        "explore_lprnd",
        "explore_agent57_enabled",
        "explore_agent57_arm_id",
        "explore_agent57_beta",
        "explore_agent57_combine_mode",
        "explore_agent57_episodic_backend",
        "explore_agent57_controller",
        "explore_agent57_ucb_epsilon",
        "explore_agent57_ucb_min_per_arm",
        "explore_agent57_ucb_value",
        "explore_agent57_ucb_dataset_aware",
        "explore_agent57_ucb_random_seed",
        "explore_agent57_lifelong_enabled",
        "explore_agent57_lifelong_key_version",
        "explore_agent57_lifelong_include_dataset",
        "explore_agent57_lifelong_include_task",
        "explore_agent57_lifelong_include_turn",
        "explore_agent57_lifelong_obs_mode",
        "explore_agent57_lifelong_count_decay",
        "explore_agent57_lifelong_capacity",
        "explore_agent57_trust_gate_mode",
        "explore_agent57_trust",
        "explore_agent57_episodic_action_count",
        "explore_agent57_episodic_empty_bucket_count",
        "explore_agent57_episodic_empty_bucket_rate",
        "explore_agent57_episodic_exact_repeat_count",
        "explore_agent57_episodic_candidate_count_mean",
        "explore_agent57_episodic_probe_count_mean",
        "explore_agent57_episodic_include_turn",
        "explore_agent57_episodic_turn_mode_code",
        "explore_agent57_lifelong_raw",
        "explore_agent57_lifelong_z",
        "explore_agent57_lifelong_stat_n",
        "explore_agent57_lifelong_stat_mean",
        "explore_agent57_lifelong_stat_std",
        "explore_agent57_lifelong_stat_error",
        "explore_agent57_lifelong_bonus",
        "explore_agent57_lifelong_bonus_unclipped",
        "explore_agent57_ngu_episodic_source",
        "explore_agent57_ngu_episodic_reducer",
        "explore_agent57_ngu_life_mod_mode",
        "explore_agent57_ngu_life_mod_std_clip",
        "explore_agent57_ngu_mod_clip",
        "explore_agent57_ngu_episodic",
        "explore_agent57_ngu_life_mod",
        "explore_agent57_intrinsic_signal",
        "explore_agent57_ngu_bonus",
        "explore_agent57_ngu_bonus_unclipped",
        "explore_agent57_bonus_unclipped",
        "explore_agent57_bonus_clipped",
        "explore_agent57_lifelong_eligible",
        "explore_agent57_lifelong_state_update_allowed",
        "explore_agent57_lifelong_suppressed_reason",
        "explore_cde_actor_bonus",
        "explore_cde_actor_log_ppl",
        "explore_cde_actor_reward_gate",
        "explore_cde_actor_eligible",
        "exploration_reward_save_stage",
        "explore_post_norm_bonus_available_at_save",
    )
    return {key: reward[key] for key in keys if key in reward}


def _save_rollout_artifacts(
    *,
    task_spec: TaskSpec,
    run_ctx: RunContext,
    sampling_params: dict,
    sample: Sample,
    samples: List[Sample],
    status: Sample.Status,
    raw_score: float,
    eval_error: str | None,
    turn_records: List[Dict[str, Any]],
    prm_meta: Dict[str, Any] | None,
    prm_coef: float,
    trajectory_save_interval: int = 1,
) -> None:
    """Persist a full rollout (dialogue + tool calls + reward) to disk.

    Side-channel observability export (training consumes the in-memory samples,
    not these files).  Failures are logged & swallowed so training is never
    blocked.
    """
    try:
        save_dir = _get_terminal_save_dir()
        if save_dir is None:
            return

        # Only save trajectories worth analyzing:
        # - Skip if no turns recorded (reset failed, no model output)
        # - Skip if status is FAILED and raw_score is 0 (infra failure, not model failure)
        if not turn_records:
            _attach_trajectory_save_metadata(
                samples,
                sample,
                {
                    "saved": False,
                    "policy": _trajectory_save_policy(),
                    "reason": "no_turns",
                    "train_step": _trajectory_step_value(run_ctx),
                    "rollout_id": run_ctx.rollout_id,
                    "uid": run_ctx.uid,
                },
            )
            return
        if (
            str(status) == "Status.FAILED"
            and raw_score == 0.0
            and len(turn_records) <= 1
            and not _env_bool("TRAJECTORY_SAVE_FAILED_SHORT_ROLLOUTS", False)
        ):
            _attach_trajectory_save_metadata(
                samples,
                sample,
                {
                    "saved": False,
                    "policy": _trajectory_save_policy(),
                    "reason": "infra_failure_short_rollout",
                    "train_step": _trajectory_step_value(run_ctx),
                    "rollout_id": run_ctx.rollout_id,
                    "uid": run_ctx.uid,
                },
            )
            return
        primary_metadata = (
            samples[0].metadata
            if samples and isinstance(samples[0].metadata, dict)
            else (sample.metadata if isinstance(sample.metadata, dict) else {})
        )
        dataset_slug = _trajectory_dataset_slug(primary_metadata.get("data_source"))

        # Build reward breakdown from the first trainable sample (all samples
        # in a rollout share accuracy/raw/base; turn_idx differs per sample).
        reward_breakdown: Dict[str, Any] = {"raw_score": raw_score}
        if samples:
            r0 = samples[0].reward if isinstance(samples[0].reward, dict) else {}
            for k in (
                "accuracy", "raw_score", "base_score", "score",
                "raw_reward", "task_reward", "exploration_reward", "total_reward",
                "prm_turn_score",
                "explore_intrinsic", "explore_intrinsic_scaled",
                "explore_intrinsic_in_total",
                "explore_intrinsic_coef", "explore_intrinsic_effective_coef",
                "explore_intrinsic_schedule", "explore_intrinsic_decay_steps",
                "explore_intrinsic_schedule_multiplier",
                "explore_intrinsic_reducer",
                "explore_intrinsic_granularity", "explore_intrinsic_scope",
                "explore_safety_penalty",
                "explore_lprnd", "explore_lprnd_raw", "explore_lprnd_coef",
                "explore_lprnd_effective_coef", "explore_lprnd_schedule",
                "explore_lprnd_decay_steps", "explore_lprnd_schedule_multiplier",
                "explore_agent57_enabled",
                "explore_agent57_arm_id", "explore_agent57_k",
                "explore_agent57_beta", "explore_agent57_controller",
                "explore_agent57_combine_mode", "explore_agent57_max_bonus",
                "explore_agent57_episodic_backend",
                "explore_agent57_ucb_c", "explore_agent57_ucb_window",
                "explore_agent57_ucb_epsilon",
                "explore_agent57_ucb_min_per_arm",
                "explore_agent57_ucb_value",
                "explore_agent57_ucb_dataset_aware",
                "explore_agent57_ucb_random_seed",
                "explore_agent57_lifelong_enabled",
                "explore_agent57_lifelong_backend",
                "explore_agent57_lifelong_state_path",
                "explore_agent57_lifelong_coef",
                "explore_agent57_lifelong_clip",
                "explore_agent57_lifelong_warmup",
                "explore_agent57_lifelong_count_decay",
                "explore_agent57_lifelong_capacity",
                "explore_agent57_lifelong_key_version",
                "explore_agent57_lifelong_include_dataset",
                "explore_agent57_lifelong_include_task",
                "explore_agent57_lifelong_include_turn",
                "explore_agent57_lifelong_obs_mode",
                "explore_agent57_trust_gate_mode",
                "explore_agent57_trust",
                "explore_agent57_episodic_action_count",
                "explore_agent57_episodic_empty_bucket_count",
                "explore_agent57_episodic_empty_bucket_rate",
                "explore_agent57_episodic_exact_repeat_count",
                "explore_agent57_episodic_candidate_count_mean",
                "explore_agent57_episodic_probe_count_mean",
                "explore_agent57_episodic_include_turn",
                "explore_agent57_episodic_turn_mode_code",
                "explore_agent57_lifelong_raw",
                "explore_agent57_lifelong_z",
                "explore_agent57_lifelong_stat_n",
                "explore_agent57_lifelong_stat_mean",
                "explore_agent57_lifelong_stat_std",
                "explore_agent57_lifelong_stat_error",
                "explore_agent57_lifelong_bonus",
                "explore_agent57_lifelong_bonus_unclipped",
                "explore_agent57_ngu_episodic_source",
                "explore_agent57_ngu_episodic_reducer",
                "explore_agent57_ngu_life_mod_mode",
                "explore_agent57_ngu_life_mod_std_clip",
                "explore_agent57_ngu_mod_clip",
                "explore_agent57_ngu_episodic",
                "explore_agent57_ngu_life_mod",
                "explore_agent57_intrinsic_signal",
                "explore_agent57_ngu_bonus",
                "explore_agent57_ngu_bonus_unclipped",
                "explore_agent57_bonus_unclipped",
                "explore_agent57_bonus_clipped",
                "explore_agent57_lifelong_unique_keys",
                "explore_agent57_lifelong_seen_before",
                "explore_agent57_lifelong_warmup_remaining",
                "explore_agent57_lifelong_eligible",
                "explore_agent57_lifelong_state_update_allowed",
                "explore_agent57_lifelong_suppressed_reason",
                "explore_cde_actor_bonus",
                "explore_cde_actor_log_ppl", "explore_cde_actor_omega",
                "explore_cde_actor_alpha", "explore_cde_actor_kappa",
                "explore_cde_actor_reward_gate", "explore_cde_actor_eligible",
                "explore_cde_actor_decay_steps",
                "explore_cde_actor_base_mean", "explore_cde_actor_base_magnitude",
                "explore_cde_actor_cap",
                "explore_cde_actor_scaled",
                "explore_cde_actor_clipped", "explore_total_bonus",
                "explore_all_bonus", "explore_score_bonus_components",
                "explore_base_score_before_bonus",
                "explore_bonus_to_base_abs_ratio",
                "explore_curiosity_pressure",
                "explore_tool_intrinsic_pressure",
                "explore_safety_pressure",
                "explore_mood", "explore_mood_code",
                "explore_reward_hacking_risk",
                "explore_over_exploration_risk",
                "explore_safety_tension",
                "explore_turn_count", "explore_tool_call_count",
                "explore_action_count", "explore_danger_command_count",
                "explore_parse_error_count",
                "dapo_overlong_reward", "dapo_overlong",
                "dapo_overlong_expected_len", "dapo_overlong_buffer_len",
            ):
                if k in r0:
                    reward_breakdown[k] = r0[k]
            reward_details = (
                samples[0].metadata.get("reward_details")
                if isinstance(samples[0].metadata, dict)
                else None
            )
            if reward_details:
                reward_breakdown["details"] = reward_details
            if (
                reward_breakdown.get("explore_agent57_enabled")
                and "explore_post_norm_bonus" not in reward_breakdown
            ):
                reward_breakdown["exploration_reward_save_stage"] = (
                    "generate_pre_reward_postprocess"
                )
                reward_breakdown["explore_post_norm_bonus_available_at_save"] = False
            reward_breakdown["per_turn_scores"] = [
                {
                    "turn_idx": s.metadata.get("turn_idx"),
                    "score": (s.reward or {}).get("score"),
                    "prm_turn_score": (s.reward or {}).get("prm_turn_score"),
                }
                for s in samples
            ]
        primary_reward_details = primary_metadata.get("reward_details")
        primary_reward_reason = (
            primary_reward_details.get("reason")
            if isinstance(primary_reward_details, dict)
            else None
        )
        task_id = _trajectory_task_id(task_spec)
        policy = _trajectory_save_policy()
        reward_value = _trajectory_reward_value(reward_breakdown)
        with _trajectory_index_lock(save_dir):
            active_records = _trajectory_load_index_cached(save_dir)
            save_decision = _trajectory_save_decision(
                policy=policy,
                run_ctx=run_ctx,
                task_id=task_id,
                reward=reward_value,
                interval=trajectory_save_interval,
                active_records=active_records,
            )
        if not save_decision.get("saved"):
            decision_metadata = {
                **save_decision,
                "dataset_slug": dataset_slug,
                "task_name": task_spec.task_name,
                "task_path": task_spec.task_path,
                "rollout_id": run_ctx.rollout_id,
                "group_index": run_ctx.group_index,
                "sample_index": run_ctx.sample_index,
                "uid": run_ctx.uid,
            }
            _attach_trajectory_save_metadata(samples, sample, decision_metadata)
            if _env_bool("TRAJECTORY_SAVE_LOG_DECISIONS", False):
                logger.info(
                    "[traj-save] skipped task=%s step=%s policy=%s reason=%s",
                    task_spec.task_name,
                    save_decision.get("train_step"),
                    policy,
                    save_decision.get("reason"),
                )
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        ts_ns = time.time_ns()
        step_for_name = _trajectory_step_value(run_ctx)
        reward_for_name = _format_reward_for_filename(reward_value)
        uid = str(run_ctx.uid or uuid.uuid4().hex)
        stem = (
            f"{dataset_slug}_task-{_sanitize_filename(task_id)[:120]}"
            f"_iter{step_for_name if step_for_name is not None else 'na'}"
            f"_rew{reward_for_name}"
            f"_r{run_ctx.rollout_id if run_ctx.rollout_id is not None else 'na'}"
            f"_g{run_ctx.group_index if run_ctx.group_index is not None else 'na'}"
            f"_s{run_ctx.sample_index if run_ctx.sample_index is not None else 'na'}"
            f"_{uid[:8]}"
            f"_{ts}"
        )
        run_dir = save_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        decision_metadata = {
            **save_decision,
            "dataset_slug": dataset_slug,
            "task_name": task_spec.task_name,
            "task_path": task_spec.task_path,
            "rollout_id": run_ctx.rollout_id,
            "group_index": run_ctx.group_index,
            "sample_index": run_ctx.sample_index,
            "uid": uid,
            "path": str(run_dir),
            "rel_path": run_dir.name,
            "traj_path": str(run_dir / "traj.json"),
            "meta_path": str(run_dir / "meta.json"),
        }

        traj_payload = {
            "trajectory_format": "openclaw-terminal-rl-1",
            "info": {
                "task_id": task_id,
                "task_name": task_spec.task_name,
                "task_path": task_spec.task_path,
                "data_source": primary_metadata.get("data_source"),
                "dataset_slug": dataset_slug,
                "safety_split": primary_metadata.get("safety_split"),
                "reward_reason": primary_reward_reason,
                "uid": run_ctx.uid,
                "group_index": run_ctx.group_index,
                "sample_index": run_ctx.sample_index,
                "rollout_id": run_ctx.rollout_id,
                "train_step": run_ctx.train_step,
                "rollout_step": run_ctx.rollout_step,
                "status": str(status),
                "num_turns": len(turn_records),
                "eval_error": eval_error,
                "prm_coef": prm_coef,
                "trajectory_save_interval": trajectory_save_interval,
                "trajectory_save_policy": policy,
                "trajectory_save_reason": save_decision.get("reason"),
                "trajectory_save": _jsonable(decision_metadata),
                "trajectory_uncertainty": _jsonable(
                    primary_metadata.get("trajectory_uncertainty")
                ),
            },
            "turns": _jsonable(turn_records),
            "reward": _jsonable(reward_breakdown),
            "exploration": _jsonable(_exploration_audit_from_reward(reward_breakdown)),
            "prm": _jsonable(prm_meta) if prm_meta else None,
        }
        (run_dir / "traj.json").write_text(
            json.dumps(traj_payload, ensure_ascii=False, indent=2, default=str)
        )

        meta_payload = {
            "task_id": task_id,
            "task_name": task_spec.task_name,
            "task_path": task_spec.task_path,
            "instruction": task_spec.instruction,
            "uid": run_ctx.uid,
            "group_index": run_ctx.group_index,
            "sample_index": run_ctx.sample_index,
            "rollout_id": run_ctx.rollout_id,
            "train_step": run_ctx.train_step,
            "rollout_step": run_ctx.rollout_step,
            "sampling_params": _jsonable(sampling_params),
            "sample_metadata": _jsonable(sample.metadata or {}),
            "sample_prompt": _jsonable(sample.prompt),
            "data_source": primary_metadata.get("data_source"),
            "dataset_slug": dataset_slug,
            "safety_split": primary_metadata.get("safety_split"),
            "reward_details": _jsonable(primary_reward_details),
            "exploration": _jsonable(_exploration_audit_from_reward(reward_breakdown)),
            "trajectory_uncertainty": _jsonable(
                primary_metadata.get("trajectory_uncertainty")
            ),
            "status": str(status),
            "raw_score": raw_score,
            "dataset": primary_metadata.get("data_source"),
            "raw_reward": reward_breakdown.get("raw_reward", reward_breakdown.get("raw_score")),
            "task_reward": reward_breakdown.get("task_reward", reward_breakdown.get("base_score")),
            "exploration_reward": reward_breakdown.get("exploration_reward", 0.0),
            "total_reward": reward_breakdown.get("total_reward", reward_breakdown.get("score")),
            "trajectory_save_interval": trajectory_save_interval,
            "trajectory_save_policy": policy,
            "trajectory_save_reason": save_decision.get("reason"),
            "trajectory_save": _jsonable(decision_metadata),
            "ts_ns": ts_ns,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str)
        )
        cleanup_deleted = 0
        index_record = {
            "event": "save",
            "schema_version": 1,
            "path": str(run_dir),
            "rel_path": run_dir.name,
            "traj_path": str(run_dir / "traj.json"),
            "meta_path": str(run_dir / "meta.json"),
            "task_id": task_id,
            "task_name": task_spec.task_name,
            "task_path": task_spec.task_path,
            "data_source": primary_metadata.get("data_source"),
            "dataset_slug": dataset_slug,
            "safety_split": primary_metadata.get("safety_split"),
            "uid": uid,
            "group_index": run_ctx.group_index,
            "sample_index": run_ctx.sample_index,
            "rollout_id": run_ctx.rollout_id,
            "train_step": _trajectory_step_value(run_ctx),
            "rollout_step": run_ctx.rollout_step,
            "status": str(status),
            "num_turns": len(turn_records),
            "reward": reward_value,
            "raw_score": reward_breakdown.get("raw_score"),
            "raw_reward": reward_breakdown.get("raw_reward"),
            "task_reward": reward_breakdown.get("task_reward", reward_breakdown.get("base_score")),
            "exploration_reward": reward_breakdown.get("exploration_reward", 0.0),
            "total_reward": reward_breakdown.get("total_reward", reward_breakdown.get("score")),
            "policy": policy,
            "decision_reason": save_decision.get("reason"),
            "created_at": ts,
            "ts_ns": ts_ns,
        }
        with _trajectory_index_lock(save_dir):
            _trajectory_append_index(save_dir, index_record)
            if policy == "task_timeseries":
                cleanup_deleted = _trajectory_cleanup(
                    save_dir,
                    _trajectory_load_index_cached(save_dir),
                    task_max_per_step=_trajectory_env_int("TRAJECTORY_TASK_MAX_PER_STEP", 2),
                    task_max_per_task=_trajectory_env_int("TRAJECTORY_TASK_MAX_PER_TASK", 24),
                    max_total=_trajectory_env_int("TRAJECTORY_MAX_TOTAL", 5000),
                    strata=_trajectory_reward_strata(),
                )
        decision_metadata["cleanup_deleted_count"] = cleanup_deleted
        _attach_trajectory_save_metadata(samples, sample, decision_metadata)
        logger.info(
            "[traj-save] wrote %s (turns=%d policy=%s reason=%s reward=%s cleanup_deleted=%d)",
            run_dir,
            len(turn_records),
            policy,
            save_decision.get("reason"),
            reward_value,
            cleanup_deleted,
        )
    except Exception as exc:
        logger.warning(
            "[traj-save] failed for task=%s uid=%s: %s",
            task_spec.task_name, run_ctx.uid, exc,
        )
