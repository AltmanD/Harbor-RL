from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from slime.utils.types import Sample

from harborrl.types import Interaction, RunContext
from harborrl.algorithms.dive_po.exploration.agent57.memory import create_episodic_memory_backend
from harborrl.algorithms.dive_po.exploration.agent57.controller import (
    coarse_observation_fingerprint as _agent57_coarse_observation_fingerprint,
    coarse_observation_label as _agent57_coarse_observation_label,
    config_from_env as _agent57_config_from_env,
    exit_code_bucket as _agent57_exit_code_bucket,
)

logger = logging.getLogger(__name__)
_AGENT57_CONFIG = _agent57_config_from_env()


from harborrl.env import (
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
)


# ── Exploration: count-based intrinsic reward (MERCI simplified) ──────────────
_EXPLORE_INTRINSIC_ENABLED = _env_bool("EXPLORE_INTRINSIC_ENABLED", False)
_EXPLORE_INTRINSIC_COEF = _env_float("EXPLORE_INTRINSIC_COEF", 0.1)
_EXPLORE_INTRINSIC_SCHEDULE = os.getenv("EXPLORE_INTRINSIC_SCHEDULE", "constant").strip().lower()
_EXPLORE_INTRINSIC_DECAY_STEPS = _env_int("EXPLORE_INTRINSIC_DECAY_STEPS", 0)
_EXPLORE_INTRINSIC_REDUCER = os.getenv("EXPLORE_INTRINSIC_REDUCER", "sum").strip().lower()
if _EXPLORE_INTRINSIC_REDUCER not in {"sum", "mean"}:
    _EXPLORE_INTRINSIC_REDUCER = "sum"
_EXPLORE_SCORE_BONUS_COMPONENTS = os.getenv("EXPLORE_SCORE_BONUS_COMPONENTS", "legacy").strip().lower()
# Granularity for novelty hashing:
#   "raw"        = full command string (default, matches v1)
#   "signature"  = tool-call signature (cmd name + first 2 args), Agent57-style
#                  sub-goal/skill granularity per the LaMer/Agent57 analysis.
_EXPLORE_INTRINSIC_GRANULARITY = os.getenv("EXPLORE_INTRINSIC_GRANULARITY", "raw").strip().lower()
_EXPLORE_INTRINSIC_SCOPE = os.getenv("EXPLORE_INTRINSIC_SCOPE", "process").strip().lower()
_EXPLORE_AGENT57_EPISODIC_OBS_MODE = (
    os.getenv(
        "EXPLORE_AGENT57_EPISODIC_OBS_MODE",
        os.getenv("EXPLORE_AGENT57_LIFELONG_OBS_MODE", "fingerprint"),
    )
    .strip()
    .lower()
)
if _EXPLORE_AGENT57_EPISODIC_OBS_MODE not in {"fingerprint", "label", "none"}:
    _EXPLORE_AGENT57_EPISODIC_OBS_MODE = "fingerprint"
_EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN = _env_bool(
    "EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN",
    True,
)
_EXPLORE_AGENT57_EPISODIC_TURN_MODE = (
    os.getenv("EXPLORE_AGENT57_EPISODIC_TURN_MODE", "bucket").strip().lower()
)
if _EXPLORE_AGENT57_EPISODIC_TURN_MODE in {"", "1", "true", "yes", "on", "coarse"}:
    _EXPLORE_AGENT57_EPISODIC_TURN_MODE = "bucket"
elif _EXPLORE_AGENT57_EPISODIC_TURN_MODE in {"0", "false", "no", "off"}:
    _EXPLORE_AGENT57_EPISODIC_TURN_MODE = "none"
elif _EXPLORE_AGENT57_EPISODIC_TURN_MODE in {"stage"}:
    _EXPLORE_AGENT57_EPISODIC_TURN_MODE = "phase"
elif _EXPLORE_AGENT57_EPISODIC_TURN_MODE not in {"none", "bucket", "phase"}:
    logger.warning(
        "Invalid EXPLORE_AGENT57_EPISODIC_TURN_MODE=%r; using bucket",
        _EXPLORE_AGENT57_EPISODIC_TURN_MODE,
    )
    _EXPLORE_AGENT57_EPISODIC_TURN_MODE = "bucket"
if not _EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN:
    _EXPLORE_AGENT57_EPISODIC_TURN_MODE = "none"
_CMD_COUNTER: Dict[str, int] = {}  # process-level counter for command novelty
_AGENT57_LAST_EPISODIC_STATS: Dict[str, float] = {}
_AGENT57_LAST_EPISODIC_BY_TURN: Dict[int, float] = {}


def _agent57_last_episodic_stats() -> Dict[str, float]:
    """Return the latest episode summary without exposing mutable module state."""
    return dict(_AGENT57_LAST_EPISODIC_STATS)


def _agent57_last_episodic_by_turn() -> Dict[int, float]:
    """Return the latest per-turn novelty values."""
    return dict(_AGENT57_LAST_EPISODIC_BY_TURN)

# ── Exploration: LP-RND lifelong novelty (草案 C, zero-extra-param) ───────────
# Reuses the rollout_log_probs already computed by slime (no extra forward pass).
# Bonus is proportional to how surprised the *current* policy is by the trajectory:
# higher mean negative-logprob → more novel → larger bonus, clipped to [0, L].
# This is the LLM analog of RND: "how surprising is this trajectory under the
# current rollout policy?" implemented without maintaining a separate net.
_EXPLORE_LPRND_ENABLED = _env_bool("EXPLORE_LPRND_ENABLED", False)
_EXPLORE_LPRND_COEF = _env_float("EXPLORE_LPRND_COEF", 0.05)
_EXPLORE_LPRND_SCHEDULE = os.getenv("EXPLORE_LPRND_SCHEDULE", "constant").strip().lower()
_EXPLORE_LPRND_DECAY_STEPS = _env_int("EXPLORE_LPRND_DECAY_STEPS", 0)
_EXPLORE_LPRND_CLIP = _env_float("EXPLORE_LPRND_CLIP", 3.0)
_EXPLORE_LPRND_WARMUP = _env_int("EXPLORE_LPRND_WARMUP", 32)
# Running stats for normalization (process-level, updated online).
_LPRND_STATS = {"warmup": 0, "n": 0, "mean": 0.0, "m2": 0.0}

# ── T2PO-style turn uncertainty diagnostics ─────────────────────────────────
# Logging-only. This does not alter sampling or rewards. T2PO's original turn
# score uses logits entropy + max-logprob during generation; HarborRL currently
# persists sampled-token log-probs, so this records a mean-logprob proxy.
_TURN_UNCERTAINTY_SCHEMA = "lightrl.t2po_turn_uncertainty"
_TURN_UNCERTAINTY_SCHEMA_VERSION = 1
_TURN_UNCERTAINTY_ENABLED = _env_bool("T2PO_TURN_UNCERTAINTY_LOGGING", True)
_TURN_UNCERTAINTY_WARMUP_TOKENS = max(
    0, _env_int("T2PO_TURN_UNCERTAINTY_WARMUP_TOKENS", 0)
)
_TURN_UNCERTAINTY_FINGERPRINT_TOKENS = max(
    1, _env_int("T2PO_TURN_UNCERTAINTY_FINGERPRINT_TOKENS", 32)
)
_TURN_LOW_PROGRESS_THRESHOLD = max(
    0.0, _env_float("T2PO_TURN_LOW_PROGRESS_THRESHOLD", 0.3)
)

# ── Exploration: CDE actor curiosity bonus (RLVR PPL bonus) ──────────────────
# Optional actor-side Curiosity-Driven Exploration bonus:
#   B_actor(q,o) = -mean_t log pi(o_t | o_<t, q)
#   r_hat = r + omega * min(|r| / kappa, alpha * B_actor)
#
# The cap is based on the pre-exploration task reward magnitude. That keeps this
# as a supplement to verifiable rewards and prevents empty/infra-failed rollouts
# with score=0 from receiving curiosity reward.
_EXPLORE_CDE_ACTOR_ENABLED = (
    os.getenv("EXPLORE_CDE_ACTOR_ENABLED", os.getenv("EXPLORE_CDE_ACTOR", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)
_EXPLORE_CDE_ACTOR_OMEGA = _env_float("EXPLORE_CDE_ACTOR_OMEGA", 0.05)
_EXPLORE_CDE_ACTOR_KAPPA = _env_float("EXPLORE_CDE_ACTOR_KAPPA", 2.0)
_EXPLORE_CDE_ACTOR_ALPHA = _env_float("EXPLORE_CDE_ACTOR_ALPHA", 0.1)
_EXPLORE_CDE_ACTOR_DECAY_STEPS = _env_int("EXPLORE_CDE_ACTOR_DECAY_STEPS", 0)
_EXPLORE_CDE_ACTOR_REWARD_GATE = os.getenv(
    "EXPLORE_CDE_ACTOR_REWARD_GATE", "nonzero"
).strip().lower()

# ── Exploration: multi-attempt reflection (LaMer-style) ───────────────────────
# When EXPLORE_RETRY_ATTEMPTS > 1, a failed rollout is annotated with a reflection
# block + traj_gamma reward shaping. Note: actual env restart is handled inside
# agent_runner; we surface the knobs here so the wrapper script and rollout_log
# can pick them up. Defaults (1 / 1.0) preserve baseline behavior.
_EXPLORE_RETRY_ATTEMPTS = _env_int("EXPLORE_RETRY_ATTEMPTS", 1)
_EXPLORE_RETRY_TRAJ_GAMMA = _env_float("EXPLORE_RETRY_TRAJ_GAMMA", 1.0)

# ── Exploration: safety pre-filter ────────────────────────────────────────────
_EXPLORE_SAFETY_FILTER_ENABLED = _env_bool("EXPLORE_SAFETY_FILTER_ENABLED", False)
_EXPLORE_SAFETY_FILTER_COEF = _env_float("EXPLORE_SAFETY_FILTER_COEF", -0.5)
_DANGER_RE = re.compile(
    r"rm\s+-[rfRF]+\s+/(?:\s|$)|"          # rm -rf /
    r"(?:curl|wget)[^|;]+\|\s*(?:bash|sh)|"  # curl|bash, wget|sh
    r"chmod\s+(?:0?7{2,3})\s+/|"             # chmod 777 /
    r">\s*/etc/(?:passwd|shadow|sudoers)|"
    r"cat\s+/etc/shadow|"
    r"eval\s+.*(?:rm\s+-[rfRF]+\s+/|curl|wget)|"
    r"(?:`|\$\()[^`)]*(?:rm\s+-[rfRF]+\s+/|curl|wget)[^`)]*(?:`|\))|"
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;:",         # fork bomb
    re.IGNORECASE,
)


def _cmd_signature(cmd: str) -> str:
    """Skill-level signature of a command (cmd name + first 2 args) for novelty hashing.

    'signature' granularity reduces hash-collision-by-paraphrase: e.g., `ls -la /tmp`
    and `ls -al /tmp/` map to the same skill bucket, while `ls -la /etc` is distinct.
    This is the sub-goal granularity proposed in the Agent57→Agentic-RL migration analysis.
    """
    import shlex
    if not cmd or not cmd.strip():
        return "__empty__"

    def _normalize_part(part: str) -> str:
        part = part.strip()
        if len(part) > 2 and part.startswith("-") and not part.startswith("--"):
            # Normalize common short-flag permutations: -al and -la -> -al.
            return "-" + "".join(sorted(part[1:]))
        if part != "/" and "/" in part:
            return part.rstrip("/")
        return part

    try:
        parts = [_normalize_part(p) for p in shlex.split(cmd)[:3]]
        return "|".join(parts) if parts else "__empty__"
    except Exception:
        return cmd[:80]


def _stable_json(value: Any, limit: int = 512) -> str:
    try:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit]


def _explore_len_bucket(text: str) -> str:
    size = len(text)
    if size == 0:
        return "len0"
    if size < 80:
        return "lenS"
    if size < 512:
        return "lenM"
    if size < 2048:
        return "lenL"
    return "lenXL"


def _explore_path_signature(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"/+", "/", text)
    if text != "/":
        text = text.rstrip("/")
    return text[:160]


def _explore_structured_tool_signature(
    tool_name: str,
    args: Any,
) -> tuple[str, str]:
    """Return a compact signature/family for non-command structured tools.

    Avoid hashing full payloads such as file contents into exploration keys. The
    key should capture the operation and target, not every byte written.
    """
    tool = str(tool_name or "tool").strip() or "tool"
    if not isinstance(args, dict):
        args_text = _stable_json(args)
        return f"{tool}|{args_text[:160]}", f"{tool}:structured"

    path_value = None
    for key in (
        "file_path",
        "path",
        "target_path",
        "filename",
        "dest",
        "destination",
        "repo_path",
    ):
        if args.get(key):
            path_value = args.get(key)
            break
    if path_value is not None:
        path = _explore_path_signature(path_value)
        ext = Path(path).suffix[:16] or "noext"
        parts = [tool, f"path:{path}", f"ext:{ext}"]
        if "content" in args:
            parts.append(f"content:{_explore_len_bucket(str(args.get('content') or ''))}")
        return "|".join(parts), f"{tool}:file"

    stable_keys = []
    for key in ("query", "url", "package", "name", "id"):
        value = args.get(key)
        if value:
            stable_keys.append(f"{key}:{str(value)[:80]}")
    if stable_keys:
        return "|".join([tool, *stable_keys]), f"{tool}:structured"

    return f"{tool}|schema:{','.join(sorted(str(k) for k in args.keys()))[:120]}", f"{tool}:structured"


def _explore_turn_bucket(turn_idx: Any) -> str:
    if _EXPLORE_AGENT57_EPISODIC_TURN_MODE == "none":
        return "turn_ignored"
    try:
        idx = int(turn_idx)
    except (TypeError, ValueError):
        return "turn_unknown"
    if _EXPLORE_AGENT57_EPISODIC_TURN_MODE == "phase":
        if idx <= 0:
            return "phase_open"
        if idx <= 2:
            return "phase_probe"
        if idx <= 5:
            return "phase_work"
        return "phase_late"
    if idx <= 0:
        return "turn0"
    if idx <= 2:
        return "turn1_2"
    if idx <= 5:
        return "turn3_5"
    return "turn6p"


def _explore_observation_bucket(value: Any, mode: str) -> str:
    if mode == "none":
        return "obs_ignored"
    if mode == "label":
        return _agent57_coarse_observation_label(value)
    return _agent57_coarse_observation_fingerprint(value)


def _iter_explore_actions(turn_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract action strings used by intrinsic reward and safety diagnostics.

    Older code looked only at turn["command"], but current terminal-rl trajectories
    store most actions as structured tool_calls. Missing those calls makes command
    novelty and danger filtering silently no-op for real rollouts.
    """
    actions: List[Dict[str, Any]] = []
    for tr in turn_records or []:
        turn_idx = tr.get("turn_idx")
        legacy_cmd = str(tr.get("command", "") or "").strip()
        if legacy_cmd:
            result = tr.get("result") or tr.get("observation") or tr.get("output")
            actions.append(
                {
                    "tool_name": "shell",
                    "raw": legacy_cmd,
                    "signature": f"shell|{_cmd_signature(legacy_cmd)}",
                    "danger_text": legacy_cmd,
                    "turn_idx": str(turn_idx) if turn_idx is not None else "",
                    "turn_bucket": _explore_turn_bucket(turn_idx),
                    "result": result,
                    "obs_bucket": _explore_observation_bucket(
                        result,
                        _EXPLORE_AGENT57_EPISODIC_OBS_MODE,
                    ),
                    "exit_bucket": _agent57_exit_code_bucket(result),
                }
            )

        for call in tr.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name") or call.get("name") or "tool").strip() or "tool"
            args = call.get("args")
            if args is None:
                args = call.get("arguments")
            command_text = ""
            if isinstance(args, dict):
                for key in ("command", "cmd", "script", "code"):
                    value = args.get(key)
                    if value:
                        command_text = str(value).strip()
                        break
            elif args is not None:
                command_text = str(args).strip()

            args_text = _stable_json(args)
            raw = f"{tool_name}:{command_text or args_text}"
            if command_text:
                signature = f"{tool_name}|{_cmd_signature(command_text)}"
                action_family = ""
            else:
                signature, action_family = _explore_structured_tool_signature(
                    tool_name,
                    args,
                )
            result = call.get("result")
            if result is None:
                result = call.get("observation") or call.get("output")
            actions.append(
                {
                    "tool_name": tool_name,
                    "raw": raw,
                    "signature": signature,
                    "action_family": action_family,
                    "danger_text": command_text or args_text,
                    "turn_idx": str(turn_idx) if turn_idx is not None else "",
                    "turn_bucket": _explore_turn_bucket(turn_idx),
                    "result": result,
                    "obs_bucket": _explore_observation_bucket(
                        result,
                        _EXPLORE_AGENT57_EPISODIC_OBS_MODE,
                    ),
                    "exit_bucket": _agent57_exit_code_bucket(result),
                }
            )
    return actions


def _explore_agent57_episodic_state(action: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "tool": str(action.get("tool_name") or "tool"),
        "signature": str(action.get("signature") or action.get("raw") or "unknown"),
    }
    if _EXPLORE_AGENT57_EPISODIC_OBS_MODE != "none":
        state["observation"] = str(action.get("obs_bucket") or "no_result")
        state["exit"] = str(action.get("exit_bucket") or "exit_unknown")
    if _EXPLORE_AGENT57_EPISODIC_TURN_MODE != "none":
        state["turn_mode"] = _EXPLORE_AGENT57_EPISODIC_TURN_MODE
        state["turn"] = str(action.get("turn_bucket") or "turn_unknown")
    return state


def _explore_intrinsic_bonus(turn_records: List[Dict[str, Any]]) -> float:
    """Sum of 1/sqrt(count) bonuses for unique commands (MERCI-style).

    Granularity controlled by EXPLORE_INTRINSIC_GRANULARITY env var:
      - "raw"       : full command text (default, v1 behavior)
      - "signature" : cmd name + first 2 args (skill-level, Agent57-style)

    Scope controlled by EXPLORE_INTRINSIC_SCOPE:
      - "process" : historical behavior, process-local counter across rollouts
      - "episode" : reset counts per rollout; lower-risk under multi-process Ray
    """
    if not _EXPLORE_INTRINSIC_ENABLED or not turn_records:
        return 0.0
    total = 0.0
    action_count = 0
    episode_counter: Dict[str, int] = {}
    for action in _iter_explore_actions(turn_records):
        action_count += 1
        if _EXPLORE_INTRINSIC_GRANULARITY == "signature":
            key_src = action["signature"]
        else:
            key_src = action["raw"]
        key = hashlib.md5(key_src.encode()).hexdigest()[:10]
        if _EXPLORE_INTRINSIC_SCOPE == "episode":
            # Bug fix / robustness: process-level counters diverge across Ray
            # rollout workers. Episode scope gives deterministic within-rollout
            # novelty and is the default for the robust_dapo_lite preset.
            episode_counter[key] = episode_counter.get(key, 0) + 1
            total += 1.0 / math.sqrt(episode_counter[key])
        else:
            _CMD_COUNTER[key] = _CMD_COUNTER.get(key, 0) + 1
            total += 1.0 / math.sqrt(_CMD_COUNTER[key])
    if _EXPLORE_INTRINSIC_REDUCER == "mean" and action_count > 0:
        return total / action_count
    return total


def _explore_episode_signature_novelty(
    turn_records: List[Dict[str, Any]],
    *,
    reducer: str = "sum",
) -> float:
    """Episode-local novelty used by Agent57 NGU-lite product mode."""
    global _AGENT57_LAST_EPISODIC_STATS, _AGENT57_LAST_EPISODIC_BY_TURN
    _AGENT57_LAST_EPISODIC_STATS = {}
    _AGENT57_LAST_EPISODIC_BY_TURN = {}
    if not turn_records:
        return 0.0
    total = 0.0
    action_count = 0
    episode_counter: Dict[str, int] = {}
    turn_total: Dict[int, float] = {}
    turn_count: Dict[int, int] = {}
    episodic_memory = create_episodic_memory_backend(_AGENT57_CONFIG.episodic_backend)
    empty_bucket_count = 0.0
    exact_repeat_count = 0.0
    candidate_count_total = 0.0
    probe_count_total = 0.0
    for action in _iter_explore_actions(turn_records):
        action_count += 1
        try:
            turn_idx = int(action.get("turn_idx", -1))
        except (TypeError, ValueError):
            turn_idx = -1
        if episodic_memory is not None:
            state = _explore_agent57_episodic_state(action)
            novelty = float(episodic_memory.compute_novelty(state))
            total += novelty
            turn_total[turn_idx] = turn_total.get(turn_idx, 0.0) + novelty
            turn_count[turn_idx] = turn_count.get(turn_idx, 0) + 1
            query_stats_fn = getattr(episodic_memory, "last_query_stats", None)
            query_stats = query_stats_fn() if callable(query_stats_fn) else {}
            empty_bucket_count += float(query_stats.get("empty_bucket", 0.0) or 0.0)
            exact_repeat_count += float(query_stats.get("exact_repeat", 0.0) or 0.0)
            candidate_count_total += float(query_stats.get("candidate_count", 0.0) or 0.0)
            probe_count_total += float(query_stats.get("probe_count", 0.0) or 0.0)
            episodic_memory.add(state)
            continue
        key_src = _stable_json(_explore_agent57_episodic_state(action))
        key = hashlib.md5(key_src.encode()).hexdigest()[:10]
        episode_counter[key] = episode_counter.get(key, 0) + 1
        novelty = 1.0 / math.sqrt(episode_counter[key])
        total += novelty
        turn_total[turn_idx] = turn_total.get(turn_idx, 0.0) + novelty
        turn_count[turn_idx] = turn_count.get(turn_idx, 0) + 1
    value = total / action_count if reducer == "mean" and action_count > 0 else total
    if action_count > 0:
        if reducer == "mean":
            _AGENT57_LAST_EPISODIC_BY_TURN = {
                idx: turn_total[idx] / max(1, turn_count.get(idx, 0))
                for idx in turn_total
                if idx >= 0
            }
        else:
            _AGENT57_LAST_EPISODIC_BY_TURN = {
                idx: turn_total[idx]
                for idx in turn_total
                if idx >= 0
            }
        _AGENT57_LAST_EPISODIC_STATS = {
            "explore_agent57_episodic_action_count": float(action_count),
            "explore_agent57_episodic_empty_bucket_count": float(empty_bucket_count),
            "explore_agent57_episodic_empty_bucket_rate": float(empty_bucket_count / action_count),
            "explore_agent57_episodic_exact_repeat_count": float(exact_repeat_count),
            "explore_agent57_episodic_candidate_count_mean": float(candidate_count_total / action_count),
            "explore_agent57_episodic_probe_count_mean": float(probe_count_total / action_count),
            "explore_agent57_episodic_include_turn": float(
                _EXPLORE_AGENT57_EPISODIC_TURN_MODE != "none"
            ),
            "explore_agent57_episodic_turn_mode_code": float(
                {"none": 0, "bucket": 1, "phase": 2}.get(
                    _EXPLORE_AGENT57_EPISODIC_TURN_MODE,
                    1,
                )
            ),
        }
    return value


def _explore_score_bonus_from_components(
    components_raw: str,
    *,
    intrinsic: float,
    safety: float,
    lprnd: float,
    agent57: float,
    cde_actor: float,
) -> float:
    """Select which exploration components are injected into reward["score"]."""
    raw = (components_raw or "").strip().lower()
    if raw in {"", "none", "off", "0"}:
        return 0.0
    values = {
        "intrinsic": intrinsic,
        "explore_intrinsic_scaled": intrinsic,
        "safety": safety,
        "explore_safety_penalty": safety,
        "lprnd": lprnd,
        "explore_lprnd": lprnd,
        "agent57": agent57,
        "ngu": agent57,
        "explore_agent57_ngu_bonus": agent57,
        "cde": cde_actor,
        "cde_actor": cde_actor,
        "explore_cde_actor_bonus": cde_actor,
    }
    if raw == "legacy":
        return intrinsic + safety + lprnd + agent57 + cde_actor
    total = 0.0
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        total += values.get(key, 0.0)
    return total


def _explore_safety_penalty(turn_records: List[Dict[str, Any]]) -> float:
    """Negative penalty if any turn matched a danger pattern."""
    if not _EXPLORE_SAFETY_FILTER_ENABLED or not turn_records:
        return 0.0
    pen = 0.0
    for action in _iter_explore_actions(turn_records):
        danger_text = action.get("danger_text", "")
        if danger_text and _DANGER_RE.search(danger_text):
            pen += _EXPLORE_SAFETY_FILTER_COEF
    return pen


def _explore_lprnd_bonus(interactions) -> float:
    """LP-RND lifelong novelty: reuse rollout_log_probs as the 'surprise' signal.

    The intuition (from the Agent57→Agentic-RL analysis, 草案 C):
      r_t^life = clip( (-mean_logprob - mu) / sigma, 0, L )

    Higher negative-logprob = trajectory is more surprising under current policy =
    indicates exploration into previously-low-density regions. Running stats keep
    the bonus normalized so it doesn't dominate task reward as training progresses.

    Zero extra parameters: relies entirely on log-probs already computed by slime.
    Returns 0.0 when disabled or during EXPLORE_LPRND_WARMUP.
    """
    if not _EXPLORE_LPRND_ENABLED or not interactions:
        return 0.0
    # Average negative logprob across all generated tokens in this rollout.
    total_logp, total_tok = 0.0, 0
    for it in interactions:
        lp = list(getattr(it, "output_token_logprobs", []) or [])
        if not lp:
            continue
        total_logp += sum(lp)
        total_tok += len(lp)
    if total_tok == 0:
        return 0.0
    surprise = -(total_logp / total_tok)  # mean negative logprob, in nats

    s = _LPRND_STATS
    if s["warmup"] < _EXPLORE_LPRND_WARMUP:
        # Bug fix: the previous implementation updated Welford statistics during
        # warmup and then returned 0. That made early high-entropy rollouts the
        # normalization baseline, suppressing the novelty signal later. Warmup
        # now only counts trajectories; normalization starts afterward.
        s["warmup"] += 1
        return 0.0

    # Welford running stats after warmup.
    s["n"] += 1
    delta = surprise - s["mean"]
    s["mean"] += delta / s["n"]
    s["m2"] += delta * (surprise - s["mean"])
    if s["n"] < 2:
        return 0.0
    var = s["m2"] / max(1, s["n"] - 1)
    std = max(math.sqrt(var), 1e-6)
    z = (surprise - s["mean"]) / std
    return max(0.0, min(_EXPLORE_LPRND_CLIP, z))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _token_fingerprint(token_ids: list[int], limit: int) -> str | None:
    if not token_ids:
        return None
    try:
        payload = json.dumps(
            [int(x) for x in token_ids[:limit]],
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        return None
    return hashlib.sha256(payload).hexdigest()[:16]


def _turn_uncertainty_metrics(
    interaction: Interaction,
    *,
    previous_turn_score: float | None = None,
) -> dict[str, Any]:
    """Build T2PO-style turn diagnostics from sampled-token log-probs."""
    if not _TURN_UNCERTAINTY_ENABLED:
        return {}

    output_ids = list(interaction.output_token_ids or [])
    raw_logprobs = list(interaction.output_token_logprobs or [])
    nums = [_finite_float(v) for v in raw_logprobs]
    nums = [v for v in nums if v is not None]

    record: dict[str, Any] = {
        "schema": _TURN_UNCERTAINTY_SCHEMA,
        "schema_version": _TURN_UNCERTAINTY_SCHEMA_VERSION,
        "source": "rollout_logprobs",
        "score_kind": "mean_sampled_token_logprob_proxy",
        "turn_idx": int(interaction.turn_idx),
        "available": False,
        "n_input_tokens": len(interaction.input_ids or []),
        "n_output_tokens": len(output_ids),
        "n_logprob_tokens": len(nums),
        "ignored_prefix_tokens": min(_TURN_UNCERTAINTY_WARMUP_TOKENS, len(nums)),
        "fingerprint": _token_fingerprint(
            output_ids, _TURN_UNCERTAINTY_FINGERPRINT_TOKENS
        ),
        "fingerprint_tokens": _TURN_UNCERTAINTY_FINGERPRINT_TOKENS,
        "finish_reason": interaction.finish_reason,
        "latency_ms": float(interaction.latency_ms or 0.0),
        "low_progress_threshold": _TURN_LOW_PROGRESS_THRESHOLD,
    }

    if not nums:
        record["missing_reason"] = "missing_output_token_logprobs"
        return record

    scored = nums[_TURN_UNCERTAINTY_WARMUP_TOKENS:]
    if not scored:
        record["missing_reason"] = "all_tokens_skipped_by_warmup"
        return record

    count = len(scored)
    mean_logprob = sum(scored) / count
    variance = sum((x - mean_logprob) ** 2 for x in scored) / count
    mean_neg_logprob = -mean_logprob
    turn_score = mean_logprob

    record.update(
        {
            "available": True,
            "n_scored_tokens": count,
            "turn_level_score": turn_score,
            "turn_level_uncertainty": mean_neg_logprob,
            "mean_logprob": mean_logprob,
            "std_logprob": math.sqrt(max(variance, 0.0)),
            "min_logprob": min(scored),
            "max_logprob": max(scored),
            "mean_neg_logprob": mean_neg_logprob,
            "sum_neg_logprob": -sum(scored),
            "log_ppl": mean_neg_logprob,
            "ppl": math.exp(min(mean_neg_logprob, 50.0)),
            "first_scored_logprob": scored[0],
            "last_scored_logprob": scored[-1],
        }
    )

    if previous_turn_score is not None and math.isfinite(previous_turn_score):
        delta = turn_score - previous_turn_score
        abs_delta = abs(delta)
        record["score_delta_from_prev"] = delta
        record["abs_score_delta_from_prev"] = abs_delta
        record["low_progress_from_prev"] = (
            abs_delta > 0.0 and abs_delta < _TURN_LOW_PROGRESS_THRESHOLD
        )
    else:
        record["score_delta_from_prev"] = None
        record["abs_score_delta_from_prev"] = None
        record["low_progress_from_prev"] = False

    return record


def _summarize_turn_uncertainty(
    records: list[dict[str, Any]],
    *,
    run_ctx: RunContext,
) -> dict[str, Any]:
    if not _TURN_UNCERTAINTY_ENABLED:
        return {}

    all_records = [r for r in records if isinstance(r, dict) and r]
    available = [r for r in all_records if r.get("available")]

    def collect(key: str) -> list[float]:
        vals: list[float] = []
        for rec in available:
            num = _finite_float(rec.get(key))
            if num is not None:
                vals.append(num)
        return vals

    def stats(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return {
            "count": len(values),
            "mean": mean,
            "std": math.sqrt(max(var, 0.0)),
            "min": min(values),
            "max": max(values),
        }

    scores = collect("turn_level_score")
    uncertainties = collect("turn_level_uncertainty")
    deltas = collect("abs_score_delta_from_prev")
    score_stats = stats(scores)
    uncertainty_stats = stats(uncertainties)
    delta_stats = stats(deltas)
    low_progress_count = sum(1 for r in available if r.get("low_progress_from_prev"))

    summary: dict[str, Any] = {
        "schema": _TURN_UNCERTAINTY_SCHEMA,
        "schema_version": _TURN_UNCERTAINTY_SCHEMA_VERSION,
        "source": "rollout_logprobs",
        "score_kind": "mean_sampled_token_logprob_proxy",
        "uid": run_ctx.uid,
        "group_index": run_ctx.group_index,
        "sample_index": run_ctx.sample_index,
        "rollout_id": run_ctx.rollout_id,
        "train_step": run_ctx.train_step,
        "rollout_step": run_ctx.rollout_step,
        "turn_count": len(all_records),
        "available_turn_count": len(available),
        "missing_turn_count": len(all_records) - len(available),
        "warmup_tokens": _TURN_UNCERTAINTY_WARMUP_TOKENS,
        "low_progress_threshold": _TURN_LOW_PROGRESS_THRESHOLD,
        "low_progress_turn_count": low_progress_count,
        "low_progress_fraction": (
            low_progress_count / len(available) if available else None
        ),
    }

    if score_stats:
        summary.update(
            {
                "mean_turn_level_score": score_stats["mean"],
                "std_turn_level_score": score_stats["std"],
                "min_turn_level_score": score_stats["min"],
                "max_turn_level_score": score_stats["max"],
            }
        )
    if uncertainty_stats:
        summary.update(
            {
                "mean_turn_level_uncertainty": uncertainty_stats["mean"],
                "std_turn_level_uncertainty": uncertainty_stats["std"],
                "min_turn_level_uncertainty": uncertainty_stats["min"],
                "max_turn_level_uncertainty": uncertainty_stats["max"],
            }
        )
    if delta_stats:
        summary.update(
            {
                "mean_abs_score_delta": delta_stats["mean"],
                "min_abs_score_delta": delta_stats["min"],
                "max_abs_score_delta": delta_stats["max"],
            }
        )

    return summary


def _explore_schedule_multiplier(schedule: str, train_step: Any, decay_steps: int) -> float:
    """SPEAR-style curriculum multiplier for auxiliary exploration rewards."""
    mode = (schedule or "constant").strip().lower()
    if mode in {"constant", "none", "off"}:
        return 1.0
    if decay_steps <= 0 or train_step is None:
        return 1.0
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return 1.0
    progress = min(1.0, step / max(1.0, float(decay_steps)))
    if mode == "cosine":
        return max(0.0, (math.cos(progress * math.pi) + 1.0) / 2.0)
    if mode == "linear":
        return max(0.0, 1.0 - progress)
    logger.warning("Unknown exploration schedule=%r; using constant", schedule)
    return 1.0


def _explore_actor_log_ppl(interactions) -> float:
    """Mean negative actor logprob over generated tokens, i.e. log perplexity."""
    total_logp, total_tok = 0.0, 0
    for it in interactions or []:
        lp = list(getattr(it, "output_token_logprobs", []) or [])
        if not lp:
            continue
        total_logp += sum(lp)
        total_tok += len(lp)
    if total_tok <= 0:
        return 0.0
    return max(0.0, -(total_logp / total_tok))


def _explore_decayed_weight(weight: float, train_step: Any, decay_steps: int) -> float:
    if decay_steps <= 0 or train_step is None:
        return max(0.0, float(weight))
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return max(0.0, float(weight))
    progress = min(1.0, step / max(1.0, float(decay_steps)))
    return max(0.0, float(weight) * (1.0 - progress))


def _explore_cde_actor_metrics(
    interactions,
    base_score_mean: float,
    train_step: Any,
) -> Dict[str, float]:
    """Actor-side CDE/PPL curiosity metrics for optional reward shaping.

    This intentionally implements only the actor bonus from the CDE paper. The
    critic bonus requires a multi-head critic/value path, which terminal-rl's
    current GRPO/DAPO rollout path does not have.
    """
    metrics = {
        "log_ppl": 0.0,
        "omega": 0.0,
        "alpha": _EXPLORE_CDE_ACTOR_ALPHA,
        "kappa": _EXPLORE_CDE_ACTOR_KAPPA,
        "decay_steps": float(_EXPLORE_CDE_ACTOR_DECAY_STEPS),
        "base_score_mean": 0.0,
        "base_score_magnitude": 0.0,
        "cap": 0.0,
        "scaled": 0.0,
        "clipped": 0.0,
        "bonus": 0.0,
        "eligible": 0.0,
    }
    if not _EXPLORE_CDE_ACTOR_ENABLED:
        return metrics

    log_ppl = _explore_actor_log_ppl(interactions)
    omega = _explore_decayed_weight(
        _EXPLORE_CDE_ACTOR_OMEGA,
        train_step,
        _EXPLORE_CDE_ACTOR_DECAY_STEPS,
    )
    base_mean = float(base_score_mean)
    base_magnitude = abs(base_mean)
    gate = _EXPLORE_CDE_ACTOR_REWARD_GATE
    if gate in {"positive", "pos"}:
        eligible = base_mean > 0.0
    elif gate in {"nonnegative", "non-negative"}:
        eligible = base_mean >= 0.0
    elif gate in {"none", "off", "always", "all"}:
        eligible = True
    else:
        # Paper-faithful default: any non-zero verifiable reward magnitude can
        # bound curiosity. For safety-heavy runs, use gate=positive to avoid
        # softening unsafe negative rewards.
        eligible = base_magnitude > 0.0

    cap = base_magnitude / max(_EXPLORE_CDE_ACTOR_KAPPA, 1e-6) if eligible else 0.0
    scaled = max(0.0, _EXPLORE_CDE_ACTOR_ALPHA * log_ppl)
    clipped = min(cap, scaled)
    metrics.update(
        {
            "log_ppl": log_ppl,
            "omega": omega,
            "base_score_mean": base_mean,
            "base_score_magnitude": base_magnitude,
            "cap": cap,
            "scaled": scaled,
            "clipped": clipped,
            "bonus": omega * clipped,
            "eligible": 1.0 if eligible else 0.0,
        }
    )
    return metrics


def _explore_debug_metrics(
    *,
    status: Sample.Status,
    base_score_mean: float,
    total_bonus: float,
    intrinsic_scaled: float,
    safety_penalty: float,
    lprnd_bonus: float,
    agent57_bonus: float,
    cde_actor: Dict[str, float],
    turn_records: List[Dict[str, Any]],
    parse_error_count: int,
) -> Dict[str, Any]:
    """Structured exploration/exploitation diagnostics for logs and trajectory audits."""
    tool_call_count = 0
    action_count = 0
    danger_command_count = 0
    actions = _iter_explore_actions(turn_records)
    for tr in turn_records or []:
        tool_calls = tr.get("tool_calls") or []
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
    for action in actions:
        action_count += 1
        danger_text = action.get("danger_text", "")
        if danger_text and _DANGER_RE.search(danger_text):
            danger_command_count += 1

    base_abs = abs(float(base_score_mean))
    bonus_to_base = abs(float(total_bonus)) / max(base_abs, 1e-6)
    curiosity_pressure = (
        max(0.0, intrinsic_scaled)
        + max(0.0, lprnd_bonus)
        + max(0.0, agent57_bonus)
        + max(0.0, float(cde_actor.get("bonus", 0.0)))
    )
    safety_pressure = max(0.0, -float(safety_penalty)) + float(danger_command_count)
    reward_hacking_risk = bool(base_score_mean <= 0.0 and total_bonus > 0.0)
    over_exploration_risk = bool(bonus_to_base > 0.5 and base_score_mean <= 0.0)
    safety_tension = bool(safety_pressure > 0.0)

    status_value = getattr(status, "value", str(status)).lower()
    if status_value in {"failed", "aborted", "truncated"}:
        mood = "stuck"
    elif safety_tension:
        mood = "risky"
    elif reward_hacking_risk:
        mood = "curious_unproven"
    elif base_score_mean > 0.0 and curiosity_pressure > 0.0:
        mood = "curious_success"
    elif base_score_mean > 0.0:
        mood = "confident_exploit"
    elif total_bonus < 0.0:
        mood = "cautious"
    else:
        mood = "low_signal"

    mood_code = {
        "low_signal": 0,
        "confident_exploit": 1,
        "curious_success": 2,
        "curious_unproven": 3,
        "cautious": 4,
        "risky": 5,
        "stuck": 6,
    }.get(mood, -1)

    return {
        "explore_base_score_before_bonus": base_score_mean,
        "explore_bonus_to_base_abs_ratio": bonus_to_base,
        "explore_curiosity_pressure": curiosity_pressure,
        "explore_tool_intrinsic_pressure": max(0.0, intrinsic_scaled),
        "explore_safety_pressure": safety_pressure,
        "explore_mood": mood,
        "explore_mood_code": mood_code,
        "explore_reward_hacking_risk": reward_hacking_risk,
        "explore_over_exploration_risk": over_exploration_risk,
        "explore_safety_tension": safety_tension,
        "explore_turn_count": len(turn_records or []),
        "explore_tool_call_count": tool_call_count,
        "explore_action_count": action_count,
        "explore_danger_command_count": danger_command_count,
        "explore_parse_error_count": int(parse_error_count or 0),
    }
