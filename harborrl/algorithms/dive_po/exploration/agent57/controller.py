from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import random
import re
import sqlite3
import threading
import time
from typing import Any, Iterable

import numpy as np



from harborrl.algorithms.dive_po.exploration.agent57.storage import (
    _SQLITE_SCHEMA_INITIALIZED,
    connect,
)
from harborrl.algorithms.dive_po.exploration.agent57.config import (
    Agent57LiteConfig,
    config_from_env,
)


_LOCAL_LOCK = threading.Lock()
_LOCAL_COUNTS: dict[str, float] = {}
_LOCAL_COUNT_LAST_SEEN: dict[str, int] = {}
_LOCAL_TRAJ_SEEN = 0
_LOCAL_ARM_EVENTS: list[dict[str, Any]] = []
_LOCAL_LIFE_RAW_N = 0
_LOCAL_LIFE_RAW_MEAN = 0.0
_LOCAL_LIFE_RAW_M2 = 0.0
_UCB_RNG_LOCK = threading.Lock()
_UCB_RNG_SEED: int | None = None
_UCB_RNG: np.random.Generator | None = None


def _normalize_dataset(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("._-")
    if text in {"", "terminal_bench", "seta_env"}:
        return "seta"
    if text in {"agent-safety-bench", "agent_safety_bench", "asb", "safety"}:
        return "agent_safetybench"
    if text in {"agent_harm", "ah"}:
        return "agentharm"
    return text or "unknown"



def _sqlite_next_counts(config: Agent57LiteConfig, keys: Iterable[str]) -> tuple[int, list[float]]:
    unique_keys = list(dict.fromkeys(keys))
    if not unique_keys:
        return 0, []
    conn = connect(
        config.state_path,
        busy_timeout_ms=config.sqlite_busy_timeout_ms,
        wal=config.sqlite_wal,
    )
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value FROM meta WHERE name='lifelong_traj_seen'"
        ).fetchone()
        seen = int(row[0]) if row else 0
        conn.execute(
            "INSERT INTO meta(name, value) VALUES('lifelong_traj_seen', 1) "
            "ON CONFLICT(name) DO UPDATE SET value=value+1"
        )
        next_seen = seen + 1
        counts_before: list[float] = []
        for key in unique_keys:
            row = conn.execute(
                "SELECT count, last_seen FROM lifelong_counts WHERE key=?", (key,)
            ).fetchone()
            before = _decayed_count(
                config,
                float(row[0]) if row else 0.0,
                int(row[1]) if row else 0,
                seen,
            )
            counts_before.append(before)
            after = before + 1.0
            if row:
                conn.execute(
                    "UPDATE lifelong_counts SET count=?, last_seen=? WHERE key=?",
                    (after, next_seen, key),
                )
            else:
                conn.execute(
                    "INSERT INTO lifelong_counts(key, count, last_seen) VALUES(?, ?, ?)",
                    (key, after, next_seen),
                )
        if config.lifelong_capacity > 0:
            row = conn.execute("SELECT COUNT(*) FROM lifelong_counts").fetchone()
            overflow = int(row[0]) - config.lifelong_capacity if row else 0
            if overflow > 0:
                conn.execute(
                    "DELETE FROM lifelong_counts WHERE key IN ("
                    "SELECT key FROM lifelong_counts ORDER BY last_seen ASC LIMIT ?"
                    ")",
                    (overflow,),
                )
        conn.execute("COMMIT")
        return seen, counts_before
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _decayed_count(
    config: Agent57LiteConfig,
    count: float,
    last_seen: int,
    seen_now: int,
) -> float:
    value = max(0.0, float(count))
    decay = min(1.0, max(0.0, float(config.lifelong_count_decay)))
    if value <= 0.0 or decay >= 1.0:
        return value
    gap = max(0, int(seen_now) - int(last_seen))
    if gap <= 0:
        return value
    if decay <= 0.0:
        return 0.0
    return value * (decay ** gap)


def _lifelong_raw_from_counts(config: Agent57LiteConfig, counts_before: list[float]) -> float:
    if counts_before:
        raw = sum(1.0 / math.sqrt(c + 1.0) for c in counts_before) / len(counts_before)
    else:
        raw = 0.0
    return min(config.lifelong_clip, max(0.0, raw)) if config.lifelong_clip > 0 else raw


def _normalize_key_groups(keys: Iterable[str] | dict[str, Iterable[str]]) -> dict[str, list[str]]:
    if isinstance(keys, dict):
        return {
            str(level): [str(key) for key in values]
            for level, values in keys.items()
            if values
        }
    return {"task": [str(key) for key in keys]}


def _flatten_key_groups(key_groups: dict[str, list[str]]) -> list[str]:
    flattened: list[str] = []
    for keys in key_groups.values():
        flattened.extend(keys)
    return list(dict.fromkeys(flattened))


def _raw_from_key_counts(
    config: Agent57LiteConfig,
    keys: list[str],
    counts_by_key: dict[str, float],
) -> float:
    unique_keys = list(dict.fromkeys(keys))
    return _lifelong_raw_from_counts(
        config,
        [float(counts_by_key.get(key, 0.0)) for key in unique_keys],
    )


def _lifelong_raw_from_key_groups(
    config: Agent57LiteConfig,
    key_groups: dict[str, list[str]],
    counts_by_key: dict[str, float],
) -> tuple[float, dict[str, float]]:
    raw_by_level = {
        level: _raw_from_key_counts(config, keys, counts_by_key)
        for level, keys in key_groups.items()
        if keys
    }
    if not raw_by_level:
        return 0.0, {}
    if not config.lifelong_hierarchical:
        return raw_by_level.get("task", next(iter(raw_by_level.values()))), raw_by_level

    weights = {
        "task": config.lifelong_task_weight,
        "skill": config.lifelong_skill_weight,
        "global": config.lifelong_global_weight,
    }
    active_weights = {
        level: max(0.0, float(weights.get(level, 0.0)))
        for level in raw_by_level
    }
    total_weight = sum(active_weights.values())
    if total_weight <= 0.0:
        return raw_by_level.get("task", next(iter(raw_by_level.values()))), raw_by_level
    raw = sum(raw_by_level[level] * active_weights[level] for level in raw_by_level) / total_weight
    raw = min(config.lifelong_clip, max(0.0, raw)) if config.lifelong_clip > 0 else raw
    return raw, raw_by_level


def _sqlite_next_counts_and_raw_stats(
    config: Agent57LiteConfig,
    keys: Iterable[str] | dict[str, Iterable[str]],
) -> tuple[int, dict[str, float], float, int, float, float, dict[str, float], float]:
    key_groups = _normalize_key_groups(keys)
    unique_keys = _flatten_key_groups(key_groups)
    if not unique_keys:
        return 0, {}, 0.0, 0, 0.0, 0.0, {}, 0.0
    conn = connect(
        config.state_path,
        busy_timeout_ms=config.sqlite_busy_timeout_ms,
        wal=config.sqlite_wal,
    )
    try:
        wait_start = time.perf_counter()
        conn.execute("BEGIN IMMEDIATE")
        lock_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        row = conn.execute(
            "SELECT value FROM meta WHERE name='lifelong_traj_seen'"
        ).fetchone()
        seen = int(row[0]) if row else 0
        conn.execute(
            "INSERT INTO meta(name, value) VALUES('lifelong_traj_seen', 1) "
            "ON CONFLICT(name) DO UPDATE SET value=value+1"
        )
        next_seen = seen + 1
        counts_before: dict[str, float] = {}
        for key in unique_keys:
            row = conn.execute(
                "SELECT count, last_seen FROM lifelong_counts WHERE key=?", (key,)
            ).fetchone()
            before = _decayed_count(
                config,
                float(row[0]) if row else 0.0,
                int(row[1]) if row else 0,
                seen,
            )
            counts_before[key] = before
            after = before + 1.0
            if row:
                conn.execute(
                    "UPDATE lifelong_counts SET count=?, last_seen=? WHERE key=?",
                    (after, next_seen, key),
                )
            else:
                conn.execute(
                    "INSERT INTO lifelong_counts(key, count, last_seen) VALUES(?, ?, ?)",
                    (key, after, next_seen),
                )
        if config.lifelong_capacity > 0:
            row = conn.execute("SELECT COUNT(*) FROM lifelong_counts").fetchone()
            overflow = int(row[0]) - config.lifelong_capacity if row else 0
            if overflow > 0:
                conn.execute(
                    "DELETE FROM lifelong_counts WHERE key IN ("
                    "SELECT key FROM lifelong_counts ORDER BY last_seen ASC LIMIT ?"
                    ")",
                    (overflow,),
                )

        raw, raw_by_level = _lifelong_raw_from_key_groups(
            config,
            key_groups,
            counts_before,
        )
        rows = {
            str(name): float(value)
            for name, value in conn.execute(
                "SELECT name, value FROM meta WHERE name IN "
                "('lifelong_raw_n', 'lifelong_raw_mean', 'lifelong_raw_m2')"
            )
        }
        n_before = int(rows.get("lifelong_raw_n", 0.0))
        mean_before = float(rows.get("lifelong_raw_mean", 0.0))
        m2_before = float(rows.get("lifelong_raw_m2", 0.0))
        std_before = math.sqrt(max(0.0, m2_before / max(1, n_before - 1))) if n_before > 1 else 0.0
        n_after = n_before + 1
        delta = raw - mean_before
        mean_after = mean_before + delta / n_after
        delta2 = raw - mean_after
        m2_after = m2_before + delta * delta2
        for name, value in (
            ("lifelong_raw_n", float(n_after)),
            ("lifelong_raw_mean", float(mean_after)),
            ("lifelong_raw_m2", float(m2_after)),
        ):
            conn.execute(
                "INSERT INTO meta(name, value) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (name, value),
            )
        conn.execute("COMMIT")
        return seen, counts_before, raw, n_before, mean_before, std_before, raw_by_level, lock_wait_ms
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _local_next_counts(config: Agent57LiteConfig, keys: Iterable[str]) -> tuple[int, list[float]]:
    seen, counts_by_key = _local_next_key_counts(config, keys)
    return seen, [float(counts_by_key.get(str(key), 0.0)) for key in list(dict.fromkeys(keys))]


def _local_next_key_counts(
    config: Agent57LiteConfig,
    keys: Iterable[str] | dict[str, Iterable[str]],
) -> tuple[int, dict[str, float]]:
    global _LOCAL_TRAJ_SEEN
    key_groups = _normalize_key_groups(keys)
    unique_keys = _flatten_key_groups(key_groups)
    with _LOCAL_LOCK:
        seen = _LOCAL_TRAJ_SEEN
        _LOCAL_TRAJ_SEEN += 1
        next_seen = _LOCAL_TRAJ_SEEN
        counts_before: dict[str, float] = {}
        for key in unique_keys:
            before = _decayed_count(
                config,
                float(_LOCAL_COUNTS.get(key, 0.0)),
                int(_LOCAL_COUNT_LAST_SEEN.get(key, 0)),
                seen,
            )
            counts_before[key] = before
            _LOCAL_COUNTS[key] = before + 1.0
            _LOCAL_COUNT_LAST_SEEN[key] = next_seen
        if config.lifelong_capacity > 0 and len(_LOCAL_COUNTS) > config.lifelong_capacity:
            overflow = len(_LOCAL_COUNTS) - config.lifelong_capacity
            oldest = sorted(_LOCAL_COUNT_LAST_SEEN.items(), key=lambda item: item[1])[:overflow]
            for key, _ in oldest:
                _LOCAL_COUNTS.pop(key, None)
                _LOCAL_COUNT_LAST_SEEN.pop(key, None)
    return seen, counts_before


def _sqlite_lifelong_raw_stats(config: Agent57LiteConfig, raw: float) -> tuple[int, float, float]:
    conn = connect(
        config.state_path,
        busy_timeout_ms=config.sqlite_busy_timeout_ms,
        wal=config.sqlite_wal,
    )
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = {
            str(name): float(value)
            for name, value in conn.execute(
                "SELECT name, value FROM meta WHERE name IN "
                "('lifelong_raw_n', 'lifelong_raw_mean', 'lifelong_raw_m2')"
            )
        }
        n_before = int(rows.get("lifelong_raw_n", 0.0))
        mean_before = float(rows.get("lifelong_raw_mean", 0.0))
        m2_before = float(rows.get("lifelong_raw_m2", 0.0))
        std_before = math.sqrt(max(0.0, m2_before / max(1, n_before - 1))) if n_before > 1 else 0.0

        n_after = n_before + 1
        delta = raw - mean_before
        mean_after = mean_before + delta / n_after
        delta2 = raw - mean_after
        m2_after = m2_before + delta * delta2
        for name, value in (
            ("lifelong_raw_n", float(n_after)),
            ("lifelong_raw_mean", float(mean_after)),
            ("lifelong_raw_m2", float(m2_after)),
        ):
            conn.execute(
                "INSERT INTO meta(name, value) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (name, value),
            )
        conn.execute("COMMIT")
        return n_before, mean_before, std_before
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _local_lifelong_raw_stats(raw: float) -> tuple[int, float, float]:
    global _LOCAL_LIFE_RAW_N, _LOCAL_LIFE_RAW_MEAN, _LOCAL_LIFE_RAW_M2
    with _LOCAL_LOCK:
        n_before = _LOCAL_LIFE_RAW_N
        mean_before = _LOCAL_LIFE_RAW_MEAN
        std_before = (
            math.sqrt(max(0.0, _LOCAL_LIFE_RAW_M2 / max(1, n_before - 1)))
            if n_before > 1
            else 0.0
        )
        n_after = n_before + 1
        delta = raw - _LOCAL_LIFE_RAW_MEAN
        _LOCAL_LIFE_RAW_MEAN += delta / n_after
        delta2 = raw - _LOCAL_LIFE_RAW_MEAN
        _LOCAL_LIFE_RAW_M2 += delta * delta2
        _LOCAL_LIFE_RAW_N = n_after
    return n_before, mean_before, std_before


def _lifelong_raw_stats(config: Agent57LiteConfig, raw: float) -> tuple[int, float, float]:
    if config.lifelong_backend == "sqlite":
        return _sqlite_lifelong_raw_stats(config, raw)
    return _local_lifelong_raw_stats(raw)


def _softplus(value: float) -> float:
    if value > 40.0:
        return value
    if value < -40.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _life_mod_from_raw(
    config: Agent57LiteConfig,
    raw: float,
    *,
    n_before: int,
    mean_before: float,
    std_before: float,
) -> tuple[float, float]:
    if config.ngu_life_mod_mode == "standardized_softplus":
        if n_before > 1 and std_before > 1e-8:
            z = (raw - mean_before) / std_before
        else:
            z = 0.0
        if config.ngu_life_mod_std_clip > 0:
            z = min(max(z, -config.ngu_life_mod_std_clip), config.ngu_life_mod_std_clip)
        life_mod = 1.0 + _softplus(z)
    else:
        z = 0.0
        life_mod = 1.0 + raw
    return min(max(life_mod, 1.0), config.ngu_mod_clip), z


def _stable_hash(text: str, n: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _metadata_value(metadata: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = metadata if isinstance(metadata, dict) else {}
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


def _normalized_base_score(base_score: float, dataset: str) -> float:
    dataset_name = _normalize_dataset(dataset)
    base = _finite_float(base_score)
    if dataset_name == "seta":
        return min(1.0, max(0.0, base))
    if dataset_name in {"agent_safetybench", "agentharm"}:
        return (min(1.0, max(-1.0, base)) + 1.0) / 2.0
    return min(1.0, max(0.0, base))


def _clamp_bonus(value: float, max_abs: float) -> tuple[float, bool]:
    if max_abs <= 0.0:
        return value, False
    clipped = min(max(value, -max_abs), max_abs)
    return clipped, clipped != value


def _bucket_len(text: str) -> str:
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


def _result_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key in ("stdout", "stderr", "output", "result", "message", "error"):
            val = value.get(key)
            if val:
                parts.append(str(val))
        if parts:
            return "\n".join(parts)
    return str(value)


def _canonical_observation_text(text: str) -> str:
    normalized = str(text or "")
    normalized = re.sub(
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
        r"\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}\b",
        "<datetime>",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?\b",
        "<datetime>",
        normalized,
    )
    normalized = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "<time>", normalized)
    normalized = re.sub(r"\bjob\s+\d+\s+at\b", "job <id> at", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bpid\s*[:=]?\s*\d+\b", "pid <id>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?m)^\s*\d+\s*$", "<num>", normalized)
    normalized = re.sub(r"\b0x[0-9a-fA-F]+\b", "<hex>", normalized)
    normalized = re.sub(
        r"\b[0-9a-fA-F]{8,}\b",
        "<hex>",
        normalized,
    )
    normalized = re.sub(r"\b\d{2,}\b", "<num>", normalized)
    return normalized


def exit_code_bucket(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("exit_code", "returncode", "return_code", "code"):
            if key in value and value[key] is not None:
                try:
                    return f"exit{int(value[key])}"
                except (TypeError, ValueError):
                    return f"exit_{str(value[key])[:16]}"
    text = _result_text(value)
    low = text.lower()
    match = re.search(r"(?:exit|return)\s*(?:code|status)?\s*[:=]\s*(-?\d+)", low)
    if match:
        return f"exit{match.group(1)}"
    if "command not found" in low:
        return "exit127"
    if "permission denied" in low:
        return "exit126"
    if (
        "command executed successfully" in low
        or "content successfully written" in low
        or re.search(r"\.\.\.\s*done\.?$", low.strip())
        or "all tests passed" in low
        or "tests passed" in low
    ):
        return "exit0"
    return "exit_unknown"


def coarse_observation_fingerprint(value: Any) -> str:
    text = _canonical_observation_text(_result_text(value))
    low = text.lower()
    if not low.strip():
        return "empty"
    patterns = (
        ("permission_denied", ("permission denied", "operation not permitted")),
        ("not_found", ("no such file", "not found", "cannot stat")),
        ("cmd_not_found", ("command not found",)),
        ("timeout", ("timed out", "timeout")),
        ("traceback", ("traceback", "exception:", "error:")),
        ("assertion", ("assertionerror", "assertion failed")),
        ("test_fail", ("failed", "failure", "tests failed")),
        ("success_no_output", ("command executed successfully (no output)",)),
        ("operation_success", ("command executed successfully", "content successfully written", "successfully written")),
        ("test_pass", ("all tests passed", "tests passed", "passed")),
        ("install", ("apt-get", "pip install", "npm install")),
        ("build", ("building", "compiling", "make:", "cmake")),
    )
    for label, needles in patterns:
        if any(needle in low for needle in needles):
            return f"{label}:{_bucket_len(text)}"
    normalized = re.sub(r"\s+", " ", text.strip())[:512]
    return f"generic:{_bucket_len(text)}:{_stable_hash(normalized, 8)}"


def coarse_observation_label(value: Any) -> str:
    """Return a low-cardinality observation label for count-based lifelong keys."""
    text = _canonical_observation_text(_result_text(value))
    low = text.lower()
    if not low.strip():
        return "empty"
    patterns = (
        ("permission_denied", ("permission denied", "operation not permitted")),
        ("not_found", ("no such file", "not found", "cannot stat")),
        ("cmd_not_found", ("command not found",)),
        ("timeout", ("timed out", "timeout")),
        ("traceback", ("traceback", "exception:", "error:")),
        ("assertion", ("assertionerror", "assertion failed")),
        ("test_fail", ("failed", "failure", "tests failed")),
        ("success_no_output", ("command executed successfully (no output)",)),
        ("operation_success", ("command executed successfully", "content successfully written", "successfully written")),
        ("test_pass", ("all tests passed", "tests passed", "passed")),
        ("install", ("apt-get", "pip install", "npm install")),
        ("build", ("building", "compiling", "make:", "cmake")),
    )
    for label, needles in patterns:
        if any(needle in low for needle in needles):
            return f"{label}:{_bucket_len(text)}"
    return f"generic:{_bucket_len(text)}"


def _observation_bucket(value: Any, mode: str = "fingerprint") -> str:
    mode = (mode or "fingerprint").strip().lower()
    if mode == "none":
        return "obs_ignored"
    if mode == "label":
        return coarse_observation_label(value)
    return coarse_observation_fingerprint(value)


def _turn_result_fingerprints(
    turn_records: list[dict[str, Any]],
    *,
    obs_mode: str = "fingerprint",
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for tr in turn_records or []:
        if tr.get("command"):
            result = tr.get("result") or tr.get("observation") or tr.get("output")
            results.append((_observation_bucket(result, obs_mode), exit_code_bucket(result)))
        for call in tr.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            result = call.get("result")
            if result is None:
                result = call.get("observation") or call.get("output")
            results.append((_observation_bucket(result, obs_mode), exit_code_bucket(result)))
    return results


class LifelongKeyBuilder(ABC):
    """Build stable count keys for Agent57-lite lifelong novelty."""

    @abstractmethod
    def keys(
        self,
        actions: list[dict[str, Any]],
        turn_records: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
    ) -> list[str]:
        raise NotImplementedError


class V1LifelongKeyBuilder(LifelongKeyBuilder):
    """Original key: action signature + coarse observation + exit-code bucket."""

    def keys(
        self,
        actions: list[dict[str, Any]],
        turn_records: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        del metadata
        result_fps = _turn_result_fingerprints(turn_records)
        keys: list[str] = []
        for idx, action in enumerate(actions or []):
            signature = str(action.get("signature") or action.get("raw") or "unknown")
            if idx < len(result_fps):
                obs_fp, exit_fp = result_fps[idx]
            else:
                obs_fp, exit_fp = "no_result", "exit_unknown"
            keys.append(_stable_hash(f"{signature}\n{obs_fp}\n{exit_fp}", 16))
        return keys


def _command_family(action: dict[str, Any]) -> str:
    explicit = str(action.get("action_family") or "").strip().lower()
    if explicit:
        return re.sub(r"[^a-z0-9_.-]+", "_", explicit)[:80]
    tool = str(action.get("tool_name") or "tool").strip().lower() or "tool"
    signature = str(action.get("signature") or action.get("raw") or "")
    parts = [part for part in signature.split("|") if part]
    if len(parts) >= 2:
        return re.sub(r"[^a-z0-9_.-]+", "_", f"{tool}:{parts[1].lower()}")[:80]
    raw = str(action.get("raw") or "").strip().lower()
    match = re.match(r"([a-z0-9_.:/-]+)", raw)
    return re.sub(r"[^a-z0-9_.-]+", "_", f"{tool}:{match.group(1) if match else 'unknown'}")[:80]


def _action_flag_text(action: dict[str, Any]) -> str:
    return str(action.get("danger_text") or action.get("raw") or "").lower()


def _is_test_action(action: dict[str, Any]) -> bool:
    text = _action_flag_text(action)
    patterns = (
        "pytest",
        "python -m pytest",
        "unittest",
        "npm test",
        "pnpm test",
        "yarn test",
        "go test",
        "cargo test",
        "make test",
        "run_tests",
        "test_outputs.py",
    )
    return any(pattern in text for pattern in patterns)


def _is_file_mod_action(action: dict[str, Any]) -> bool:
    tool = str(action.get("tool_name") or "").strip().lower()
    if tool in {"shell_write_content_to_file", "write_file", "edit_file"}:
        return True
    text = _action_flag_text(action)
    return bool(
        re.search(
            r"(^|\s)(?:touch|mkdir|rm|mv|cp|chmod|chown|tee|sed\s+-i|install)\b",
            text,
        )
        or ">" in text
        or ">>" in text
        or "apply_patch" in text
    )


def _turn_bucket(action: dict[str, Any]) -> str:
    try:
        idx = int(action.get("turn_idx", -1))
    except (TypeError, ValueError):
        idx = -1
    if idx < 0:
        return "turn_unknown"
    if idx == 0:
        return "turn0"
    if idx <= 2:
        return "turn1_2"
    if idx <= 5:
        return "turn3_5"
    return "turn6p"


def _task_bucket(metadata: dict[str, Any] | None) -> str:
    values = (
        _metadata_value(metadata, "task_path")
        or _metadata_value(metadata, "task_name")
        or _metadata_value(metadata, "task_id")
        or _metadata_value(metadata, "task_meta", "task_path")
        or _metadata_value(metadata, "task_meta", "task_name")
        or _metadata_value(metadata, "task_meta", "task_id")
        or ""
    )
    return _stable_hash(str(values), 12) if values else "task_unknown"


class V2LifelongKeyBuilder(LifelongKeyBuilder):
    """Context-aware key before moving to embedding/k-NN novelty."""

    def __init__(self, config: Agent57LiteConfig) -> None:
        self.config = config

    def keys(
        self,
        actions: list[dict[str, Any]],
        turn_records: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        result_fps = _turn_result_fingerprints(
            turn_records,
            obs_mode=self.config.lifelong_obs_mode,
        )
        dataset = _normalize_dataset(
            _metadata_value(metadata, "agent57_dataset")
            or _metadata_value(metadata, "data_source")
            or _metadata_value(metadata, "task_meta", "data_source")
        )
        split = str(
            _metadata_value(metadata, "safety_split")
            or _metadata_value(metadata, "task_meta", "safety_split")
            or _metadata_value(metadata, "task_meta", "agentharm_task_type")
            or ""
        ).strip().lower()
        task_bucket = _task_bucket(metadata)

        keys: list[str] = []
        for idx, action in enumerate(actions or []):
            signature = str(action.get("signature") or action.get("raw") or "unknown")
            if idx < len(result_fps):
                obs_fp, exit_fp = result_fps[idx]
            else:
                obs_fp, exit_fp = "no_result", "exit_unknown"
            parts = ["v2"]
            if self.config.lifelong_include_dataset:
                parts.append(f"dataset:{dataset}")
                if split:
                    parts.append(f"split:{split[:80]}")
            if self.config.lifelong_include_task:
                parts.append(f"task:{task_bucket}")
            if self.config.lifelong_include_turn:
                parts.append(f"turn:{_turn_bucket(action)}")
            parts.extend(
                [
                    f"family:{_command_family(action)}",
                    f"test:{int(_is_test_action(action))}",
                    f"filemod:{int(_is_file_mod_action(action))}",
                    f"sig:{signature}",
                    f"obs:{obs_fp}",
                    f"exit:{exit_fp}",
                ]
            )
            keys.append(_stable_hash("\n".join(parts), 16))
        return keys

    def key_groups(
        self,
        actions: list[dict[str, Any]],
        turn_records: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        task_keys = self.keys(actions, turn_records, metadata)
        if not self.config.lifelong_hierarchical:
            return {"task": task_keys}

        skill_result_fps = _turn_result_fingerprints(turn_records, obs_mode="label")
        dataset = _normalize_dataset(
            _metadata_value(metadata, "agent57_dataset")
            or _metadata_value(metadata, "data_source")
            or _metadata_value(metadata, "task_meta", "data_source")
        )
        split = str(
            _metadata_value(metadata, "safety_split")
            or _metadata_value(metadata, "task_meta", "safety_split")
            or _metadata_value(metadata, "task_meta", "agentharm_task_type")
            or ""
        ).strip().lower()

        skill_keys: list[str] = []
        global_keys: list[str] = []
        for idx, action in enumerate(actions or []):
            if idx < len(skill_result_fps):
                obs_label, exit_fp = skill_result_fps[idx]
            else:
                obs_label, exit_fp = "no_result", "exit_unknown"
            common = ["v2"]
            if self.config.lifelong_include_dataset:
                common.append(f"dataset:{dataset}")
                if split:
                    common.append(f"split:{split[:80]}")
            family = _command_family(action)
            flags = [
                f"family:{family}",
                f"test:{int(_is_test_action(action))}",
                f"filemod:{int(_is_file_mod_action(action))}",
            ]
            skill_parts = [
                *common,
                "level:skill",
                *flags,
                f"obs:{obs_label}",
                f"exit:{exit_fp}",
            ]
            global_parts = [
                *common,
                "level:global",
                *flags,
            ]
            skill_keys.append(f"skill:{_stable_hash(chr(10).join(skill_parts), 16)}")
            global_keys.append(f"global:{_stable_hash(chr(10).join(global_parts), 16)}")
        return {"task": task_keys, "skill": skill_keys, "global": global_keys}


def _key_builder(config: Agent57LiteConfig | None) -> LifelongKeyBuilder:
    if config is not None and config.lifelong_key_version == "v2":
        return V2LifelongKeyBuilder(config)
    return V1LifelongKeyBuilder()


def lifelong_keys(
    actions: list[dict[str, Any]],
    turn_records: list[dict[str, Any]],
    *,
    config: Agent57LiteConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    return _key_builder(config).keys(actions, turn_records, metadata)


def lifelong_key_groups(
    actions: list[dict[str, Any]],
    turn_records: list[dict[str, Any]],
    *,
    config: Agent57LiteConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    builder = _key_builder(config)
    if isinstance(builder, V2LifelongKeyBuilder):
        return builder.key_groups(actions, turn_records, metadata)
    return {"task": builder.keys(actions, turn_records, metadata)}


def _status_value(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value).lower()


def _lifelong_trust_gate(
    config: Agent57LiteConfig,
    *,
    status: Any,
    parse_error_count: int,
    seen_before: int,
) -> tuple[float, str]:
    status_text = _status_value(status)
    bad_status = any(part in status_text for part in ("failed", "aborted", "truncated"))
    if config.trust_gate_mode != "soft":
        if bad_status:
            return 0.0, f"status:{status_text}"
        if parse_error_count > 0:
            return 0.0, "parse_error"
        if seen_before < config.lifelong_warmup:
            return 0.0, "warmup"
        return 1.0, ""

    trust = float(config.trust_completed)
    reasons: list[str] = []
    if "aborted" in status_text:
        trust = 0.0
        reasons.append(f"status:{status_text}")
    elif "truncated" in status_text:
        trust = min(trust, config.trust_truncated)
        reasons.append(f"status:{status_text}")
    elif "failed" in status_text:
        trust = min(trust, config.trust_failed)
        reasons.append(f"status:{status_text}")
    if parse_error_count > 0:
        trust = min(trust, config.trust_parse_error)
        reasons.append("parse_error")
    if seen_before < config.lifelong_warmup:
        trust = min(trust, config.trust_warmup)
        reasons.append("warmup")
    return max(0.0, trust), ",".join(dict.fromkeys(reasons))


def compute_lifelong_bonus(
    *,
    config: Agent57LiteConfig,
    arm_id: int,
    actions: list[dict[str, Any]],
    turn_records: list[dict[str, Any]],
    status: Any,
    parse_error_count: int,
    metadata: dict[str, Any] | None = None,
    state_update_allowed: bool = True,
) -> dict[str, Any]:
    beta = config.beta_for_arm(arm_id)
    metrics: dict[str, Any] = {
        "explore_agent57_enabled": bool(config.enabled),
        "explore_agent57_arm_id": int(arm_id),
        "explore_agent57_k": int(config.k),
        "explore_agent57_beta": float(beta),
        "explore_agent57_combine_mode": config.combine_mode,
        "explore_agent57_max_bonus": float(config.max_bonus),
        "explore_agent57_episodic_backend": config.episodic_backend,
        "explore_agent57_ngu_episodic_reducer": config.ngu_episodic_reducer,
        "explore_agent57_ngu_life_mod_mode": config.ngu_life_mod_mode,
        "explore_agent57_ngu_life_mod_std_clip": float(config.ngu_life_mod_std_clip),
        "explore_agent57_controller": config.controller,
        "explore_agent57_ucb_c": float(config.ucb_c),
        "explore_agent57_ucb_window": int(config.ucb_window),
        "explore_agent57_ucb_epsilon": float(config.ucb_epsilon),
        "explore_agent57_ucb_min_per_arm": int(config.ucb_min_per_arm),
        "explore_agent57_ucb_value": config.ucb_value,
        "explore_agent57_ucb_parse_penalty": float(config.ucb_parse_penalty),
        "explore_agent57_ucb_trunc_penalty": float(config.ucb_trunc_penalty),
        "explore_agent57_ucb_skip_infra_failures": bool(config.ucb_skip_infra_failures),
        "explore_agent57_ucb_dataset_aware": bool(config.ucb_dataset_aware),
        "explore_agent57_ucb_random_seed": (
            -1 if config.ucb_random_seed is None else int(config.ucb_random_seed)
        ),
        "explore_agent57_ucb_seed_salt": config.ucb_seed_salt,
        "explore_agent57_lifelong_enabled": bool(config.lifelong_enabled),
        "explore_agent57_lifelong_backend": config.lifelong_backend,
        "explore_agent57_lifelong_state_path": config.state_path,
        "explore_agent57_sqlite_busy_timeout_ms": int(config.sqlite_busy_timeout_ms),
        "explore_agent57_sqlite_wal": bool(config.sqlite_wal),
        "explore_agent57_lifelong_coef": float(config.lifelong_coef),
        "explore_agent57_lifelong_clip": float(config.lifelong_clip),
        "explore_agent57_lifelong_warmup": int(config.lifelong_warmup),
        "explore_agent57_lifelong_count_decay": float(config.lifelong_count_decay),
        "explore_agent57_lifelong_capacity": int(config.lifelong_capacity),
        "explore_agent57_lifelong_key_version": config.lifelong_key_version,
        "explore_agent57_lifelong_include_dataset": bool(config.lifelong_include_dataset),
        "explore_agent57_lifelong_include_task": bool(config.lifelong_include_task),
        "explore_agent57_lifelong_include_turn": bool(config.lifelong_include_turn),
        "explore_agent57_lifelong_obs_mode": config.lifelong_obs_mode,
        "explore_agent57_lifelong_hierarchical": bool(config.lifelong_hierarchical),
        "explore_agent57_lifelong_task_weight": float(config.lifelong_task_weight),
        "explore_agent57_lifelong_skill_weight": float(config.lifelong_skill_weight),
        "explore_agent57_lifelong_global_weight": float(config.lifelong_global_weight),
        "explore_agent57_trust_gate_mode": config.trust_gate_mode,
        "explore_agent57_trust": 0.0,
        "explore_agent57_lifelong_raw": 0.0,
        "explore_agent57_lifelong_z": 0.0,
        "explore_agent57_lifelong_stat_n": 0,
        "explore_agent57_lifelong_stat_mean": 0.0,
        "explore_agent57_lifelong_stat_std": 0.0,
        "explore_agent57_lifelong_stat_error": "",
        "explore_agent57_ngu_life_mod": 1.0,
        "explore_agent57_lifelong_bonus": 0.0,
        "explore_agent57_lifelong_bonus_unclipped": 0.0,
        "explore_agent57_lifelong_unique_keys": 0,
        "explore_agent57_lifelong_key_count": 0,
        "explore_agent57_lifelong_duplicate_key_count": 0,
        "explore_agent57_lifelong_task_unique_keys": 0,
        "explore_agent57_lifelong_skill_unique_keys": 0,
        "explore_agent57_lifelong_global_unique_keys": 0,
        "explore_agent57_lifelong_task_raw": 0.0,
        "explore_agent57_lifelong_skill_raw": 0.0,
        "explore_agent57_lifelong_global_raw": 0.0,
        "explore_agent57_sqlite_lock_wait_ms": 0.0,
        "explore_agent57_lifelong_seen_before": 0,
        "explore_agent57_lifelong_warmup_remaining": int(config.lifelong_warmup),
        "explore_agent57_lifelong_eligible": 0.0,
        "explore_agent57_lifelong_suppressed_reason": "",
        "explore_agent57_lifelong_state_update_allowed": bool(state_update_allowed),
        "explore_agent57_bonus_unclipped": 0.0,
        "explore_agent57_bonus_clipped": 0.0,
    }
    if not config.active or not config.lifelong_enabled:
        return metrics
    key_groups = lifelong_key_groups(actions, turn_records, config=config, metadata=metadata)
    total_key_count = sum(len(level_keys) for level_keys in key_groups.values())
    keys = _flatten_key_groups(key_groups)
    metrics["explore_agent57_lifelong_key_count"] = int(total_key_count)
    metrics["explore_agent57_lifelong_unique_keys"] = int(len(set(keys)))
    metrics["explore_agent57_lifelong_duplicate_key_count"] = int(
        max(0, total_key_count - len(set(keys)))
    )
    for level in ("task", "skill", "global"):
        level_keys = key_groups.get(level, [])
        metrics[f"explore_agent57_lifelong_{level}_unique_keys"] = int(
            len(set(level_keys))
        )
    if not keys:
        metrics["explore_agent57_lifelong_suppressed_reason"] = "no_actions"
        return metrics
    if not state_update_allowed:
        # Environment/grader failures do not describe policy novelty. Keeping
        # them out of counts and running statistics prevents transient worker
        # outages from making real actions look artificially familiar later.
        metrics["explore_agent57_lifelong_suppressed_reason"] = "infra_failure"
        return metrics

    try:
        if config.lifelong_backend == "sqlite":
            (
                seen_before,
                counts_by_key,
                raw,
                stat_n,
                stat_mean,
                stat_std,
                raw_by_level,
                lock_wait_ms,
            ) = _sqlite_next_counts_and_raw_stats(config, key_groups)
        else:
            seen_before, counts_by_key = _local_next_key_counts(config, key_groups)
            raw, raw_by_level = _lifelong_raw_from_key_groups(
                config,
                key_groups,
                counts_by_key,
            )
            stat_n, stat_mean, stat_std = _lifelong_raw_stats(config, raw)
            lock_wait_ms = 0.0
    except Exception as exc:
        metrics["explore_agent57_lifelong_suppressed_reason"] = (
            f"state_error:{type(exc).__name__}"
        )
        return metrics

    metrics["explore_agent57_lifelong_seen_before"] = int(seen_before)
    metrics["explore_agent57_sqlite_lock_wait_ms"] = float(lock_wait_ms)
    warmup_remaining = max(0, config.lifelong_warmup - seen_before - 1)
    metrics["explore_agent57_lifelong_warmup_remaining"] = int(warmup_remaining)
    metrics["explore_agent57_lifelong_raw"] = float(raw)
    for level in ("task", "skill", "global"):
        metrics[f"explore_agent57_lifelong_{level}_raw"] = float(
            raw_by_level.get(level, 0.0)
        )
    life_mod, life_z = _life_mod_from_raw(
        config,
        raw,
        n_before=stat_n,
        mean_before=stat_mean,
        std_before=stat_std,
    )
    metrics["explore_agent57_lifelong_z"] = float(life_z)
    metrics["explore_agent57_lifelong_stat_n"] = int(stat_n)
    metrics["explore_agent57_lifelong_stat_mean"] = float(stat_mean)
    metrics["explore_agent57_lifelong_stat_std"] = float(stat_std)
    metrics["explore_agent57_ngu_life_mod"] = float(life_mod)

    trust, reason = _lifelong_trust_gate(
        config,
        status=status,
        parse_error_count=parse_error_count,
        seen_before=seen_before,
    )
    metrics["explore_agent57_trust"] = float(trust)
    if reason:
        metrics["explore_agent57_lifelong_suppressed_reason"] = reason
    if trust <= 0.0:
        return metrics

    metrics["explore_agent57_lifelong_eligible"] = 1.0
    unclipped = float(beta * config.lifelong_coef * raw * trust)
    bonus, clipped = _clamp_bonus(unclipped, config.max_bonus)
    metrics["explore_agent57_lifelong_bonus_unclipped"] = unclipped
    metrics["explore_agent57_lifelong_bonus"] = float(bonus)
    metrics["explore_agent57_bonus_unclipped"] = unclipped
    metrics["explore_agent57_bonus_clipped"] = 1.0 if clipped else 0.0
    return metrics


def compute_ngu_lite_bonus(
    *,
    config: Agent57LiteConfig,
    arm_id: int,
    episodic_novelty: float,
    lifelong_raw: float,
    lifelong_eligible: bool,
    trust_gate: float = 1.0,
    life_mod_override: float | None = None,
) -> dict[str, Any]:
    """Compute the optional NGU-lite product bonus.

    The function is intentionally pure and side-effect free: lifelong count
    updates remain in `compute_lifelong_bonus`, while this combines the current
    rollout's episode novelty with the already-measured lifelong raw signal.
    """
    beta = config.beta_for_arm(arm_id)
    episodic = max(0.0, _finite_float(episodic_novelty))
    raw_life = max(0.0, _finite_float(lifelong_raw))
    if life_mod_override is not None:
        life_mod = min(
            max(_finite_float(life_mod_override, 1.0), 1.0),
            config.ngu_mod_clip,
        )
    else:
        life_mod, _ = _life_mod_from_raw(
            config,
            raw_life,
            n_before=0,
            mean_before=0.0,
            std_before=0.0,
        )
    trust = max(0.0, _finite_float(trust_gate, 1.0))
    intrinsic_signal = episodic * life_mod
    metrics: dict[str, Any] = {
        "explore_agent57_ngu_mod_clip": float(config.ngu_mod_clip),
        "explore_agent57_ngu_episodic_source": config.ngu_episodic_source,
        "explore_agent57_ngu_episodic_reducer": config.ngu_episodic_reducer,
        "explore_agent57_ngu_life_mod_mode": config.ngu_life_mod_mode,
        "explore_agent57_ngu_life_mod_std_clip": float(config.ngu_life_mod_std_clip),
        "explore_agent57_ngu_episodic": float(episodic),
        "explore_agent57_ngu_life_mod": float(life_mod),
        "explore_agent57_intrinsic_signal": float(intrinsic_signal),
        "explore_agent57_trust": float(trust),
        "explore_agent57_ngu_bonus": 0.0,
        "explore_agent57_ngu_bonus_unclipped": 0.0,
    }
    if (
        not config.active
        or config.combine_mode != "ngu_lite"
        or not config.lifelong_enabled
        or not lifelong_eligible
        or trust <= 0.0
    ):
        return metrics

    unclipped = float(beta * config.lifelong_coef * intrinsic_signal * trust)
    bonus, clipped = _clamp_bonus(unclipped, config.max_bonus)
    metrics["explore_agent57_ngu_bonus_unclipped"] = unclipped
    metrics["explore_agent57_ngu_bonus"] = float(bonus)
    metrics["explore_agent57_bonus_unclipped"] = unclipped
    metrics["explore_agent57_bonus_clipped"] = 1.0 if clipped else 0.0
    return metrics


def _local_arm_stats(
    config: Agent57LiteConfig,
    *,
    dataset: str | None = None,
) -> list[dict[str, float]]:
    with _LOCAL_LOCK:
        events = list(_LOCAL_ARM_EVENTS[-config.ucb_window:])
    return _aggregate_arm_stats(
        config.k,
        events,
        dataset=dataset,
        skip_infra_failures=config.ucb_skip_infra_failures,
    )


def _sqlite_arm_stats(
    config: Agent57LiteConfig,
    *,
    dataset: str | None = None,
) -> list[dict[str, float]]:
    conn = connect(
        config.state_path,
        busy_timeout_ms=config.sqlite_busy_timeout_ms,
        wal=config.sqlite_wal,
    )
    try:
        if dataset:
            rows = conn.execute(
                "SELECT arm_id, base_score, normalized_base_score, success, "
                "parse_error, truncated, dataset, infra_failure FROM arm_events "
                "WHERE dataset=? ORDER BY id DESC LIMIT ?",
                (_normalize_dataset(dataset), config.ucb_window),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT arm_id, base_score, normalized_base_score, success, "
                "parse_error, truncated, dataset, infra_failure FROM arm_events "
                "ORDER BY id DESC LIMIT ?",
                (config.ucb_window,),
            ).fetchall()
    finally:
        conn.close()
    events = [
        {
            "arm_id": int(row[0]),
            "base_score": float(row[1]),
            "normalized_base_score": float(row[2]),
            "success": float(row[3]),
            "parse_error": float(row[4]),
            "truncated": float(row[5]),
            "dataset": str(row[6] or ""),
            "infra_failure": float(row[7] or 0.0),
        }
        for row in rows
    ]
    return _aggregate_arm_stats(
        config.k,
        events,
        dataset=dataset,
        skip_infra_failures=config.ucb_skip_infra_failures,
    )


def _aggregate_arm_stats(
    k: int,
    events: list[dict[str, Any]],
    *,
    dataset: str | None = None,
    skip_infra_failures: bool = False,
) -> list[dict[str, float]]:
    target_dataset = _normalize_dataset(dataset) if dataset else ""
    stats = [
        {
            "n": 0.0,
            "base_sum": 0.0,
            "normalized_base_sum": 0.0,
            "success_sum": 0.0,
            "parse_sum": 0.0,
            "trunc_sum": 0.0,
            "infra_sum": 0.0,
        }
        for _ in range(k)
    ]
    for event in events:
        if target_dataset and _normalize_dataset(event.get("dataset")) != target_dataset:
            continue
        infra_failure = _finite_float(event.get("infra_failure", 0.0))
        if skip_infra_failures and infra_failure > 0.0:
            continue
        arm_id = int(event.get("arm_id", 0)) % k
        row = stats[arm_id]
        row["n"] += 1.0
        row["base_sum"] += _finite_float(event.get("base_score", 0.0))
        row["normalized_base_sum"] += _finite_float(
            event.get("normalized_base_score", 0.0)
        )
        row["success_sum"] += _finite_float(event.get("success", 0.0))
        row["parse_sum"] += _finite_float(event.get("parse_error", 0.0))
        row["trunc_sum"] += _finite_float(event.get("truncated", 0.0))
        row["infra_sum"] += infra_failure
    return stats


def _reset_ucb_rng_for_tests() -> None:
    global _UCB_RNG, _UCB_RNG_SEED
    with _UCB_RNG_LOCK:
        _UCB_RNG = None
        _UCB_RNG_SEED = None


def _ucb_seeded(config: Agent57LiteConfig) -> bool:
    return config.ucb_random_seed is not None


def _effective_ucb_seed(config: Agent57LiteConfig) -> int:
    seed = int(config.ucb_random_seed or 0)
    if not config.ucb_seed_salt:
        return seed
    payload = f"{seed}:{config.ucb_seed_salt}".encode("utf-8", errors="ignore")
    digest = hashlib.md5(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _ucb_rng(config: Agent57LiteConfig) -> np.random.Generator:
    global _UCB_RNG, _UCB_RNG_SEED
    seed = _effective_ucb_seed(config)
    if _UCB_RNG is None or _UCB_RNG_SEED != seed:
        _UCB_RNG = np.random.default_rng(seed)
        _UCB_RNG_SEED = seed
    return _UCB_RNG


def _ucb_random(config: Agent57LiteConfig) -> float:
    if not _ucb_seeded(config):
        return random.random()
    with _UCB_RNG_LOCK:
        return float(_ucb_rng(config).random())


def _ucb_randrange(config: Agent57LiteConfig, start: int, stop: int) -> int:
    if stop <= start:
        return start
    if not _ucb_seeded(config):
        return random.randrange(start, stop)
    with _UCB_RNG_LOCK:
        return int(_ucb_rng(config).integers(start, stop))


def _rank_ucb_scores(
    config: Agent57LiteConfig,
    scored: list[tuple[float, int]],
) -> list[tuple[float, int]]:
    if not _ucb_seeded(config):
        return sorted(scored, key=lambda item: (-item[0], item[1]))

    groups: dict[float, list[int]] = {}
    for score, arm_id in scored:
        groups.setdefault(score, []).append(arm_id)

    ranked: list[tuple[float, int]] = []
    for score in sorted(groups.keys(), reverse=True):
        arms = list(groups[score])
        if len(arms) > 1:
            with _UCB_RNG_LOCK:
                arms = [int(arm) for arm in _ucb_rng(config).permutation(arms).tolist()]
        else:
            arms.sort()
        ranked.extend((score, arm_id) for arm_id in arms)
    return ranked


def _ucb_scores(
    config: Agent57LiteConfig,
    *,
    dataset: str | None = None,
) -> list[tuple[float, int]]:
    target_dataset = _normalize_dataset(dataset) if config.ucb_dataset_aware and dataset else None
    try:
        stats = (
            _sqlite_arm_stats(config, dataset=target_dataset)
            if config.lifelong_backend == "sqlite"
            else _local_arm_stats(config, dataset=target_dataset)
        )
    except Exception:
        stats = _aggregate_arm_stats(config.k, [], dataset=target_dataset)
    total = max(1.0, sum(row["n"] for row in stats))
    scored: list[tuple[float, int]] = []
    for arm_id, row in enumerate(stats):
        n = row["n"]
        if n <= 0 or n < config.ucb_min_per_arm:
            score = float("inf")
        else:
            mean_success = row["success_sum"] / n
            mean_base = row["base_sum"] / n
            mean_normalized_base = row["normalized_base_sum"] / n
            parse_rate = row["parse_sum"] / n
            trunc_rate = row["trunc_sum"] / n
            parse_penalty = config.ucb_parse_penalty * parse_rate
            trunc_penalty = config.ucb_trunc_penalty * trunc_rate
            if config.ucb_value == "success":
                value = mean_success - parse_penalty - trunc_penalty
            elif config.ucb_value == "base":
                value = mean_base - parse_penalty - trunc_penalty
            elif config.ucb_value == "normalized_base":
                value = mean_normalized_base - parse_penalty - trunc_penalty
            elif config.ucb_value in {"quality", "quality_gate"}:
                outcome_aware_trunc_penalty = trunc_penalty * (1.0 - mean_normalized_base)
                value = mean_normalized_base - parse_penalty - outcome_aware_trunc_penalty
            else:
                value = mean_success + 0.25 * mean_base - parse_penalty - trunc_penalty
            score = value + config.ucb_c * math.sqrt(math.log(total + 1.0) / n)
        scored.append((score, arm_id))
    return _rank_ucb_scores(config, scored)


def assign_group_arms(
    group_size: int,
    *,
    evaluation: bool = False,
    dataset: str | None = None,
) -> list[int]:
    config = config_from_env()
    if evaluation or not config.active:
        return [0 for _ in range(max(0, group_size))]
    group_size = max(0, int(group_size))
    if group_size == 0:
        return []
    if config.controller == "ucb":
        arms: list[int] = []
        if config.keep_baseline:
            arms.append(0)
        ranked = [arm for _, arm in _ucb_scores(config, dataset=dataset)]
        if config.keep_baseline and group_size > 1:
            ranked = [arm for arm in ranked if arm != 0]
            if not ranked:
                ranked = [arm for arm in range(config.k) if arm != 0] or [0]
        cursor = 0
        while len(arms) < group_size:
            if config.ucb_epsilon > 0.0 and _ucb_random(config) < config.ucb_epsilon:
                if config.keep_baseline and group_size > 1 and config.k > 1:
                    arms.append(_ucb_randrange(config, 1, config.k))
                else:
                    arms.append(_ucb_randrange(config, 0, config.k))
            else:
                arms.append(ranked[cursor % len(ranked)] if ranked else len(arms) % config.k)
            cursor += 1
        return arms[:group_size]
    return [idx % config.k for idx in range(group_size)]


def record_arm_event(
    *,
    config: Agent57LiteConfig,
    arm_id: int,
    base_score: float,
    final_score: float,
    status: Any,
    parse_error_count: int,
    bonus: float,
    dataset: str | None = None,
    normalized_base_score: float | None = None,
    success_score: float | None = None,
    infra_failure: bool = False,
) -> None:
    if not config.active:
        return
    status_text = _status_value(status)
    truncated = 1 if "truncated" in status_text else 0
    base = _finite_float(base_score)
    final = _finite_float(final_score, base)
    shaped_bonus = _finite_float(bonus)
    dataset_name = _normalize_dataset(dataset)
    if normalized_base_score is None:
        normalized_base = _normalized_base_score(base, dataset_name)
    else:
        normalized_base = min(
            1.0,
            max(
                0.0,
                _finite_float(
                    normalized_base_score,
                    _normalized_base_score(base, dataset_name),
                ),
            ),
        )
    success_value = base if success_score is None else _finite_float(success_score, base)
    success = 1 if success_value > config.success_threshold else 0
    event = {
        "arm_id": float(int(arm_id) % max(1, config.k)),
        "base_score": base,
        "normalized_base_score": normalized_base,
        "final_score": final,
        "success": float(success),
        "parse_error": float(1 if parse_error_count > 0 else 0),
        "truncated": float(truncated),
        "infra_failure": float(1 if infra_failure else 0),
        "bonus": shaped_bonus,
        "dataset": dataset_name,
    }
    if config.lifelong_backend == "sqlite":
        try:
            conn = connect(
                config.state_path,
                busy_timeout_ms=config.sqlite_busy_timeout_ms,
                wal=config.sqlite_wal,
            )
            try:
                conn.execute(
                    "INSERT INTO arm_events"
                    "(ts, arm_id, base_score, normalized_base_score, final_score, "
                    "success, parse_error, truncated, infra_failure, bonus, dataset) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        int(event["arm_id"]),
                        event["base_score"],
                        event["normalized_base_score"],
                        event["final_score"],
                        int(event["success"]),
                        int(event["parse_error"]),
                        int(event["truncated"]),
                        int(event["infra_failure"]),
                        event["bonus"],
                        event["dataset"],
                    ),
                )
            finally:
                conn.close()
        except Exception:
            return
        return
    with _LOCAL_LOCK:
        _LOCAL_ARM_EVENTS.append(event)
        if len(_LOCAL_ARM_EVENTS) > 10000:
            del _LOCAL_ARM_EVENTS[: len(_LOCAL_ARM_EVENTS) - 10000]
