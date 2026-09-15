from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from harborrl.algorithms.dive_po.exploration.agent57.memory import resolve_episodic_backend_name


from harborrl.env import (
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
    env_optional_int as _env_optional_int,
)


def _parse_betas(raw: str, k: int) -> list[float]:
    values: list[float] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    if not values:
        values = [0.0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04]
    if len(values) < k:
        values.extend([values[-1]] * (k - len(values)))
    return values[:k]


def _default_state_path() -> str:
    explicit = (
        os.getenv("EXPLORE_AGENT57_STATE_PATH")
        or os.getenv("EXPLORE_AGENT57_SQLITE_PATH")
        or ""
    ).strip()
    if explicit:
        return explicit

    run_dir = os.getenv("RUN_DIR", "").strip()
    if run_dir:
        return str(Path(run_dir) / "agent57_lite.sqlite3")

    traj_dir = os.getenv("TERMINAL_SAVE_TRAJ_DIR", "").strip()
    if traj_dir:
        return str(Path(traj_dir).parent / "agent57_lite.sqlite3")

    run_id = os.getenv("RUN_ID", "").strip() or f"pid{os.getpid()}"
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    return str(Path("/tmp") / f"lightrl_agent57_lite_{safe_run_id}.sqlite3")


@dataclass(frozen=True)
class Agent57LiteConfig:
    enabled: bool
    k: int
    arm_betas: tuple[float, ...]
    combine_mode: str
    ngu_mod_clip: float
    ngu_episodic_source: str
    ngu_episodic_reducer: str
    ngu_life_mod_mode: str
    ngu_life_mod_std_clip: float
    episodic_backend: str
    max_bonus: float
    controller: str
    ucb_c: float
    ucb_window: int
    ucb_epsilon: float
    ucb_min_per_arm: int
    ucb_value: str
    ucb_parse_penalty: float
    ucb_trunc_penalty: float
    ucb_skip_infra_failures: bool
    ucb_dataset_aware: bool
    ucb_random_seed: int | None
    ucb_seed_salt: str
    keep_baseline: bool
    lifelong_enabled: bool
    lifelong_coef: float
    lifelong_clip: float
    lifelong_warmup: int
    lifelong_count_decay: float
    lifelong_capacity: int
    lifelong_backend: str
    lifelong_key_version: str
    lifelong_include_dataset: bool
    lifelong_include_task: bool
    lifelong_include_turn: bool
    lifelong_obs_mode: str
    lifelong_hierarchical: bool
    lifelong_task_weight: float
    lifelong_skill_weight: float
    lifelong_global_weight: float
    trust_gate_mode: str
    trust_completed: float
    trust_truncated: float
    trust_failed: float
    trust_parse_error: float
    trust_warmup: float
    state_path: str
    sqlite_busy_timeout_ms: int
    sqlite_wal: bool
    success_threshold: float

    @property
    def active(self) -> bool:
        return self.enabled or self.lifelong_enabled or self.controller != "fixed"

    def beta_for_arm(self, arm_id: int | None) -> float:
        if not self.arm_betas:
            return 0.0
        try:
            idx = int(arm_id or 0) % len(self.arm_betas)
        except (TypeError, ValueError):
            idx = 0
        return float(self.arm_betas[idx])


def config_from_env() -> Agent57LiteConfig:
    k = max(1, _env_int("EXPLORE_AGENT57_K", 8))
    enabled = _env_bool(
        "EXPLORE_AGENT57_LITE_ENABLED",
        _env_bool("EXPLORE_AGENT57_LITE", False),
    )
    lifelong_enabled = _env_bool(
        "EXPLORE_AGENT57_LIFELONG_ENABLED",
        _env_bool("EXPLORE_AGENT57_LIFELONG", False),
    )
    controller = os.getenv("EXPLORE_AGENT57_CONTROLLER", "fixed").strip().lower()
    if controller not in {"fixed", "ucb"}:
        controller = "fixed"
    backend = (
        os.getenv("EXPLORE_AGENT57_BACKEND")
        or os.getenv("EXPLORE_AGENT57_STATE_BACKEND")
        or os.getenv("EXPLORE_AGENT57_LIFELONG_BACKEND", "local")
    ).strip().lower()
    if backend not in {"local", "sqlite"}:
        backend = "local"
    betas = _parse_betas(os.getenv("EXPLORE_AGENT57_ARM_BETAS", ""), k)
    combine_mode = os.getenv("EXPLORE_AGENT57_COMBINE_MODE", "add").strip().lower()
    if combine_mode not in {"add", "ngu_lite"}:
        combine_mode = "add"
    ngu_episodic_source = (
        os.getenv("EXPLORE_AGENT57_NGU_EPISODIC_SOURCE", "signature_intrinsic")
        .strip()
        .lower()
    )
    if ngu_episodic_source not in {"signature_intrinsic", "intrinsic"}:
        ngu_episodic_source = "signature_intrinsic"
    ngu_episodic_reducer = (
        os.getenv("EXPLORE_AGENT57_NGU_EPISODIC_REDUCER")
        or os.getenv("EXPLORE_INTRINSIC_REDUCER")
        or "sum"
    ).strip().lower()
    if ngu_episodic_reducer not in {"sum", "mean"}:
        ngu_episodic_reducer = "sum"
    ngu_life_mod_mode = os.getenv("EXPLORE_AGENT57_NGU_LIFE_MOD_MODE", "linear").strip().lower()
    if ngu_life_mod_mode in {"standardized", "std", "softplus"}:
        ngu_life_mod_mode = "standardized_softplus"
    if ngu_life_mod_mode not in {"linear", "standardized_softplus"}:
        ngu_life_mod_mode = "linear"
    ucb_value = os.getenv("EXPLORE_AGENT57_UCB_VALUE", "legacy").strip().lower()
    if ucb_value not in {"legacy", "success", "base", "normalized_base", "quality", "quality_gate"}:
        ucb_value = "legacy"
    key_version = (
        os.getenv("EXPLORE_AGENT57_LIFELONG_KEY_VERSION", "v1").strip().lower()
    )
    if key_version not in {"v1", "v2"}:
        key_version = "v1"
    obs_mode = os.getenv("EXPLORE_AGENT57_LIFELONG_OBS_MODE", "fingerprint").strip().lower()
    if obs_mode not in {"fingerprint", "label", "none"}:
        obs_mode = "fingerprint"
    trust_gate_mode = os.getenv("EXPLORE_AGENT57_TRUST_GATE", "hard").strip().lower()
    if trust_gate_mode not in {"hard", "soft"}:
        trust_gate_mode = "hard"
    return Agent57LiteConfig(
        enabled=enabled,
        k=k,
        arm_betas=tuple(betas),
        combine_mode=combine_mode,
        ngu_mod_clip=max(1.0, _env_float("EXPLORE_AGENT57_NGU_MOD_CLIP", 5.0)),
        ngu_episodic_source=ngu_episodic_source,
        ngu_episodic_reducer=ngu_episodic_reducer,
        ngu_life_mod_mode=ngu_life_mod_mode,
        ngu_life_mod_std_clip=max(0.0, _env_float("EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP", 5.0)),
        episodic_backend=resolve_episodic_backend_name(
            os.getenv("EXPLORE_AGENT57_EPISODIC_BACKEND")
            or os.getenv("EPISODIC_MEMORY_BACKEND")
            or "legacy"
        ),
        max_bonus=max(0.0, _env_float("EXPLORE_AGENT57_MAX_BONUS", 0.0)),
        controller=controller,
        ucb_c=max(0.0, _env_float("EXPLORE_AGENT57_UCB_C", 0.5)),
        ucb_window=max(1, _env_int("EXPLORE_AGENT57_UCB_WINDOW", 256)),
        ucb_epsilon=min(
            1.0,
            max(0.0, _env_float("EXPLORE_AGENT57_UCB_EPSILON", 0.0)),
        ),
        ucb_min_per_arm=max(0, _env_int("EXPLORE_AGENT57_UCB_MIN_PER_ARM", 0)),
        ucb_value=ucb_value,
        ucb_parse_penalty=max(0.0, _env_float("EXPLORE_AGENT57_UCB_PARSE_PENALTY", 0.5)),
        ucb_trunc_penalty=max(0.0, _env_float("EXPLORE_AGENT57_UCB_TRUNC_PENALTY", 0.5)),
        ucb_skip_infra_failures=_env_bool("EXPLORE_AGENT57_UCB_SKIP_INFRA_FAILURES", True),
        ucb_dataset_aware=_env_bool("EXPLORE_AGENT57_UCB_DATASET_AWARE", False),
        ucb_random_seed=(
            _env_optional_int("EXPLORE_AGENT57_UCB_RANDOM_SEED")
            if os.getenv("EXPLORE_AGENT57_UCB_RANDOM_SEED") is not None
            else _env_optional_int("EXPLORE_RANDOM_SEED")
        ),
        ucb_seed_salt=os.getenv("EXPLORE_AGENT57_UCB_SEED_SALT", "").strip(),
        keep_baseline=_env_bool("EXPLORE_AGENT57_KEEP_BASELINE", True),
        lifelong_enabled=lifelong_enabled,
        lifelong_coef=max(0.0, _env_float("EXPLORE_AGENT57_LIFELONG_COEF", 0.01)),
        lifelong_clip=max(0.0, _env_float("EXPLORE_AGENT57_LIFELONG_CLIP", 2.0)),
        lifelong_warmup=max(0, _env_int("EXPLORE_AGENT57_LIFELONG_WARMUP", 64)),
        lifelong_count_decay=min(
            1.0,
            max(0.0, _env_float("EXPLORE_AGENT57_LIFELONG_COUNT_DECAY", 1.0)),
        ),
        lifelong_capacity=max(0, _env_int("EXPLORE_AGENT57_LIFELONG_CAPACITY", 0)),
        lifelong_backend=backend,
        lifelong_key_version=key_version,
        lifelong_include_dataset=_env_bool(
            "EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET", True
        ),
        lifelong_include_task=_env_bool(
            "EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK", False
        ),
        lifelong_include_turn=_env_bool(
            "EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN", False
        ),
        lifelong_obs_mode=obs_mode,
        lifelong_hierarchical=_env_bool(
            "EXPLORE_AGENT57_LIFELONG_HIERARCHICAL",
            key_version == "v2",
        ),
        lifelong_task_weight=max(
            0.0,
            _env_float("EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT", 0.5),
        ),
        lifelong_skill_weight=max(
            0.0,
            _env_float("EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT", 0.35),
        ),
        lifelong_global_weight=max(
            0.0,
            _env_float("EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT", 0.15),
        ),
        trust_gate_mode=trust_gate_mode,
        trust_completed=max(0.0, _env_float("EXPLORE_AGENT57_TRUST_COMPLETED", 1.0)),
        trust_truncated=max(0.0, _env_float("EXPLORE_AGENT57_TRUST_TRUNCATED", 0.3)),
        trust_failed=max(0.0, _env_float("EXPLORE_AGENT57_TRUST_FAILED", 0.1)),
        trust_parse_error=max(0.0, _env_float("EXPLORE_AGENT57_TRUST_PARSE_ERROR", 0.1)),
        trust_warmup=max(0.0, _env_float("EXPLORE_AGENT57_TRUST_WARMUP", 0.3)),
        state_path=_default_state_path(),
        sqlite_busy_timeout_ms=max(
            1,
            _env_int("EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS", 30000),
        ),
        sqlite_wal=_env_bool("EXPLORE_AGENT57_SQLITE_WAL", False),
        success_threshold=_env_float("EXPLORE_AGENT57_SUCCESS_THRESHOLD", 0.0),
    )
