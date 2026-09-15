"""Single home for environment-variable parsing in harborrl.

Before this module existed, every subsystem carried its own copy of
``_env_bool`` / ``_env_int`` / ``_env_float`` (more than a dozen, with subtly
different warning and empty-string behaviour).  Import from here instead of
re-defining them; call sites typically alias the underscore names so existing
code keeps working::

    from harborrl.env import env_bool as _env_bool, env_float as _env_float

Two helper families exist on purpose:

* ``env_bool`` treats an unset *or empty* value as "use the default" — the
  dominant convention in the rollout path.
* ``env_flag`` mirrors ``os.getenv(name, default)`` exactly: an explicitly
  empty value is *falsey*, never replaced by the default.  Reward/algorithm
  toggles rely on this (e.g. ``EXPLORE_ADVANTAGE_BONUS_ENABLED`` falling back
  to another variable's value only when truly unset).

``terminal/runtime.py`` and ``terminal/docker_compose.py`` intentionally keep
their legacy "anything not falsey is true" bool helper; do not "unify" those
two without auditing every call site.

``ENV_VARS`` is the living declaration table for rollout-domain variables:
name -> human-readable summary.  Extend it when adding a variable.
"""

from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in _TRUTHY


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %.6g", name, raw, default)
        return default
    if not math.isfinite(value):
        logger.warning("Non-finite %s=%r; using %.6g", name, raw, default)
        return default
    return value


def env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def env_csv_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default)
    return {part.strip() for part in raw.split(",") if part.strip()}


def env_int_set(name: str, default: str) -> set[int]:
    raw = os.getenv(name, default)
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            logger.warning("Invalid integer %r in %s=%r; skipping", part, name, raw)
    return out


def env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; ignoring", name, raw)
        return None


# Rollout-domain environment variable declaration table.  Shell-side defaults
# live in platform/slime_train.sh and algorithms/dive_po/defaults.sh; this
# table documents what the Python side actually reads.
ENV_VARS: dict[str, str] = {
    # ---- environment connectivity ----
    "ENV_SERVER_URL": "单 worker 时训练进程直连的 worker 地址(多 worker 见 WORKER_URLS/router)",
    "AGENT_SAFETYBENCH_REMOTE_ENV": "1 = agent_safetybench 走远程 worker 而非进程内 runtime",
    "AGENTHARM_REMOTE_ENV": "1 = agentharm 走远程 worker",
    "TAU2_REMOTE_ENV": "1 = tau2 走远程 worker",
    "ENV_HTTP_MAX_RETRIES": "环境 HTTP 默认重试次数(client.py)",
    "ENV_ALLOCATE_MAX_RETRIES": "allocate 重试次数(默认 100,含退避)",
    "ENV_RESET_MAX_RETRIES": "reset 重试次数",
    "ENV_RESET_FRESH_LEASE_RETRIES": "reset 410/5xx 时换新 lease 的重试轮数",
    "ENV_RESET_HTTP_TIMEOUT": "单次 reset HTTP 超时(秒)",
    "ENV_HEARTBEAT_INTERVAL": "rollout 期间后台心跳间隔(秒)",
    "ENV_TASK_CIRCUIT_BREAKER_ENABLED": "task 级熔断开关(默认 1)",
    "ENV_TASK_CIRCUIT_BREAKER_THRESHOLD": "连续失败多少次后开闸(默认 2)",
    "ENV_TASK_CIRCUIT_BREAKER_COOLDOWN": "熔断冷却秒数(默认 1800)",
    "ENV_REMOTE_MAX_ACTIVE_TASKS": "远程环境准入:最大并发 task 数",
    "ENV_REMOTE_MAX_ACTIVE_RUNS": "远程环境准入:最大并发 run 总数(0=不限)",
    "ENV_REMOTE_MAX_RUNS_PER_TASK": "远程环境准入:单 task 最大并发 run",
    "ENV_REMOTE_ADMISSION_TIMEOUT": "准入等待超时(秒)",
    "ENV_REMOTE_MAX_CONCURRENT_CLOSES": "并发 close 上限",
    # ---- rollout shaping ----
    "ALGO": "算法选择(grpo|dapo);sample_builder 用它决定是否启用 DAPO overlong 惩罚",
    "DAPO_OVERLONG_BUFFER_ENABLE": "DAPO overlong 奖励惩罚开关(默认 1)",
    "DAPO_MAX_RESPONSE_LEN": "overlong 判定的最大回复长度(缺省取 args.rollout_max_response_len)",
    "DAPO_OVERLONG_BUFFER_LEN": "overlong 缓冲区长度(默认 4096)",
    "DAPO_OVERLONG_PENALTY_FACTOR": "overlong 惩罚系数(默认 1.0)",
    "MAX_TURN": "单条轨迹最大交互轮数",
    # ---- reward / exploration(dive_po) ----
    "SETA_SAFETY": "SETA 数据源的安全奖励模式(none)",
    "SAFETY_BENCH_REWARD": "Agent-SafetyBench 奖励模式(rule|dense_rule)",
    "AGENTHARM_REWARD": "AgentHarm 奖励模式(rule|dense_rule)",
    "EXPLORATION_PROFILE": "探索配置档位(off|robust_dapo_lite|spear_lite)",
    "EXPLORE_INTRINSIC": "计数式内在奖励开关",
    "EXPLORE_ADVANTAGE_BONUS": "探索 advantage bonus 总开关",
    "EXPLORE_ADVANTAGE_BONUS_ENABLED": "同上(优先于 EXPLORE_ADVANTAGE_BONUS)",
    "EXPLORE_ADVANTAGE_BONUS_MODE": "bonus 注入模式(dual_stream|components)",
    "EXPLORE_TRUNCATION_PENALTY": "截断轨迹惩罚(默认 -0.03)",
    # ---- observability ----
    "TERMINAL_SAVE_TRAJ_DIR": "traj.json/index.jsonl 落盘目录(paths.py 注入)",
    "TRAJECTORY_SAVE_INTERVAL": "轨迹保存间隔(按数据源别名可覆盖)",
    "TRAJECTORY_MAX_TOTAL": "index.jsonl 容量上限(默认 5000)",
    "TERMINAL_STRUCTURED_METRICS": "JSONL 结构化指标开关(默认 1)",
    "TERMINAL_METRICS_JSONL": "结构化指标文件路径(默认 <RUN_DIR>/logs/metrics.jsonl)",
    "TERMINAL_WANDB_METRIC_PROFILE": "wandb 指标集(compact|full)",
    "SWEBENCH_RESULTS_DIR": "设置后 eval 时导出 SWE-bench 官方格式产物",
    "SWEBENCH_EVAL_DATA_PATH": "SWE-bench 实例 ID 数据集路径(导出 coverage 用)",
    # ---- run paths(paths.py 注入) ----
    "RUNS_ROOT": "运行目录根路径(默认仓库 runs/)",
    "RUN_CATEGORY": "运行分类(training|evaluation|testing；debug 自动归入 testing/debug)",
    "RUN_ID": "本次训练 run 标识",
    "RUN_DIR": "runs/<category>/<RUN_ID> 根目录（训练 training、评测 evaluation、测试 testing；debug 在 testing/debug）",
    "RUN_LOG_DIR": "日志目录",
    "TBENCH_OUTPUT_ROOT": "环境侧产物输出根目录",
}
