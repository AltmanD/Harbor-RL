"""Environment registry: one declarative table for every rollout data source.

Before this module, adding an environment meant editing if-else chains in six
places (``rollout/admission.py`` and ``rollout/environment_factory.py`` each
carried their own copy of the ``_uses_local_*`` predicates, plus hardcoded
branches in ``entrypoint.py``, ``sample_builder.py`` and
``trajectory_store.py``).  Everything now keys off :data:`ENV_SPECS`:

* admission / environment_factory ask :func:`local_env_spec` whether a task
  uses an in-process runtime (and which one);
* entrypoint asks :func:`direct_score_source`;
* trajectory_store asks :func:`slug_for` / :func:`interval_candidates_for_slug`.

Adding a new local environment = append one ``EnvSpec`` here and point
``local_runtime`` at its runtime class.  Alias-tolerant lookups (slug,
intervals) intentionally accept the historical aliases, while behavior
decisions (local vs remote, direct score, safety split) match only the
canonical ``data_source`` string, exactly as the previous code did.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class EnvSpec:
    data_source: str
    # (module, class) implementing the in-process runtime; None = remote-only.
    local_runtime: tuple[str, str] | None
    # Env var that forces the remote worker path even when a runtime exists.
    remote_env_flag: str | None
    # Dummy lease id returned by the local client (kept for log compatibility).
    local_lease_id: str | None
    # Whether _safety_split_from_meta distinguishes benign/harmful tasks.
    safety_split: bool
    # True when env.evaluate() already returns a score (skip 2x-1 mapping).
    direct_score: bool
    # Trajectory-store dataset slug and the historical aliases resolving to it.
    slug: str
    slug_aliases: tuple[str, ...]
    interval_arg_names: tuple[str, ...]
    interval_env_names: tuple[str, ...]


SETA_SPEC = EnvSpec(
    data_source="seta",
    local_runtime=None,
    remote_env_flag=None,
    local_lease_id=None,
    safety_split=False,
    direct_score=False,
    slug="seta",
    slug_aliases=("", "terminal_bench", "seta", "seta_env"),
    interval_arg_names=(
        "trajectory_save_interval_seta",
        "trajectory_save_interval_terminal_bench",
    ),
    interval_env_names=("TRAJECTORY_SAVE_INTERVAL_SETA", "SAVE_INTERVAL_SETA"),
)

AGENT_SAFETYBENCH_SPEC = EnvSpec(
    data_source="agent_safetybench",
    local_runtime=(
        "harborrl.environments.agent_safetybench.runtime",
        "AgentSafetyBenchEnv",
    ),
    remote_env_flag="AGENT_SAFETYBENCH_REMOTE_ENV",
    local_lease_id="local-agent-safetybench",
    safety_split=True,
    direct_score=True,
    slug="agent_safetybench",
    slug_aliases=("agent_safetybench", "agent-safety-bench", "safety", "asb"),
    interval_arg_names=(
        "trajectory_save_interval_agent_safetybench",
        "trajectory_save_interval_asb",
        "trajectory_save_interval_safety",
    ),
    interval_env_names=(
        "TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH",
        "TRAJECTORY_SAVE_INTERVAL_ASB",
        "TRAJECTORY_SAVE_INTERVAL_SAFETY",
        "SAVE_INTERVAL_AGENT_SAFETYBENCH",
        "SAVE_INTERVAL_ASB",
        "SAVE_INTERVAL_SAFETY",
    ),
)

AGENTHARM_SPEC = EnvSpec(
    data_source="agentharm",
    local_runtime=("harborrl.environments.agentharm.runtime", "AgentHarmEnv"),
    remote_env_flag="AGENTHARM_REMOTE_ENV",
    local_lease_id="local-agentharm",
    safety_split=True,
    direct_score=True,
    slug="agentharm",
    slug_aliases=("agentharm", "agent_harm", "ah"),
    interval_arg_names=(
        "trajectory_save_interval_agentharm",
        "trajectory_save_interval_agent_harm",
    ),
    interval_env_names=(
        "TRAJECTORY_SAVE_INTERVAL_AGENTHARM",
        "TRAJECTORY_SAVE_INTERVAL_AGENT_HARM",
        "SAVE_INTERVAL_AGENTHARM",
        "SAVE_INTERVAL_AGENT_HARM",
    ),
)

TAU2_SPEC = EnvSpec(
    data_source="tau2",
    local_runtime=("harborrl.environments.tau2.runtime", "Tau2Env"),
    remote_env_flag="TAU2_REMOTE_ENV",
    local_lease_id="local-tau2",
    safety_split=False,
    direct_score=True,
    slug="tau2",
    slug_aliases=("tau2",),
    interval_arg_names=("trajectory_save_interval_tau2",),
    interval_env_names=("TRAJECTORY_SAVE_INTERVAL_TAU2",),
)

ENV_SPECS: tuple[EnvSpec, ...] = (
    SETA_SPEC,
    AGENT_SAFETYBENCH_SPEC,
    AGENTHARM_SPEC,
    TAU2_SPEC,
)

_CANONICAL: dict[str, EnvSpec] = {spec.data_source: spec for spec in ENV_SPECS}
_SLUG_ALIASES: dict[str, EnvSpec] = {
    alias: spec for spec in ENV_SPECS for alias in spec.slug_aliases
}


def _sanitize_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def canonical_spec(data_source: Any) -> EnvSpec | None:
    """Exact ``data_source`` match; None for empty/unknown sources."""
    if not isinstance(data_source, str):
        return None
    return _CANONICAL.get(data_source.strip().lower())


def spec_for_source(data_source: Any) -> EnvSpec:
    """Alias-tolerant lookup used for trajectory naming.

    Unknown sources derive a seta-like remote spec with a sanitized slug and
    default-generated interval variable names (previous fallback behaviour).
    """
    raw = str(data_source or "").strip().lower()
    spec = _SLUG_ALIASES.get(raw)
    if spec is not None:
        return spec
    slug = _sanitize_slug(raw) or "unknown"
    return replace(
        SETA_SPEC,
        data_source=raw or "seta",
        slug=slug,
        interval_arg_names=(f"trajectory_save_interval_{slug}",),
        interval_env_names=(f"TRAJECTORY_SAVE_INTERVAL_{slug.upper()}",),
    )


def local_env_spec(task_meta: dict[str, Any] | None) -> EnvSpec | None:
    """Spec when the task should run in-process; None for the remote worker."""
    if not isinstance(task_meta, dict):
        return None
    spec = canonical_spec(task_meta.get("data_source"))
    if spec is None or spec.local_runtime is None:
        return None
    if spec.remote_env_flag and os.getenv(spec.remote_env_flag, "0") == "1":
        return None
    return spec


def uses_remote_terminal_env(task_meta: dict[str, Any] | None) -> bool:
    return local_env_spec(task_meta) is None


def direct_score_source(data_source: Any) -> bool:
    spec = canonical_spec(data_source)
    return bool(spec and spec.direct_score)


def safety_split_applies(data_source: Any) -> bool:
    spec = canonical_spec(data_source)
    return bool(spec and spec.safety_split)


def slug_for(data_source: Any) -> str:
    return spec_for_source(data_source).slug


def interval_candidates_for_slug(
    dataset_slug: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    for spec in ENV_SPECS:
        if spec.slug == dataset_slug:
            return spec.interval_arg_names, spec.interval_env_names
    return (
        (f"trajectory_save_interval_{dataset_slug}",),
        (f"TRAJECTORY_SAVE_INTERVAL_{dataset_slug.upper()}",),
    )
