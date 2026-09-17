"""Step implementations for rollout.entrypoint.generate().

generate() used to be one ~1300-line function; the sections are now explicit
steps with dataclass state bundles so each piece can be read and tested in
isolation.  Everything here is private to the rollout package and wired only
from entrypoint.generate().
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from slime.utils.types import Sample
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span

from harborrl.algorithms.prm.agent import TerminalPRMAgent
from harborrl.types import (
    Interaction,
    RunContext,
    TaskTimeouts,
    TurnContext,
    TurnResult,
)
from harborrl.environments.registry import (
    direct_score_source,
    slug_for,
    uses_remote_terminal_env as _uses_remote_terminal_env,
)
from harborrl.rollout.admission import (
    _acquire_remote_env_admission,
    _await_with_optional_timeout,
    _is_reset_fresh_lease_retryable,
    _release_remote_env_admission,
    _remote_env_close_semaphore,
    _task_circuit_open_reason,
)
from harborrl.algorithms.dive_po.exploration.rollout_bonus import (
    _AGENT57_CONFIG,
    _EXPLORE_CDE_ACTOR_ENABLED,
    _EXPLORE_CDE_ACTOR_REWARD_GATE,
    _EXPLORE_INTRINSIC_COEF,
    _EXPLORE_INTRINSIC_DECAY_STEPS,
    _EXPLORE_INTRINSIC_ENABLED,
    _EXPLORE_INTRINSIC_GRANULARITY,
    _EXPLORE_INTRINSIC_REDUCER,
    _EXPLORE_INTRINSIC_SCHEDULE,
    _EXPLORE_INTRINSIC_SCOPE,
    _EXPLORE_LPRND_COEF,
    _EXPLORE_LPRND_DECAY_STEPS,
    _EXPLORE_LPRND_ENABLED,
    _EXPLORE_LPRND_SCHEDULE,
    _EXPLORE_SAFETY_FILTER_ENABLED,
    _EXPLORE_SCORE_BONUS_COMPONENTS,
    _agent57_last_episodic_by_turn,
    _agent57_last_episodic_stats,
    _explore_cde_actor_metrics,
    _explore_debug_metrics,
    _explore_episode_signature_novelty,
    _explore_intrinsic_bonus,
    _explore_lprnd_bonus,
    _explore_safety_penalty,
    _explore_schedule_multiplier,
    _explore_score_bonus_from_components,
    _finite_float,
    _iter_explore_actions,
    _turn_uncertainty_metrics,
)
from harborrl.rollout.trajectory_store import (
    _jsonable,
    _sample_or_env_int,
    _trajectory_save_interval,
)
from harborrl.tasks.metadata import _extract_task_meta, _make_task_spec
from harborrl.rollout.sample_builder import (
    _build_agent_safetybench_eval_payload,
    _env_flag,
    _last_eval_details,
    _safety_split_from_meta,
    _sync_reward_aliases,
)
from harborrl.algorithms.dive_po.exploration.agent57.controller import (
    compute_lifelong_bonus as _agent57_compute_lifelong_bonus,
    compute_ngu_lite_bonus as _agent57_compute_ngu_lite_bonus,
    record_arm_event as _agent57_record_arm_event,
)
from harborrl.rollout.backends.factory import (
    _create_sglang_client,
    _normalize_tool_schemas,
)
from harborrl.rollout.environment_factory import (
    _create_env_client,
    _normalize_tau2_conversation_mode,
)
from harborrl.rollout.runner import create_agent_runner, normalize_harness_option
from slime.world_model.metadata import attach_terminal_world_model_metadata

from harborrl.env import (
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
)

logger = logging.getLogger(__name__)


# ── State bundles passed between steps ──────────────────────────────────────


@dataclass
class _RunPlan:
    task_meta: Dict[str, Any]
    data_source: str
    evaluation: bool
    task_spec: Any
    run_ctx: RunContext
    run_ctx_payload: dict[str, Any]
    timeouts: TaskTimeouts
    prm_enable: bool
    prm_coef: float
    traj_save_interval: int
    task_key: str
    log_tag: str
    trace_target: Sample


@dataclass
class _EnvSession:
    env_client: Any = None
    lease_id: Optional[str] = None
    admission_key: Optional[str] = None
    heartbeat_task: asyncio.Task | None = None
    heartbeat_interval: float = 30.0
    user_msg: Optional[str] = None  # None until env reset succeeds
    raw_tools: list = field(default_factory=list)


@dataclass
class _TurnClients:
    sglang_client: Any = None
    agent_runner: Any = None
    prm_agent: Any = None
    agent_type: Optional[str] = None
    tau2_conversation_mode: str = "solo"


@dataclass
class _TurnLoopResult:
    interactions: List[Interaction] = field(default_factory=list)
    turn_records: List[Dict[str, Any]] = field(default_factory=list)
    turn_uncertainty_records: List[Dict[str, Any]] = field(default_factory=list)
    prm_pending: List[tuple[int, asyncio.Task]] = field(default_factory=list)
    final_response: Any = None
    final_model_response: Any = None
    reached_iteration_limit: bool = False
    reached_parse_error_limit: bool = False


# ── Step 1: run configuration ───────────────────────────────────────────────


def _timeout_arg(
    args,
    attr_name: str,
    env_name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    raw = getattr(args, attr_name, None)
    if raw is None:
        raw = os.getenv(env_name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = default
    if value <= 0:
        value = default
    if minimum is not None and value < minimum:
        value = minimum
    return value


def _prepare_rollout_plan(args, sample: Sample, evaluation: bool) -> _RunPlan:
    task_meta = _extract_task_meta(sample)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    data_source = str(task_meta.get("data_source", ""))
    uid = (sample.metadata or {}).get("uid") or uuid.uuid4().hex[:8]
    group_index = int(sample.group_index) if sample.group_index is not None else -1
    sample_index = int(sample.index) if sample.index is not None else -1
    rollout_id = _sample_or_env_int(sample, "rollout_id", "_CURRENT_ROLLOUT_ID")
    train_step = _sample_or_env_int(sample, "train_step", "_CURRENT_TRAIN_STEP")
    rollout_step = _sample_or_env_int(sample, "rollout_step", "_CURRENT_ROLLOUT_STEP")
    task_spec = _make_task_spec(task_meta)
    output_root = (
        getattr(args, "tbench_output_root", None)
        or os.getenv("TBENCH_OUTPUT_ROOT")
        or str(Path(os.getenv("RUN_DIR", "runs/unscoped")) / "environment_outputs")
    )
    run_ctx = RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=Path(output_root) / "AgentRunner_Output",
        rollout_id=rollout_id,
        train_step=train_step,
        rollout_step=rollout_step,
    )

    timeouts = TaskTimeouts(
        ensure_image=_timeout_arg(
            args,
            "ensure_image_timeout",
            "ENSURE_IMAGE_TIMEOUT",
            1200.0,
            minimum=1200.0,
        ),
        reset_session=_timeout_arg(
            args,
            "reset_session_timeout",
            "RESET_SESSION_TIMEOUT",
            600.0,
            minimum=600.0,
        ),
        close_session=_timeout_arg(
            args,
            "close_session_timeout",
            "CLOSE_SESSION_TIMEOUT",
            60.0,
        ),
        eval=_timeout_arg(args, "eval_timeout", "EVAL_TIMEOUT", 600.0),
    )

    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    log_tag = (
        f"[task={task_spec.task_name} uid={run_ctx.uid} "
        f"group_idx={run_ctx.group_index} sample_idx={run_ctx.sample_index}]"
    )
    return _RunPlan(
        task_meta=task_meta,
        data_source=data_source,
        evaluation=bool(evaluation),
        task_spec=task_spec,
        run_ctx=run_ctx,
        run_ctx_payload=run_ctx.to_payload(),
        timeouts=timeouts,
        prm_enable=bool(getattr(args, "prm_enable", False)) and (not evaluation),
        prm_coef=float(getattr(args, "prm_turn_coef", 1.0)),
        traj_save_interval=_trajectory_save_interval(args, data_source=data_source),
        task_key=task_key,
        log_tag=log_tag,
        trace_target=sample,
    )


# ── Step 2: env lease, admission, reset-with-fresh-lease, heartbeat ─────────


async def _open_env_session(plan: _RunPlan, session: _EnvSession) -> None:
    task_meta = plan.task_meta
    run_ctx = plan.run_ctx
    _log_tag = plan.log_tag

    # Admission must precede the remote /allocate call.  Acquiring it after
    # allocation lets every pending rollout consume (or wait for) a worker
    # lease before the client-side cap applies, so two independent clients can
    # overrun the worker's shared capacity and time out while retrying 429s.
    if _uses_remote_terminal_env(task_meta):
        open_reason = _task_circuit_open_reason(plan.task_key)
        if open_reason is not None:
            raise RuntimeError(
                f"TASK_CIRCUIT_OPEN task_key={plan.task_key}: {open_reason}"
            )
        session.admission_key = await _acquire_remote_env_admission(
            plan.task_key,
            log_tag=_log_tag,
        )
    session.env_client, session.lease_id = await _create_env_client(
        _make_task_spec(task_meta), run_ctx, task_meta
    )
    reset_kwargs = {
        "lease_id": session.lease_id,
        "task_meta": task_meta,
        "run_ctx": plan.run_ctx_payload,
        "task_timeouts": plan.timeouts.to_payload(),
    }
    default_reset_http_timeout = (
        float(plan.timeouts.ensure_image) + float(plan.timeouts.reset_session) + 300.0
    )
    reset_http_timeout = _env_float(
        "ENV_RESET_HTTP_TIMEOUT",
        default_reset_http_timeout,
    )
    reset_payload: dict[str, Any] | None = None

    # Reset can fail because the remote worker admitted the lease but could not
    # enter reset before its admission timeout.  In that state the server may
    # clean up the lease and subsequent reset attempts return 410.  Never retry
    # the same lease after those failures; close it best-effort and allocate a
    # fresh lease with a unique request_id instead.
    reset_fresh_lease_retries = (
        max(0, _env_int("ENV_RESET_FRESH_LEASE_RETRIES", 2))
        if _uses_remote_terminal_env(task_meta)
        else 0
    )
    for reset_attempt in range(reset_fresh_lease_retries + 1):
        reset_kwargs["lease_id"] = session.lease_id
        if session.admission_key is not None:
            reset_kwargs["request_id"] = (
                f"{plan.task_key}:{run_ctx.uid}:{run_ctx.group_index}:"
                f"{run_ctx.sample_index}:reset:{reset_attempt}"
            )
        reset_coro = session.env_client.reset(**reset_kwargs)
        try:
            reset_payload = await _await_with_optional_timeout(
                reset_coro,
                reset_http_timeout,
                op_name=f"{_log_tag} env reset",
            )
            break
        except (TimeoutError, asyncio.TimeoutError) as reset_exc:
            should_retry_reset = reset_attempt < reset_fresh_lease_retries
            logger.error(
                "%s Reset timed out after %.1fs on lease %s%s",
                _log_tag,
                reset_http_timeout,
                session.lease_id,
                "; allocating fresh lease" if should_retry_reset else "",
            )
            try:
                await session.env_client.close(session.lease_id)
            except Exception as close_exc:
                logger.debug(
                    "%s Best-effort close after reset timeout: %s",
                    _log_tag,
                    close_exc,
                )
            if not should_retry_reset:
                raise reset_exc
        except Exception as reset_exc:
            should_retry_reset = (
                reset_attempt < reset_fresh_lease_retries
                and _is_reset_fresh_lease_retryable(reset_exc)
            )
            if not should_retry_reset:
                raise
            logger.warning(
                "%s Reset failed on lease %s with retryable remote error; "
                "allocating fresh lease (attempt %d/%d): %s",
                _log_tag,
                session.lease_id,
                reset_attempt + 1,
                reset_fresh_lease_retries,
                reset_exc,
            )
            try:
                await session.env_client.close(session.lease_id)
            except Exception as close_exc:
                logger.debug(
                    "%s Best-effort close after reset failure: %s",
                    _log_tag,
                    close_exc,
                )

        fresh_request_id = (
            f"{plan.task_key}:{run_ctx.uid}:{run_ctx.group_index}:"
            f"{run_ctx.sample_index}:reset-fresh:{reset_attempt + 1}:"
            f"{uuid.uuid4().hex[:8]}"
        )
        allocate_timeout = _env_float("ENV_ALLOCATE_HTTP_TIMEOUT", 300.0)
        fresh_lease = await _await_with_optional_timeout(
            session.env_client.allocate(
                task_key=plan.task_key, request_id=fresh_request_id
            ),
            allocate_timeout,
            op_name=f"{_log_tag} terminal env re-allocate after reset failure",
        )
        session.lease_id = str(fresh_lease["lease_id"])
        logger.info(
            "%s Re-allocated remote terminal env lease=%s after reset failure",
            _log_tag,
            session.lease_id,
        )
    if reset_payload is None:
        raise RuntimeError(f"{_log_tag} env reset did not return a payload")

    session.heartbeat_interval = _env_float("ENV_HEARTBEAT_INTERVAL", 30.0)
    if _uses_remote_terminal_env(task_meta) and session.heartbeat_interval > 0:
        env_client = session.env_client
        lease_id = session.lease_id
        interval = session.heartbeat_interval

        async def _remote_env_heartbeat_loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    await env_client.heartbeat(lease_id)
                except asyncio.CancelledError:
                    raise
                except Exception as heartbeat_exc:
                    logger.warning(
                        "%s Background heartbeat failed for lease %s: %s",
                        _log_tag,
                        lease_id,
                        heartbeat_exc,
                    )
                    if _is_reset_fresh_lease_retryable(heartbeat_exc):
                        return

        session.heartbeat_task = asyncio.create_task(_remote_env_heartbeat_loop())

    session.user_msg = str(reset_payload.get("user_msg", ""))
    session.raw_tools = list(reset_payload.get("tool_schemas", []))
    logger.info("%s Start terminal rollout", _log_tag)


# ── Step 3: sglang / PRM / harness clients ──────────────────────────────────


def _build_turn_clients(
    args,
    state,
    plan: _RunPlan,
    session: _EnvSession,
    clients: _TurnClients,
    sampling_params: Dict[str, Any],
) -> None:
    task_meta = plan.task_meta
    _log_tag = plan.log_tag
    task_spec = _make_task_spec(task_meta)

    tool_schemas = _normalize_tool_schemas(session.raw_tools)
    clients.tau2_conversation_mode = _normalize_tau2_conversation_mode(
        task_meta.get("tau2_mode") or os.getenv("TAU2_MODE", "solo")
    )
    clients.agent_type = normalize_harness_option(
        getattr(args, "harness_option", None)
        or getattr(args, "terminal_agent_type", None)
        or "camel_agent"
    )
    model_type = str(getattr(args, "model_type", "slime-sglang"))
    non_think_mode = bool(getattr(args, "non_think_mode", True))
    non_think_mode_source = str(
        getattr(args, "non_think_mode_source", "prompt")
    ).lower()
    if non_think_mode_source not in {"prompt", "sglang", "both"}:
        non_think_mode_source = "prompt"
    enable_prompt_non_think = non_think_mode and non_think_mode_source in {
        "prompt",
        "both",
    }
    enable_sglang_non_think = non_think_mode and non_think_mode_source in {
        "sglang",
        "both",
    }

    terminal_max_iterations = max(1, int(getattr(args, "max_iteration", 10)))
    terminal_max_parse_errors = max(1, int(getattr(args, "max_parse_errors", 3)))
    max_total_tokens = int(getattr(args, "max_total_tokens", 32768))
    clients.sglang_client = _create_sglang_client(
        args=args,
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        max_total_tokens=max_total_tokens,
        enable_sglang_non_think=enable_sglang_non_think,
    )

    if plan.prm_enable and plan.data_source != "harbor_terminal":
        prm_router_ip = getattr(args, "prm_router_ip", None)
        prm_router_port = getattr(args, "prm_router_port", None)
        if prm_router_ip and prm_router_port:
            prm_sglang_url = f"http://{prm_router_ip}:{prm_router_port}/generate"
        else:
            prm_sglang_url = getattr(args, "prm_sglang_url", None) or os.getenv(
                "PRM_SGLANG_URL", ""
            )
        if not prm_sglang_url:
            raise RuntimeError(
                "prm_enable=True but no PRM endpoint: set prm_router_ip/port, "
                "prm_sglang_url, or PRM_SGLANG_URL env var."
            )
        prm_sampling_params = {
            "temperature": float(getattr(args, "prm_temperature", 0.0)),
            "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 4096)),
        }
        prm_max_total_tokens = int(getattr(args, "prm_max_total_tokens", 16384))
        prm_sglang_client = _create_sglang_client(
            args=args,
            tokenizer=state.tokenizer,
            sampling_params=prm_sampling_params,
            max_total_tokens=prm_max_total_tokens,
            enable_sglang_non_think=True,
            sglang_url=prm_sglang_url,
            max_retries=10,
        )
        clients.prm_agent = TerminalPRMAgent(
            sglang_client=prm_sglang_client,
            task_instruction=task_spec.instruction,
            history_mode=str(getattr(args, "prm_history_mode", "head_tail")),
        )
        logger.info(
            "%s PRM enabled: url=%s coef=%.3f", _log_tag, prm_sglang_url, plan.prm_coef
        )

    clients.agent_runner = create_agent_runner(
        agent_type=clients.agent_type,
        sglang_client=clients.sglang_client,
        model_type=model_type,
        tool_schemas=tool_schemas,
        non_think_mode=enable_prompt_non_think,
        max_total_tokens=max_total_tokens,
        env_client=session.env_client,
        lease_id=session.lease_id,
        run_context=plan.run_ctx,
        task_meta=task_meta,
    )
    clients.agent_runner.reset(session.user_msg)
    clients.agent_runner.set_max_parse_errors(terminal_max_parse_errors)
    clients.agent_runner.set_max_iterations(terminal_max_iterations)


# ── Step 4: the agent <-> env turn loop ─────────────────────────────────────


async def _run_turn_loop(
    plan: _RunPlan,
    session: _EnvSession,
    clients: _TurnClients,
    loop: _TurnLoopResult,
) -> None:
    task_meta = plan.task_meta
    _log_tag = plan.log_tag
    agent_runner = clients.agent_runner
    env_client = session.env_client
    lease_id = session.lease_id
    previous_turn_uncertainty_score: float | None = None

    while True:
        context_result: TurnContext = await agent_runner.get_turn_context()
        if context_result.terminated_response is not None:
            logger.warning("%s Rollout pre-terminated before model turn.", _log_tag)
            loop.final_response = context_result.terminated_response
            break
        if context_result.context_messages is None:
            logger.warning("%s Rollout context is empty; aborting loop.", _log_tag)
            break

        with trace_span(
            plan.trace_target,
            "sglang_generate",
            attrs={"turn_index": agent_runner.model_turn_count},
        ) as generation_span:
            turn_state: TurnResult = await agent_runner.run_model_turn(
                context_result.context_messages
            )
            generation_interactions = (
                getattr(turn_state, "interactions", None) or [turn_state.interaction]
            )
            generation_meta = getattr(generation_interactions[-1], "generation_meta", None)
            if isinstance(generation_meta, dict):
                generation_span.update(build_sglang_meta_trace_attrs(generation_meta))
        turn_interactions = (
            getattr(turn_state, "interactions", None) or [turn_state.interaction]
        )
        turn_uncertainties: list[dict[str, Any]] = []
        for it in turn_interactions:
            uncertainty = _turn_uncertainty_metrics(
                it,
                previous_turn_score=previous_turn_uncertainty_score,
            )
            if uncertainty:
                turn_uncertainties.append(uncertainty)
                score = _finite_float(uncertainty.get("turn_level_score"))
                if score is not None:
                    previous_turn_uncertainty_score = score
        loop.turn_uncertainty_records.extend(turn_uncertainties)
        interaction = turn_interactions[-1]
        turn_idx = int(interaction.turn_idx)
        loop.interactions.extend(turn_interactions)
        sdk_tool_calls = getattr(turn_state.model_response, "tool_calls", None)
        sdk_tool_calls_count = getattr(
            turn_state.model_response,
            "tool_calls_count",
            len(sdk_tool_calls or []),
        )

        current_turn_record: dict[str, Any] = {
            "turn_idx": turn_idx,
            "harness_option": clients.agent_type,
            "context_messages": context_result.context_messages,
            "assistant_output": interaction.output_text or "",
            "finish_reason": interaction.finish_reason,
            "latency_ms": float(interaction.latency_ms),
            "n_input_tokens": len(interaction.input_ids or []),
            "n_output_tokens": len(interaction.output_token_ids or []),
            "parse_error_recorded": bool(turn_state.parse_error_recorded),
            "sdk_model_turns": [
                {
                    "turn_idx": int(it.turn_idx),
                    "assistant_output": it.output_text or "",
                    "finish_reason": it.finish_reason,
                    "latency_ms": float(it.latency_ms),
                    "n_input_tokens": len(it.input_ids or []),
                    "n_output_tokens": len(it.output_token_ids or []),
                    **(
                        {"uncertainty": uncertainty}
                        if uncertainty
                        else {}
                    ),
                }
                for it, uncertainty in zip(
                    turn_interactions,
                    turn_uncertainties or [{} for _ in turn_interactions],
                )
            ],
            "sdk_tool_calls": _jsonable(sdk_tool_calls) if sdk_tool_calls else [],
            "sdk_tool_calls_count": int(sdk_tool_calls_count or 0),
            "tool_calls": [],
        }
        if turn_uncertainties:
            current_turn_record["uncertainty"] = turn_uncertainties[-1]
        if sdk_tool_calls:
            for call in _jsonable(sdk_tool_calls):
                if isinstance(call, dict):
                    normalized_call = dict(call)
                    normalized_call.setdefault("source", "harness-sdk")
                    current_turn_record["tool_calls"].append(normalized_call)
        loop.turn_records.append(current_turn_record)

        if clients.prm_agent is not None:
            tool_calls_for_prm = [
                {"tool_name": tc.tool_name, "args": tc.args}
                for tc in (turn_state.tool_call_requests or [])
            ]
            if not tool_calls_for_prm and sdk_tool_calls:
                tool_calls_for_prm = [
                    call for call in _jsonable(sdk_tool_calls)
                    if isinstance(call, dict)
                ]
            clients.prm_agent.record_model_turn(
                turn_idx,
                assistant_text=interaction.output_text or "",
                tool_calls=tool_calls_for_prm or None,
                parse_error_recorded=turn_state.parse_error_recorded,
                finish_reason=interaction.finish_reason,
            )

        if turn_state.terminated_response is not None:
            logger.warning(
                "%s Rollout terminated during model turn %d.", _log_tag, turn_idx
            )
            loop.final_response = turn_state.terminated_response
            break
        if turn_state.model_response is None:
            logger.warning(
                "%s Model turn %d returned empty model_response.",
                _log_tag,
                turn_idx,
            )
            break

        should_continue_loop = False
        if tool_call_requests := turn_state.tool_call_requests:
            logger.info(
                "%s Turn %d: executing %d tool call(s).",
                _log_tag,
                turn_idx,
                len(tool_call_requests),
            )
            for tool_call_request in tool_call_requests:
                assert env_client is not None and lease_id is not None
                if session.heartbeat_interval <= 0:
                    # Background heartbeat is disabled; keep the lease alive
                    # inline instead of paying one extra HTTP roundtrip per
                    # tool call when the 30s loop is already doing it.
                    await env_client.heartbeat(lease_id)
                with trace_span(
                    plan.trace_target,
                    "environment_tool",
                    attrs={
                        "turn_index": turn_idx,
                        "tool_name": tool_call_request.tool_name,
                    },
                ):
                    raw_result = await env_client.exec_tool(
                        lease_id,
                        tool_call_request.tool_name,
                        tool_call_request.args,
                    )
                agent_runner.record_tool_result(tool_call_request, raw_result)
                if clients.prm_agent is not None:
                    clients.prm_agent.record_tool_result(
                        turn_idx, tool_call_request, raw_result
                    )
                current_turn_record["tool_calls"].append({
                    "tool_call_id": getattr(tool_call_request, "tool_call_id", None),
                    "tool_name": tool_call_request.tool_name,
                    "args": tool_call_request.args,
                    "result": raw_result[:4096] if isinstance(raw_result, str) else str(raw_result)[:4096],
                })
            should_continue_loop = True

        if turn_state.parse_error_recorded:
            logger.warning(
                "%s Turn %d: tool-call parse error.",
                _log_tag,
                turn_idx,
            )
            should_continue_loop = True

        if clients.prm_agent is not None:
            task = asyncio.create_task(clients.prm_agent.judge_turn(turn_idx))
            loop.prm_pending.append((turn_idx, task))

        if should_continue_loop:
            if (
                turn_state.parse_error_recorded
                and agent_runner.reached_parse_error_limit()
            ):
                logger.error(
                    "%s Max parse errors (%d) reached at turn %d.",
                    _log_tag,
                    agent_runner.max_parse_errors,
                    turn_idx,
                )
                loop.reached_parse_error_limit = True
                loop.final_model_response = turn_state.model_response
                break
            if agent_runner.reached_iteration_limit():
                logger.warning(
                    "%s Max iterations (%d) reached.",
                    _log_tag,
                    agent_runner.max_iterations,
                )
                loop.reached_iteration_limit = True
                loop.final_model_response = turn_state.model_response
                break
            continue

        if (
            task_meta.get("data_source") == "tau2"
            and clients.tau2_conversation_mode == "non_solo"
            and env_client is not None
            and lease_id is not None
        ):
            follow_up = await env_client.agent_reply(
                lease_id,
                interaction.output_text or "",
            )
            follow_up_message = str(follow_up.get("user_message", "") or "").strip()
            if follow_up.get("continue") and follow_up_message:
                agent_runner.record_user_message(follow_up_message)
                current_turn_record["env_user_message"] = follow_up_message
                if agent_runner.reached_iteration_limit():
                    logger.warning(
                        "%s Max iterations (%d) reached after non-solo follow-up.",
                        _log_tag,
                        agent_runner.max_iterations,
                    )
                    loop.reached_iteration_limit = True
                    loop.final_model_response = turn_state.model_response
                    break
                continue

        loop.final_model_response = turn_state.model_response
        break

    if loop.final_response is None and loop.final_model_response is not None:
        loop.final_response = agent_runner.finalize_response(loop.final_model_response)


def _decide_status(plan: _RunPlan, clients: _TurnClients, loop: _TurnLoopResult):
    finish_reasons = loop.final_response.info.get("termination_reasons", [])
    is_aborted = not loop.final_response.msg

    if loop.final_response.terminated and "max_tokens_exceeded" in finish_reasons:
        status = Sample.Status.TRUNCATED
    elif loop.reached_iteration_limit:
        status = Sample.Status.TRUNCATED
    elif loop.reached_parse_error_limit:
        status = Sample.Status.FAILED
    elif is_aborted:
        status = Sample.Status.ABORTED
    else:
        status = Sample.Status.COMPLETED
    logger.info(
        "%s Rollout finished: status=%s turns=%d parse_errors=%d",
        plan.log_tag,
        status,
        clients.agent_runner.model_turn_count,
        clients.agent_runner.parse_error_count,
    )
    return status, is_aborted


# ── Step 5: environment scoring ─────────────────────────────────────────────


async def _evaluate_outcome(
    plan: _RunPlan,
    session: _EnvSession,
    clients: _TurnClients,
    loop: _TurnLoopResult,
    status,
    is_aborted: bool,
):
    reward = 0.0
    eval_error: str | None = None
    eval_details: dict[str, Any] | None = None
    deferred_sweverified = (
        plan.data_source == "sweverified"
        and _env_bool("SWEBENCH_DEFER_GRADING", False)
    )

    if (not is_aborted) and (
        status != Sample.Status.FAILED or deferred_sweverified
    ):
        try:
            assert session.env_client is not None and session.lease_id is not None
            await session.env_client.heartbeat(session.lease_id)
            eval_payload = None
            if direct_score_source(plan.data_source):
                eval_payload = _build_agent_safetybench_eval_payload(
                    task_meta=plan.task_meta,
                    turn_records=loop.turn_records,
                    final_response=loop.final_response,
                    interactions=loop.interactions,
                    status=status,
                    parse_error_count=clients.agent_runner.parse_error_count,
                )
            elif deferred_sweverified:
                eval_payload = {"swebench_defer_grading": True}
            with trace_span(plan.trace_target, "environment_evaluate"):
                raw_score = await session.env_client.evaluate(
                    session.lease_id, trajectory=eval_payload
                )
            reward = float(raw_score)
            eval_details = _last_eval_details(session.env_client)
            logger.info("%s Evaluation reward=%.4f", plan.log_tag, reward)
            if eval_details:
                reason = eval_details.get("reason")
                base = eval_details.get("base")
                split = _safety_split_from_meta(plan.task_meta)
                logger.info(
                    "%s Reward details: source=%s split=%s mode=%s reason=%s base=%s "
                    "refused=%s verbal_refused=%s tools=%s turns=%s parse_errors=%s",
                    plan.log_tag,
                    plan.data_source or "seta",
                    split,
                    eval_details.get("mode"),
                    reason,
                    base,
                    eval_details.get("refused"),
                    eval_details.get("verbal_refused", eval_details.get("text_refused")),
                    eval_details.get("n_tool_calls"),
                    eval_details.get("n_turns"),
                    eval_details.get("parse_errors"),
                )
        except Exception as exc:
            eval_error = f"{type(exc).__name__}: {exc}"
            eval_details = _last_eval_details(session.env_client)
            status = Sample.Status.FAILED
            reward = 0.0
            logger.error(
                "%s Evaluation failed, marking FAILED: %s",
                plan.log_tag,
                eval_error,
            )
    return reward, eval_details, eval_error, status


# ── Step 6: PRM collection ──────────────────────────────────────────────────


async def _collect_prm_scores(
    plan: _RunPlan,
    clients: _TurnClients,
    loop: _TurnLoopResult,
    sample: Sample,
) -> dict[int, float]:
    prm_turn_scores: dict[int, float] = {}
    if clients.prm_agent is None:
        return prm_turn_scores
    prm_turn_details: list[dict[str, Any]] = []
    for turn_idx, prm_task in loop.prm_pending:
        try:
            output_text, score = await prm_task
            prm_turn_scores[turn_idx] = float(score)
            prm_turn_details.append(
                {
                    "turn_idx": turn_idx,
                    "score": float(score),
                    "output_text": output_text,
                }
            )
            logger.info(
                "%s PRM judge turn %d score=%.4f, output_text=%s",
                plan.log_tag,
                turn_idx,
                float(score),
                output_text.replace("\n", ""),
            )
        except Exception as exc:
            logger.warning(
                "%s PRM judge failed for turn %d (ignored): %s",
                plan.log_tag,
                turn_idx,
                exc,
            )
            prm_turn_scores[turn_idx] = 0.0
            prm_turn_details.append(
                {"turn_idx": turn_idx, "score": 0.0, "error": str(exc)}
            )

    sample.metadata["prm"] = {
        "enabled": True,
        "coef": plan.prm_coef,
        "turn_scores": prm_turn_scores,
        "turn_details": prm_turn_details,
    }
    return prm_turn_scores


# ── Step 7: exploration bonus injection (intrinsic/safety/LP-RND/Agent57/CDE) ──


def _agent57_normalized_outcome(
    samples: List[Sample],
    data_source: str | None,
) -> float | None:
    """Return the task-native [0, 1] outcome used by Agent57's UCB.

    Converted SETA rows deliberately retain ``data_source=terminal_bench`` for
    reward routing.  Checking only for the literal string ``seta`` therefore
    silently fed the shaped score (including truncation/overlong penalties)
    into the controller and classified positive partial credit as failure.
    Resolve aliases through the environment registry and prefer ``raw_score``.
    """
    if slug_for(data_source) != "seta":
        return None
    values: list[float] = []
    for item in samples:
        if not isinstance(item.reward, dict):
            continue
        raw_score = item.reward.get("raw_score", item.reward.get("accuracy"))
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        values.append(min(1.0, max(0.0, value)))
    return sum(values) / len(values) if values else None


def _inject_exploration_bonuses(
    samples: List[Sample],
    *,
    sample: Sample,
    plan: _RunPlan,
    clients: _TurnClients,
    loop: _TurnLoopResult,
    status,
    eval_error: str | None,
) -> None:
    # Evaluation must be observational. In particular, fixed-set evaluation
    # must not advance Agent57 lifelong counts/UCB events or inject an
    # exploration reward that changes the reported task score.
    if plan.evaluation:
        return
    if not (
        _EXPLORE_INTRINSIC_ENABLED
        or _EXPLORE_SAFETY_FILTER_ENABLED
        or _EXPLORE_LPRND_ENABLED
        or _EXPLORE_CDE_ACTOR_ENABLED
        or _AGENT57_CONFIG.active
    ):
        return
    turn_records = loop.turn_records
    interactions = loop.interactions
    run_ctx = plan.run_ctx
    data_source = plan.data_source
    parse_error_count = clients.agent_runner.parse_error_count

    _intr_bonus = _explore_intrinsic_bonus(turn_records)
    _intr_schedule_multiplier = _explore_schedule_multiplier(
        _EXPLORE_INTRINSIC_SCHEDULE,
        run_ctx.train_step,
        _EXPLORE_INTRINSIC_DECAY_STEPS,
    )
    _intr_effective_coef = _EXPLORE_INTRINSIC_COEF * _intr_schedule_multiplier
    _intr_scaled = _intr_bonus * _intr_effective_coef
    _safe_penalty = _explore_safety_penalty(turn_records)
    _lprnd_raw = _explore_lprnd_bonus(interactions)
    _lprnd_schedule_multiplier = _explore_schedule_multiplier(
        _EXPLORE_LPRND_SCHEDULE,
        run_ctx.train_step,
        _EXPLORE_LPRND_DECAY_STEPS,
    )
    _lprnd_effective_coef = _EXPLORE_LPRND_COEF * _lprnd_schedule_multiplier
    _lprnd_bonus = _lprnd_raw * _lprnd_effective_coef
    try:
        _agent57_arm_id = int(
            (sample.metadata or {}).get(
                "agent57_arm_id",
                int(sample.index or 0) % max(1, _AGENT57_CONFIG.k),
            )
        )
    except (TypeError, ValueError):
        _agent57_arm_id = 0
    _agent57_lifelong_metadata = dict(sample.metadata or {})
    _agent57_lifelong_metadata.setdefault("data_source", data_source)
    _agent57_metrics = _agent57_compute_lifelong_bonus(
        config=_AGENT57_CONFIG,
        arm_id=_agent57_arm_id,
        actions=_iter_explore_actions(turn_records),
        turn_records=turn_records,
        status=status,
        parse_error_count=parse_error_count,
        metadata=_agent57_lifelong_metadata,
        state_update_allowed=eval_error is None,
    )
    _agent57_bonus = float(
        _agent57_metrics.get("explore_agent57_lifelong_bonus", 0.0) or 0.0
    )
    if _AGENT57_CONFIG.active and _AGENT57_CONFIG.combine_mode == "ngu_lite":
        _agent57_episodic = (
            _intr_bonus
            if _AGENT57_CONFIG.ngu_episodic_source == "intrinsic"
            else _explore_episode_signature_novelty(
                turn_records,
                reducer=_AGENT57_CONFIG.ngu_episodic_reducer,
            )
        )
        _agent57_ngu_metrics = _agent57_compute_ngu_lite_bonus(
            config=_AGENT57_CONFIG,
            arm_id=_agent57_arm_id,
            episodic_novelty=_agent57_episodic,
            lifelong_raw=float(
                _agent57_metrics.get("explore_agent57_lifelong_raw", 0.0) or 0.0
            ),
            lifelong_eligible=bool(
                _agent57_metrics.get("explore_agent57_lifelong_eligible", 0.0)
            ),
            trust_gate=float(
                _agent57_metrics.get("explore_agent57_trust", 1.0) or 0.0
            ),
            life_mod_override=_agent57_metrics.get("explore_agent57_ngu_life_mod"),
        )
        _agent57_metrics.update(_agent57_ngu_metrics)
        _agent57_metrics.update(_agent57_last_episodic_stats())
        _agent57_bonus = float(
            _agent57_ngu_metrics.get("explore_agent57_ngu_bonus", 0.0) or 0.0
        )
    _base_score_values = []
    for _sample in samples:
        if isinstance(_sample.reward, dict) and "score" in _sample.reward:
            try:
                _base_score_values.append(float(_sample.reward["score"]))
            except (TypeError, ValueError):
                pass
    _base_score_mean = (
        sum(_base_score_values) / len(_base_score_values) if _base_score_values else 0.0
    )
    _agent57_normalized_score_mean = _agent57_normalized_outcome(
        samples,
        data_source,
    )
    _cde_actor = _explore_cde_actor_metrics(
        interactions,
        _base_score_mean,
        run_ctx.train_step,
    )
    _cde_actor_bonus = _cde_actor["bonus"]
    _intr_for_total = (
        0.0
        if (_AGENT57_CONFIG.active and _AGENT57_CONFIG.combine_mode == "ngu_lite")
        else _intr_scaled
    )
    _explore_total = (
        _intr_for_total
        + _safe_penalty
        + _lprnd_bonus
        + _agent57_bonus
        + _cde_actor_bonus
    )
    _explore_score_bonus = _explore_score_bonus_from_components(
        _EXPLORE_SCORE_BONUS_COMPONENTS,
        intrinsic=_intr_for_total,
        safety=_safe_penalty,
        lprnd=_lprnd_bonus,
        agent57=_agent57_bonus,
        cde_actor=_cde_actor_bonus,
    )
    _explore_debug = _explore_debug_metrics(
        status=status,
        base_score_mean=_base_score_mean,
        total_bonus=_explore_total,
        intrinsic_scaled=_intr_for_total,
        safety_penalty=_safe_penalty,
        lprnd_bonus=_lprnd_bonus,
        agent57_bonus=_agent57_bonus,
        cde_actor=_cde_actor,
        turn_records=turn_records,
        parse_error_count=parse_error_count,
    )
    _agent57_record_arm_event(
        config=_AGENT57_CONFIG,
        arm_id=_agent57_arm_id,
        base_score=_base_score_mean,
        final_score=_base_score_mean + _explore_score_bonus,
        status=status,
        parse_error_count=parse_error_count,
        bonus=_agent57_bonus,
        dataset=data_source,
        normalized_base_score=_agent57_normalized_score_mean,
        success_score=_agent57_normalized_score_mean,
        infra_failure=eval_error is not None,
    )
    for s in samples:
        if isinstance(s.reward, dict) and "score" in s.reward:
            s.reward["score"] += _explore_score_bonus
            s.reward["explore_intrinsic"] = _intr_bonus
            s.reward["explore_intrinsic_scaled"] = _intr_scaled
            s.reward["explore_intrinsic_in_total"] = _intr_for_total
            s.reward["explore_intrinsic_coef"] = _EXPLORE_INTRINSIC_COEF
            s.reward["explore_intrinsic_effective_coef"] = _intr_effective_coef
            s.reward["explore_intrinsic_schedule"] = _EXPLORE_INTRINSIC_SCHEDULE
            s.reward["explore_intrinsic_decay_steps"] = _EXPLORE_INTRINSIC_DECAY_STEPS
            s.reward["explore_intrinsic_reducer"] = _EXPLORE_INTRINSIC_REDUCER
            s.reward["explore_intrinsic_schedule_multiplier"] = _intr_schedule_multiplier
            s.reward["explore_intrinsic_granularity"] = _EXPLORE_INTRINSIC_GRANULARITY
            s.reward["explore_intrinsic_scope"] = _EXPLORE_INTRINSIC_SCOPE
            s.reward["explore_safety_penalty"] = _safe_penalty
            s.reward["explore_lprnd"] = _lprnd_bonus
            s.reward["explore_lprnd_raw"] = _lprnd_raw
            s.reward["explore_lprnd_coef"] = _EXPLORE_LPRND_COEF
            s.reward["explore_lprnd_effective_coef"] = _lprnd_effective_coef
            s.reward["explore_lprnd_schedule"] = _EXPLORE_LPRND_SCHEDULE
            s.reward["explore_lprnd_decay_steps"] = _EXPLORE_LPRND_DECAY_STEPS
            s.reward["explore_lprnd_schedule_multiplier"] = _lprnd_schedule_multiplier
            if _AGENT57_CONFIG.active:
                if not isinstance(s.metadata, dict):
                    s.metadata = {}
                s.reward.update(_agent57_metrics)
                try:
                    _turn_idx = int(s.metadata.get("turn_idx", -1))
                except (TypeError, ValueError):
                    _turn_idx = -1
                _turn_episodic = float(
                    _agent57_last_episodic_by_turn().get(_turn_idx, 0.0)
                )
                _turn_life_mod = float(
                    _agent57_metrics.get("explore_agent57_ngu_life_mod", 1.0) or 1.0
                )
                s.reward["explore_agent57_turn_episodic"] = _turn_episodic
                s.reward["explore_agent57_turn_intrinsic_signal"] = (
                    _turn_episodic * _turn_life_mod
                )
                s.metadata["agent57"] = {
                    "enabled": bool(_AGENT57_CONFIG.enabled),
                    "arm_id": _agent57_arm_id,
                    "k": _AGENT57_CONFIG.k,
                    "beta": _AGENT57_CONFIG.beta_for_arm(_agent57_arm_id),
                    "controller": _AGENT57_CONFIG.controller,
                    "combine_mode": _AGENT57_CONFIG.combine_mode,
                    "lifelong_enabled": bool(_AGENT57_CONFIG.lifelong_enabled),
                    "lifelong_backend": _AGENT57_CONFIG.lifelong_backend,
                    "lifelong_key_version": _AGENT57_CONFIG.lifelong_key_version,
                    "bonus": _agent57_bonus,
                    "lifelong_bonus": _agent57_metrics.get(
                        "explore_agent57_lifelong_bonus", 0.0
                    ),
                    "ngu_bonus": _agent57_metrics.get("explore_agent57_ngu_bonus", 0.0),
                    "lifelong_raw": _agent57_metrics.get(
                        "explore_agent57_lifelong_raw", 0.0
                    ),
                    "lifelong_eligible": _agent57_metrics.get(
                        "explore_agent57_lifelong_eligible", 0.0
                    ),
                }
            if _EXPLORE_CDE_ACTOR_ENABLED:
                s.reward["explore_cde_actor_bonus"] = _cde_actor_bonus
                s.reward["explore_cde_actor_log_ppl"] = _cde_actor["log_ppl"]
                s.reward["explore_cde_actor_omega"] = _cde_actor["omega"]
                s.reward["explore_cde_actor_alpha"] = _cde_actor["alpha"]
                s.reward["explore_cde_actor_kappa"] = _cde_actor["kappa"]
                s.reward["explore_cde_actor_reward_gate"] = _EXPLORE_CDE_ACTOR_REWARD_GATE
                s.reward["explore_cde_actor_eligible"] = _cde_actor["eligible"]
                s.reward["explore_cde_actor_decay_steps"] = _cde_actor["decay_steps"]
                s.reward["explore_cde_actor_base_mean"] = _cde_actor["base_score_mean"]
                s.reward["explore_cde_actor_base_magnitude"] = _cde_actor["base_score_magnitude"]
                s.reward["explore_cde_actor_cap"] = _cde_actor["cap"]
                s.reward["explore_cde_actor_scaled"] = _cde_actor["scaled"]
                s.reward["explore_cde_actor_clipped"] = _cde_actor["clipped"]
            s.reward["explore_total_bonus"] = _explore_score_bonus
            s.reward["explore_score_bonus"] = _explore_score_bonus
            s.reward["explore_all_bonus"] = _explore_total
            s.reward["explore_score_bonus_components"] = _EXPLORE_SCORE_BONUS_COMPONENTS
            s.reward.update(_explore_debug)
            _sync_reward_aliases(s.reward)


# ── Step 8: per-sample metadata finalization ────────────────────────────────


def _finalize_sample_metadata(
    samples: List[Sample],
    *,
    args: Any,
    plan: _RunPlan,
    clients: _TurnClients,
    loop: _TurnLoopResult,
    trajectory_uncertainty: dict[str, Any] | None,
    status: Any,
    eval_details: dict[str, Any] | None,
    eval_error: str | None,
) -> None:
    run_ctx = plan.run_ctx
    agent_runner = clients.agent_runner
    turn_uncertainty_by_idx = {
        int(r["turn_idx"]): r
        for r in loop.turn_uncertainty_records
        if isinstance(r, dict) and r.get("turn_idx") is not None
    }
    for s in samples:
        s.metadata["train_step"] = run_ctx.train_step
        s.metadata["rollout_step"] = run_ctx.rollout_step
        s.metadata["rollout_id"] = run_ctx.rollout_id
        s.metadata["uid"] = run_ctx.uid
        s.metadata["model_turn_count"] = agent_runner.model_turn_count
        s.metadata["parse_error_count"] = agent_runner.parse_error_count
        s.metadata["data_source"] = plan.data_source or s.metadata.get("data_source")
        s.metadata["safety_split"] = _safety_split_from_meta(plan.task_meta)
        claude_backend = str(os.getenv("CLAUDE_CODE_LLM_BACKEND", "sglang")).strip().lower()
        claude_sglang_backend = claude_backend.replace("_", "-") in {
            "sglang",
            "qwen",
            "qwen-sglang",
            "local",
            "local-sglang",
        }
        if clients.agent_type == "claude-code" and _env_flag(
            "CLAUDE_CODE_MARK_NON_TRAINABLE",
            not claude_sglang_backend,
        ):
            s.remove_sample = True
            s.metadata["non_trainable"] = True
            s.metadata["non_trainable_reason"] = (
                "claude-code CLI uses an external model path without "
                "terminal-rl policy logprobs"
            )
        if trajectory_uncertainty:
            s.metadata["trajectory_uncertainty"] = trajectory_uncertainty
        turn_uncertainty = turn_uncertainty_by_idx.get(
            int(s.metadata.get("turn_idx", -1))
        )
        if turn_uncertainty:
            s.metadata["turn_uncertainty"] = turn_uncertainty
        if eval_details is not None:
            s.metadata["reward_details"] = _jsonable(eval_details)
        if eval_error is not None:
            s.metadata["evaluation_failed"] = True
            s.metadata["evaluation_error"] = eval_error
    # Records are opt-in and remain separate from reward/advantage fields. The
    # additive loss hook consumes only explicit precomputed latent tensors.
    attach_terminal_world_model_metadata(
        args=args,
        samples=samples,
        turn_records=loop.turn_records,
        task_meta=plan.task_meta,
        run_ctx=run_ctx,
        status=status,
        eval_details=eval_details,
        eval_error=eval_error,
    )


# ── Failure specimen (except path) ──────────────────────────────────────────


def _failure_specimen_record(
    *,
    plan: _RunPlan,
    session: _EnvSession,
    clients: _TurnClients,
    exc: BaseException,
) -> dict[str, Any]:
    agent_artifacts: dict[str, Any] = {}
    rollout_agent = (
        getattr(clients.agent_runner, "_rollout_agent", None)
        if clients.agent_runner is not None
        else None
    )
    for attr, key in (
        ("_local_run_dir", "local_run_dir"),
        ("_workspace", "workspace"),
        ("_stdout_path", "stdout_path"),
        ("_stderr_path", "stderr_path"),
        ("_tool_log_path", "tool_calls_path"),
        ("_qwen_records_path", "qwen_gateway_records_path"),
    ):
        value = getattr(rollout_agent, attr, None) if rollout_agent is not None else None
        if value is not None:
            agent_artifacts[key] = str(value)
    return {
        "turn_idx": 0,
        "harness_option": clients.agent_type,
        "context_messages": (
            [] if session.user_msg is None
            else [{"role": "user", "content": session.user_msg}]
        ),
        "assistant_output": "",
        "finish_reason": "generate_failed",
        "latency_ms": 0.0,
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "parse_error_recorded": False,
        "tool_calls": [],
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "agent_artifacts": agent_artifacts,
    }


# ── Cleanup (finally path) ──────────────────────────────────────────────────


async def _close_rollout_session(
    plan: _RunPlan,
    session: _EnvSession,
    clients: _TurnClients,
    loop: _TurnLoopResult,
) -> None:
    _log_tag = plan.log_tag
    if session.heartbeat_task is not None:
        session.heartbeat_task.cancel()
        try:
            await session.heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception as heartbeat_exc:
            logger.debug(
                "%s Background heartbeat task ended with error: %s",
                _log_tag,
                heartbeat_exc,
            )

    for _turn_idx, t in loop.prm_pending:
        if not t.done():
            t.cancel()

    if clients.agent_runner is not None:
        try:
            await clients.agent_runner.close()
        except Exception as exc:
            logger.debug("%s Agent runner close ignored: %s", _log_tag, exc)

    try:
        if session.env_client is not None and session.lease_id is not None:
            try:
                close_timeout = _env_float(
                    "ENV_CLOSE_HTTP_TIMEOUT",
                    float(plan.timeouts.close_session) + 30.0,
                )
                close_sem = (
                    _remote_env_close_semaphore()
                    if session.admission_key is not None
                    else None
                )
                if close_sem is None:
                    await _await_with_optional_timeout(
                        session.env_client.close(session.lease_id),
                        close_timeout,
                        op_name=f"{_log_tag} env close",
                    )
                else:
                    async with close_sem:
                        await _await_with_optional_timeout(
                            session.env_client.close(session.lease_id),
                            close_timeout,
                            op_name=f"{_log_tag} env close",
                        )
            except Exception as exc:
                logger.debug(
                    "%s Best-effort remote close failed lease=%s: %s",
                    _log_tag,
                    session.lease_id,
                    exc,
                )
    finally:
        if session.admission_key is not None:
            await _release_remote_env_admission(session.admission_key)
