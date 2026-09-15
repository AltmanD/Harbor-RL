from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample
from slime.utils.trace_utils import bind_trace, trace_span

from harborrl.environments.registry import (
    direct_score_source,
    uses_remote_terminal_env as _uses_remote_terminal_env,
)
from harborrl.rollout.admission import (
    _task_circuit_record_failure,
    _task_circuit_record_success,
)
from harborrl.algorithms.dive_po.exploration.rollout_bonus import (
    _finite_float,
    _summarize_turn_uncertainty,
)
from harborrl.rollout.generate_steps import (
    _EnvSession,
    _TurnClients,
    _TurnLoopResult,
    _build_turn_clients,
    _close_rollout_session,
    _collect_prm_scores,
    _decide_status,
    _evaluate_outcome,
    _failure_specimen_record,
    _finalize_sample_metadata,
    _inject_exploration_bonuses,
    _open_env_session,
    _prepare_rollout_plan,
    _run_turn_loop,
)
from harborrl.rollout.trajectory_store import _save_rollout_artifacts
from harborrl.rollout.sample_builder import (
    _build_samples,
    _dapo_overlong_cfg,
    _mark_non_trainable_samples,
    _sync_reward_aliases,
)

from harborrl.env import env_bool as _env_bool

logger = logging.getLogger(__name__)


async def generate(
    args,
    sample: Sample,
    sampling_params: Dict[str, Any],
    evaluation: bool = False,
) -> List[Sample]:
    """Slime custom generate hook: orchestrates the per-sample rollout steps.

    The step implementations live in rollout/generate_steps.py; the state
    bundles (_RunPlan/_EnvSession/_TurnClients/_TurnLoopResult) are created
    up front and mutated progressively so the except/finally paths observe
    partial state exactly like the old monolith did.
    """
    state = GenerateState(args)
    bind_trace(sample)
    plan = _prepare_rollout_plan(args, sample, evaluation)
    session = _EnvSession()
    clients = _TurnClients()
    loop = _TurnLoopResult()

    try:
        with trace_span(sample, "environment_open"):
            await _open_env_session(plan, session)
        _build_turn_clients(args, state, plan, session, clients, sampling_params)
        with trace_span(sample, "agent_turn_loop"):
            await _run_turn_loop(plan, session, clients, loop)

        if loop.final_response is None:
            logger.error(
                "%s No final response produced; mark sample aborted.", plan.log_tag
            )
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            return [sample]

        status, is_aborted = _decide_status(plan, clients, loop)
        reward, eval_details, eval_error, status = await _evaluate_outcome(
            plan, session, clients, loop, status, is_aborted
        )

        if not loop.interactions:
            logger.warning("%s No interactions recorded; remove sample.", plan.log_tag)
            sample.status = status
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            return [sample]

        trajectory_uncertainty = _summarize_turn_uncertainty(
            loop.turn_uncertainty_records,
            run_ctx=plan.run_ctx,
        )
        if trajectory_uncertainty:
            sample.metadata["trajectory_uncertainty"] = trajectory_uncertainty
            mean_uncertainty = _finite_float(
                trajectory_uncertainty.get("mean_turn_level_uncertainty")
            )
            mean_delta = _finite_float(trajectory_uncertainty.get("mean_abs_score_delta"))
            logger.info(
                "%s Turn uncertainty: available=%s/%s mean_nll=%s "
                "mean_abs_delta=%s low_progress=%s/%s",
                plan.log_tag,
                trajectory_uncertainty.get("available_turn_count"),
                trajectory_uncertainty.get("turn_count"),
                f"{mean_uncertainty:.4f}" if mean_uncertainty is not None else "n/a",
                f"{mean_delta:.4f}" if mean_delta is not None else "n/a",
                trajectory_uncertainty.get("low_progress_turn_count"),
                trajectory_uncertainty.get("available_turn_count"),
            )

        prm_turn_scores = await _collect_prm_scores(plan, clients, loop, sample)
        # Build training samples
        dapo_overlong_cfg = _dapo_overlong_cfg(args)
        samples = _build_samples(
            interactions=loop.interactions,
            base_sample=sample,
            outcome=reward,
            status=status,
            prm_turn_scores=(prm_turn_scores if clients.prm_agent is not None else None),
            prm_coef=plan.prm_coef,
            discount=1.0,
            encourage=False,
            outcome_is_score=direct_score_source(plan.data_source),
            penalize_short_response=not direct_score_source(plan.data_source),
            dapo_overlong_cfg=dapo_overlong_cfg,
        )
        # AgenticRL emits one training Sample per turn.  All turn samples are
        # deep copies of the trajectory carrier, so retaining the trace on
        # every turn would multiply request timing counts.  Keep one canonical
        # carrier per trajectory for accurate aggregation and visualization.
        for turn_sample in samples[1:]:
            if hasattr(turn_sample, "trace"):
                delattr(turn_sample, "trace")
        if dapo_overlong_cfg is not None:
            logger.info(
                "%s DAPO overlong cfg: max_resp_len=%s buffer_len=%s expected_len=%s penalty_factor=%s",
                plan.log_tag,
                dapo_overlong_cfg["max_resp_len"],
                dapo_overlong_cfg["buffer_len"],
                dapo_overlong_cfg["expected_len"],
                dapo_overlong_cfg["penalty_factor"],
            )

        # Exploration bonuses mutate samples' reward dicts in place (no-op when
        # every EXPLORE_* / Agent57 switch is off).
        _inject_exploration_bonuses(
            samples,
            sample=sample,
            plan=plan,
            clients=clients,
            loop=loop,
            status=status,
            eval_error=eval_error,
        )

        _finalize_sample_metadata(
            samples,
            args=args,
            plan=plan,
            clients=clients,
            loop=loop,
            trajectory_uncertainty=trajectory_uncertainty,
            status=status,
            eval_details=eval_details,
            eval_error=eval_error,
        )
        _mark_non_trainable_samples(samples)

        await asyncio.to_thread(
            _save_rollout_artifacts,
            task_spec=plan.task_spec,
            run_ctx=plan.run_ctx,
            sampling_params=sampling_params,
            sample=sample,
            samples=samples,
            status=status,
            raw_score=reward,
            eval_error=eval_error,
            turn_records=loop.turn_records,
            prm_meta=sample.metadata.get("prm") if sample.metadata else None,
            prm_coef=plan.prm_coef,
            trajectory_save_interval=plan.traj_save_interval,
        )

        if session.admission_key is not None:
            _task_circuit_record_success(plan.task_key)
        return samples

    except Exception as exc:
        if _uses_remote_terminal_env(plan.task_meta):
            _task_circuit_record_failure(plan.task_key, exc)
        log_traceback = _env_bool("TERMINAL_RL_GENERATE_FAILURE_TRACEBACK", False)
        logger.error(
            "%s Generate failed (%s): %s%s",
            plan.log_tag,
            type(exc).__name__,
            exc,
            "" if log_traceback else " (set TERMINAL_RL_GENERATE_FAILURE_TRACEBACK=1 for traceback)",
            exc_info=log_traceback,
        )
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = dict(metadata)
        sample.metadata["generate_failed"] = True
        sample.metadata["generate_error_type"] = type(exc).__name__
        sample.metadata["generate_error"] = str(exc)
        sample.status = Sample.Status.FAILED
        sample.remove_sample = True
        sample.reward = {"score": 0.0}
        _sync_reward_aliases(sample.reward)

        eos = state.tokenizer.eos_token_id
        if eos is None:
            sample.tokens = []
            sample.response_length = 0
            sample.rollout_log_probs = []
            sample.loss_mask = []
        else:
            sample.tokens = [eos, eos]
            sample.response_length = 1
            sample.rollout_log_probs = [0.0]
            sample.loss_mask = [0]

        failed_turn_records = list(loop.turn_records)
        if not failed_turn_records and _env_bool(
            "TRAJECTORY_SAVE_FAILED_SHORT_ROLLOUTS", False
        ):
            failed_turn_records = [
                _failure_specimen_record(
                    plan=plan,
                    session=session,
                    clients=clients,
                    exc=exc,
                )
            ]
        if failed_turn_records:
            await asyncio.to_thread(
                _save_rollout_artifacts,
                task_spec=plan.task_spec,
                run_ctx=plan.run_ctx,
                sampling_params=sampling_params,
                sample=sample,
                samples=[sample],
                status=sample.status,
                raw_score=0.0,
                eval_error=f"{type(exc).__name__}: {exc}",
                turn_records=failed_turn_records,
                prm_meta=sample.metadata.get("prm") if sample.metadata else None,
                prm_coef=plan.prm_coef,
                trajectory_save_interval=plan.traj_save_interval,
            )
        return [sample]

    finally:
        await _close_rollout_session(plan, session, clients, loop)
