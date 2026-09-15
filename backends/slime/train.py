import logging
import time

import ray
import wandb

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.checkpoint_utils import run_checkpoint_action
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_eval, should_run_periodic_action, should_save_checkpoint
from slime.utils.rollout_skip import is_skip_train_result

logger = logging.getLogger(__name__)
_SKIPPED_ROLLOUT = object()


def _looks_like_env_storm(exc: Exception) -> bool:
    text = repr(exc)
    return any(
        marker in text
        for marker in (
            "dynamic sampling aborted after repeated all-failed rollout groups",
            "TASK_SLOTS_EXHAUSTED",
            "ALL_WORKERS_UNAVAILABLE_OR_PRESSURED",
            "/allocate",
        )
    )


def _relay_pending_metrics(result):
    """Log pending wandb metrics relayed from secondary processes."""
    if not result:
        return
    if wandb.run is None:
        return
    metrics_list = result if isinstance(result, list) else [result]
    for item in metrics_list:
        if isinstance(item, list):
            for m in item:
                if isinstance(m, dict):
                    wandb.log(m)
        elif isinstance(item, dict):
            wandb.log(item)


def _resolve_generation_result(gen_result):
    pending = None
    if gen_result is _SKIPPED_ROLLOUT:
        return _SKIPPED_ROLLOUT, pending
    if isinstance(gen_result, tuple):
        rollout_data_ref, pending = gen_result
    else:
        rollout_data_ref = gen_result
    if is_skip_train_result(rollout_data_ref):
        return _SKIPPED_ROLLOUT, pending
    return rollout_data_ref, pending


def _get_rollout_generation_result(args, rollout_manager, rollout_id):
    max_retries = int(getattr(args, "rollout_generation_max_retries", 0) or 0)
    initial_backoff = max(0.0, float(getattr(args, "rollout_generation_retry_initial_backoff", 30.0) or 0.0))
    max_backoff = max(0.0, float(getattr(args, "rollout_generation_retry_max_backoff", 300.0) or 0.0))
    multiplier = max(1.0, float(getattr(args, "rollout_generation_retry_backoff_multiplier", 2.0) or 1.0))
    env_storm_max_retries = int(getattr(args, "rollout_generation_env_storm_max_retries", 3) or 0)
    attempt = 0
    env_storm_attempt = 0

    while True:
        try:
            return ray.get(rollout_manager.generate.remote(rollout_id))
        except Exception as exc:
            attempt += 1
            if _looks_like_env_storm(exc):
                env_storm_attempt += 1
            else:
                env_storm_attempt = 0
            if env_storm_max_retries >= 0 and env_storm_attempt > env_storm_max_retries:
                logger.error(
                    "Rollout generation failed after repeated environment-allocation storms: "
                    "rollout_id=%s attempts=%s env_storm_attempts=%s env_storm_max_retries=%s error=%r",
                    rollout_id,
                    attempt,
                    env_storm_attempt,
                    env_storm_max_retries,
                    exc,
                )
                raise
            if max_retries >= 0 and attempt > max_retries:
                if getattr(args, "rollout_generation_skip_on_failure", False):
                    logger.error(
                        "Rollout generation failed permanently; skipping rollout: "
                        "rollout_id=%s attempts=%s max_retries=%s error=%r",
                        rollout_id,
                        attempt,
                        max_retries,
                        exc,
                    )
                    return _SKIPPED_ROLLOUT
                logger.error(
                    "Rollout generation failed permanently: rollout_id=%s attempts=%s max_retries=%s error=%r",
                    rollout_id,
                    attempt,
                    max_retries,
                    exc,
                )
                raise

            wait_s = initial_backoff * (multiplier ** max(0, attempt - 1))
            if max_backoff > 0:
                wait_s = min(wait_s, max_backoff)
            logger.warning(
                "Rollout generation failed; retrying same rollout after %.1fs: "
                "rollout_id=%s attempt=%s max_retries=%s error=%r",
                wait_s,
                rollout_id,
                attempt,
                "unlimited" if max_retries < 0 else max_retries,
                exc,
            )
            if wait_s > 0:
                time.sleep(wait_s)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    explicit_eval_steps = set(args.eval_steps or [])

    # special case for eval-only
    if args.num_rollout == 0 and (args.eval_interval is not None or 0 in explicit_eval_steps):
        _relay_pending_metrics(ray.get(rollout_manager.eval.remote(rollout_id=0)))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        fatal = bool(getattr(args, "checkpoint_save_fatal", False))
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            run_checkpoint_action(
                "actor",
                rollout_id,
                lambda: actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                ),
                fatal=fatal,
            )
        if args.use_critic:
            run_checkpoint_action(
                "critic",
                rollout_id,
                lambda: critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                ),
                fatal=fatal,
            )
        if args.rollout_global_dataset:
            run_checkpoint_action(
                "rollout-dataset",
                rollout_id,
                lambda: ray.get(rollout_manager.save.remote(rollout_id)),
                fatal=fatal,
            )

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if (
            rollout_id == 0
            and not args.skip_eval_before_train
            and (0 in explicit_eval_steps or (args.eval_steps is None and args.eval_interval is not None))
        ):
            _relay_pending_metrics(ray.get(rollout_manager.eval.remote(0)))

        gen_result = _get_rollout_generation_result(args, rollout_manager, rollout_id)
        rollout_data_ref, pending = _resolve_generation_result(gen_result)
        if pending:
            _relay_pending_metrics(pending)
        if rollout_data_ref is _SKIPPED_ROLLOUT:
            logger.warning("Skipping training for rollout_id=%s", rollout_id)
            continue

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        train_iters_per_rollout = max(1, int(getattr(args, "train_iters_per_rollout", 1) or 1))
        if train_iters_per_rollout > 1 and getattr(args, "loss_type", "policy_loss") != "decoupled_policy_loss":
            logger.warning(
                "train_iters_per_rollout=%s requires decoupled_policy_loss; falling back to 1",
                train_iters_per_rollout,
            )
            train_iters_per_rollout = 1

        for train_iter in range(train_iters_per_rollout):
            if train_iter == 0:
                current_rollout_data_ref = rollout_data_ref
            else:
                sample_result = ray.get(rollout_manager.sample_training_data.remote(rollout_id, train_iter))
                if sample_result is None:
                    logger.warning(
                        "Replay buffer exhausted at rollout_id=%s train_iter=%s; stopping extra train iterations",
                        rollout_id,
                        train_iter,
                    )
                    break
                if isinstance(sample_result, tuple):
                    current_rollout_data_ref, pending = sample_result
                    _relay_pending_metrics(pending)
                else:
                    current_rollout_data_ref = sample_result

            if args.use_critic:
                critic_train_handle = critic_model.async_train(rollout_id, current_rollout_data_ref)
                if rollout_id >= args.num_critic_only_steps:
                    _relay_pending_metrics(ray.get(actor_model.async_train(rollout_id, current_rollout_data_ref)))
                _relay_pending_metrics(ray.get(critic_train_handle))
            else:
                _relay_pending_metrics(ray.get(actor_model.async_train(rollout_id, current_rollout_data_ref)))

            if getattr(args, "update_policy_version_every_train_iter", False):
                ray.get(rollout_manager.on_policy_update.remote())

        if not getattr(args, "update_policy_version_every_train_iter", False):
            ray.get(rollout_manager.on_policy_update.remote())

        if should_save_checkpoint(
            rollout_id,
            args.save_interval,
            num_rollout_per_epoch,
            args.num_rollout,
            args.save_first_rollout,
        ):
            save(rollout_id)

        offload_train()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        completed_step = rollout_id + 1
        should_eval = (
            completed_step in explicit_eval_steps
            if args.eval_steps is not None
            else should_run_eval(rollout_id, args.eval_interval, args.num_rollout)
        )
        if should_eval:
            _relay_pending_metrics(ray.get(rollout_manager.eval.remote(completed_step)))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
