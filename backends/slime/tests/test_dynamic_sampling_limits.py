from argparse import Namespace

from slime.rollout.sglang_rollout import (
    _dynamic_sampling_failed_group_abort_min_groups,
    _dynamic_sampling_failed_group_abort_ratio,
    _dynamic_sampling_max_groups,
    _group_sample_stats,
    _rollout_abort_wait_timeout,
    _should_abort_for_failed_rollout_groups,
)
from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from slime.utils.types import Sample


def test_dynamic_sampling_default_max_groups_is_finite_when_filter_enabled():
    args = Namespace(
        dynamic_sampling_filter_path="slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        dynamic_sampling_max_groups=None,
        over_sampling_batch_size=8,
    )

    assert _dynamic_sampling_max_groups(args, target_data_size=8) == 512


def test_dynamic_sampling_max_groups_can_be_overridden_or_disabled():
    args = Namespace(
        dynamic_sampling_filter_path="filter.path",
        dynamic_sampling_max_groups=17,
        over_sampling_batch_size=8,
    )
    assert _dynamic_sampling_max_groups(args, target_data_size=8) == 17

    args.dynamic_sampling_max_groups = 0
    assert _dynamic_sampling_max_groups(args, target_data_size=8) is None

    args.dynamic_sampling_max_groups = None
    args.dynamic_sampling_filter_path = None
    assert _dynamic_sampling_max_groups(args, target_data_size=8) is None


def test_rollout_abort_wait_timeout_defaults_and_overrides():
    args = Namespace(rollout_abort_wait_timeout=None, dynamic_sampling_max_seconds=None)
    assert _rollout_abort_wait_timeout(args) == 300.0

    args.rollout_abort_wait_timeout = 17
    assert _rollout_abort_wait_timeout(args) == 17.0

    args.rollout_abort_wait_timeout = None
    args.dynamic_sampling_max_seconds = 23
    assert _rollout_abort_wait_timeout(args) == 23.0

    args.dynamic_sampling_max_seconds = 0
    assert _rollout_abort_wait_timeout(args) == 300.0


def test_group_sample_stats_counts_nested_removed_failed_samples():
    failed_removed = Sample(remove_sample=True, status=Sample.Status.FAILED)
    removed = Sample(remove_sample=True, status=Sample.Status.COMPLETED)
    kept = Sample(remove_sample=False, status=Sample.Status.COMPLETED)

    stats = _group_sample_stats([[failed_removed], [removed], kept])

    assert stats == {
        "samples": 3,
        "removed": 2,
        "failed": 1,
        "aborted": 0,
        "all_removed": 0,
        "all_failed": 0,
        "any_removed": 1,
        "any_failed": 1,
    }

    all_removed_stats = _group_sample_stats([[failed_removed], [removed]])
    assert all_removed_stats["all_removed"] == 1
    assert all_removed_stats["all_failed"] == 0

    all_failed_stats = _group_sample_stats([[failed_removed], Sample(status=Sample.Status.FAILED)])
    assert all_failed_stats["all_failed"] == 1


def test_dynamic_sampling_failed_group_abort_config_and_trigger():
    args = Namespace(
        dynamic_sampling_failed_group_abort_min_groups=12,
        dynamic_sampling_failed_group_abort_ratio=2.0,
    )

    assert _dynamic_sampling_failed_group_abort_min_groups(args, target_data_size=8) == 8
    assert _dynamic_sampling_failed_group_abort_ratio(args) == 1.0

    args.dynamic_sampling_failed_group_abort_min_groups = 0
    assert _dynamic_sampling_failed_group_abort_min_groups(args, target_data_size=8) is None

    assert _should_abort_for_failed_rollout_groups(
        completed_groups=8,
        kept_groups=0,
        failed_groups=8,
        min_groups=8,
        ratio=1.0,
    )
    assert not _should_abort_for_failed_rollout_groups(
        completed_groups=8,
        kept_groups=1,
        failed_groups=7,
        min_groups=8,
        ratio=1.0,
    )


def test_dynamic_sampling_filter_drops_all_non_trainable_group():
    args = Namespace(reward_key="score")
    failed_low = Sample(
        reward={"score": -1.0},
        remove_sample=True,
        status=Sample.Status.FAILED,
    )
    failed_high = Sample(
        reward={"score": 0.0},
        remove_sample=True,
        status=Sample.Status.FAILED,
    )

    out = check_reward_nonzero_std(args, [failed_low, failed_high])

    assert not out.keep
    assert out.reason == "all_non_trainable_group"


def test_dynamic_sampling_filter_uses_trainable_rewards_only():
    args = Namespace(reward_key="score")
    failed = Sample(
        reward={"score": -1.0},
        remove_sample=True,
        status=Sample.Status.FAILED,
    )
    kept_low = Sample(reward={"score": 0.0}, status=Sample.Status.COMPLETED)
    kept_high = Sample(reward={"score": 1.0}, status=Sample.Status.COMPLETED)

    out = check_reward_nonzero_std(args, [failed, kept_low, kept_high])

    assert out.keep
