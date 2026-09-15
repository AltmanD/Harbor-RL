from slime.utils.misc import should_save_checkpoint


def test_periodic_checkpoint_schedule_is_unchanged_by_default():
    saved = [
        rollout_id
        for rollout_id in range(20)
        if should_save_checkpoint(rollout_id, interval=8, num_rollout=20)
    ]
    assert saved == [7, 15, 19]


def test_first_rollout_checkpoint_closes_startup_gap():
    saved = [
        rollout_id
        for rollout_id in range(20)
        if should_save_checkpoint(
            rollout_id,
            interval=8,
            num_rollout=20,
            save_first_rollout=True,
        )
    ]
    assert saved == [0, 7, 15, 19]


def test_first_rollout_does_not_enable_disabled_checkpointing():
    assert not should_save_checkpoint(0, interval=None, save_first_rollout=True)
