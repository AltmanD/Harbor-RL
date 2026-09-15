from types import SimpleNamespace

from slime.world_model.target_provider import frozen_wm_target_provider


def _sample(obs_text=None):
    record = {"next_observation_text": obs_text} if obs_text is not None else {}
    return SimpleNamespace(metadata={"world_model": record}, train_metadata=None)


def test_provider_hash_mode_attaches_targets(monkeypatch):
    monkeypatch.setenv("LWM_TARGET_PROVIDER_ENCODER", "hash")
    samples = [_sample("total 0\ndrwxr-xr-x"), _sample(None), _sample("exit code 1")]
    args = SimpleNamespace(world_model_latent_dim=32)

    result = frozen_wm_target_provider(args=args, samples=samples, turn_records=[], task_meta={}, run_ctx=None)

    assert result is None
    first, missing, third = [s.metadata["world_model"] for s in samples]
    assert len(first["target_latents"]) == 32 and first["target_mask"] == 1.0
    assert missing["target_latents"] is None and missing["target_mask"] == 0.0
    assert len(third["target_latents"]) == 32 and third["target_mask"] == 1.0
    for sample in samples:
        assert sample.train_metadata["world_model"] is sample.metadata["world_model"]


def test_provider_hash_mode_is_deterministic(monkeypatch):
    monkeypatch.setenv("LWM_TARGET_PROVIDER_ENCODER", "hash")
    args = SimpleNamespace(world_model_latent_dim=16)
    left = [_sample("some observation")]
    right = [_sample("some observation")]

    frozen_wm_target_provider(args=args, samples=left)
    frozen_wm_target_provider(args=args, samples=right)

    assert left[0].metadata["world_model"]["target_latents"] == right[0].metadata["world_model"]["target_latents"]


def test_provider_hf_mode_requires_checkpoint(monkeypatch):
    monkeypatch.delenv("LWM_TARGET_CHECKPOINT", raising=False)
    monkeypatch.setenv("LWM_TARGET_PROVIDER_ENCODER", "hf")
    sample = _sample("observation")

    try:
        frozen_wm_target_provider(args=SimpleNamespace(), samples=[sample])
    except RuntimeError as exc:
        assert "LWM_TARGET_CHECKPOINT" in str(exc)
    else:
        raise AssertionError("hf mode without checkpoint must raise RuntimeError")
