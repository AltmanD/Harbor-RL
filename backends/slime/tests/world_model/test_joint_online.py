from types import SimpleNamespace

import torch

from slime.world_model import joint_online, target_provider
from slime.world_model.hidden_encoder import hash_hidden_batch
from slime.world_model.modules import TextLatentWorldModel, TextLatentWorldModelConfig


class _StubEncoder:
    """GPU/model-free stand-in for PolicyHiddenEncoder (hash-grade hiddens)."""

    hidden_size = 16

    def __call__(self, transitions):
        return hash_hidden_batch(transitions, self.hidden_size)

    def _target_ids(self, text):
        return [1]

    def _forward_hidden(self, rows, *, require_grad):
        torch.manual_seed(0)
        hidden = torch.randn(len(rows), 3, self.hidden_size)
        mask = torch.ones(len(rows), 3, dtype=torch.long)
        return hidden, mask


def _tiny_model():
    config = TextLatentWorldModelConfig(
        state_hidden_dim=16,
        action_hidden_dim=16,
        target_hidden_dim=16,
        latent_dim=8,
        predictor_type="mlp",
        value_head=False,
        uncertainty_head=False,
    )
    return TextLatentWorldModel(config)


def _sample(index, obs="total 0\n-rw-r--r--"):
    record = {
        "uid": f"u{index}",
        "turn_idx": 0,
        "action_text": "ls -la",
        "next_observation_text": obs,
        "context_messages": [{"role": "user", "content": "list files"}],
        "done": False,
    }
    return SimpleNamespace(metadata={"world_model": record}, train_metadata=None)


def _stub_hf_stack(monkeypatch):
    stack = (_StubEncoder(), _tiny_model(), torch.device("cpu"))
    monkeypatch.setattr(target_provider, "_load_hf_stack", lambda args: stack)
    return stack


def test_raw_mode_returns_normalized_encoder_hidden(monkeypatch):
    monkeypatch.setenv("LWM_TARGET_PROVIDER_ENCODER", "raw")
    monkeypatch.setattr(
        target_provider,
        "_load_raw_encoder",
        lambda: (_StubEncoder(), torch.device("cpu")),
    )
    samples = [_sample(0), _sample(1, obs=None)]
    target_provider._state.clear()

    target_provider.frozen_wm_target_provider(args=SimpleNamespace(), samples=samples)

    target = samples[0].metadata["world_model"]["target_latents"]
    assert len(target) == 16
    assert abs(torch.tensor(target).norm().item() - 1.0) < 1e-4
    assert samples[1].metadata["world_model"]["target_mask"] == 0.0


def test_joint_mode_trains_and_serves_ema_targets(monkeypatch, tmp_path):
    monkeypatch.setenv("LWM_ONLINE_JOINT", "1")
    monkeypatch.setenv("LWM_TARGET_CHECKPOINT", "dummy.pt")
    monkeypatch.setenv("LWM_JOINT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LWM_JOINT_EMA_DECAY", "0.5")
    _, model, _ = _stub_hf_stack(monkeypatch)
    joint_online._state.clear()
    target_provider._state.clear()

    samples = [_sample(0), _sample(1)]
    frozen_wm = SimpleNamespace(world_model_latent_dim=8)
    out = joint_online.joint_update_and_targets(args=frozen_wm, samples=samples)

    assert set(out) == {0, 1}
    assert len(out[0]) == 8
    trainer = joint_online._state["joint_trainer"]
    assert trainer.total_steps > 0
    assert len(trainer.buffer) == 2
    assert (tmp_path / "latent_world_model_joint.pt").is_file()
    assert (tmp_path / "replay_buffer_joint.pt").is_file()
    # EMA starts from the warm-start weights and must lag behind the trained ones.
    ema_weight = trainer.ema_shared_projector.net[1].weight
    live_weight = trainer.model.shared_projector.net[1].weight
    assert not torch.allclose(ema_weight, live_weight)

    steps_after_first = trainer.total_steps
    out_again = joint_online.joint_update_and_targets(args=frozen_wm, samples=samples)
    assert trainer.total_steps == steps_after_first  # dedup: no retraining on seen ids
    assert set(out_again) == {0, 1}


def test_provider_routes_to_joint_and_writes_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("LWM_ONLINE_JOINT", "1")
    monkeypatch.setenv("LWM_TARGET_CHECKPOINT", "dummy.pt")
    monkeypatch.setenv("LWM_JOINT_OUTPUT_DIR", str(tmp_path))
    _stub_hf_stack(monkeypatch)
    joint_online._state.clear()
    target_provider._state.clear()

    samples = [_sample(0), _sample(1, obs=None)]
    target_provider.frozen_wm_target_provider(
        args=SimpleNamespace(world_model_latent_dim=8), samples=samples
    )

    assert len(samples[0].metadata["world_model"]["target_latents"]) == 8
    assert samples[0].metadata["world_model"]["target_mask"] == 1.0
    assert samples[1].metadata["world_model"]["target_mask"] == 0.0
