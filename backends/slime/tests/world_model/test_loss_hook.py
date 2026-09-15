from types import SimpleNamespace

import torch

from slime.world_model.loss_hook import apply_world_model_loss


def test_apply_world_model_loss_disabled_noop():
    logits = torch.ones(2, 3)
    base_loss = logits.sum()
    out_loss, log = apply_world_model_loss(
        args=SimpleNamespace(world_model_enable=False, world_model_loss_coef=1.0),
        batch={},
        logits=logits,
        loss=base_loss,
        reported_loss={"loss": base_loss.detach()},
    )
    assert out_loss is base_loss
    assert list(log.keys()) == ["loss"]


def test_apply_world_model_loss_with_precomputed_latents():
    logits = torch.zeros(1)
    base_loss = logits.sum()
    pred = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    target = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    out_loss, log = apply_world_model_loss(
        args=SimpleNamespace(world_model_enable=True, world_model_loss_coef=0.5, world_model_loss_hook_path=None),
        batch={"wm_pred_latents": pred, "wm_target_latents": target, "wm_metadata": [{"a": 1}, {"a": 2}]},
        logits=logits,
        loss=base_loss,
        reported_loss={"loss": base_loss.detach()},
    )
    assert torch.isclose(out_loss, torch.tensor(0.25))
    assert torch.isclose(log["wm/loss"], torch.tensor(0.5))
    assert torch.isclose(log["wm/metadata_count"], torch.tensor(2.0))


def test_policy_latent_hook_keeps_gradient_on_policy_logits():
    logits = torch.randn(1, 6, 11, requires_grad=True)
    args = SimpleNamespace(
        world_model_enable=True,
        world_model_loss_coef=0.5,
        world_model_loss_hook_path=None,
        world_model_backprop_to_llm=True,
        world_model_latent_dim=4,
        world_model_policy_projection="hash",
        qkv_format="thd",
    )
    target = torch.zeros(1, 4)
    loss, metrics = apply_world_model_loss(
        args=args,
        batch={
            "wm_target_latents": target,
            "total_lengths": [6],
            "response_lengths": [3],
            "wm_metadata": [{}],
        },
        logits=logits,
        loss=logits.sum() * 0.0,
        reported_loss={},
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0
    assert torch.isclose(metrics["wm/policy_gradient_path"], torch.tensor(1.0))


def test_policy_latent_hook_honors_target_mask():
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    args = SimpleNamespace(
        world_model_enable=True,
        world_model_loss_coef=1.0,
        world_model_loss_hook_path=None,
    )
    pred = [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])]
    target = [torch.zeros(2), torch.zeros(2)]
    loss, metrics = apply_world_model_loss(
        args=args,
        batch={
            "wm_pred_latents": pred,
            "wm_target_latents": target,
            "wm_target_mask": [1, 0],
        },
        logits=logits,
        loss=logits.sum() * 0.0,
        reported_loss={},
    )
    assert torch.isclose(loss, torch.tensor(0.5))
    assert torch.isclose(metrics["wm/target_mask_fraction"], torch.tensor(0.5))
