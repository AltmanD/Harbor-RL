from __future__ import annotations

import torch
from torch import nn

from slime.backends.fsdp_utils import lora_utils


class ToyAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.ones(1, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 1))
        self.modules_to_save = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})


def test_filter_trainable_state_dict_keeps_modules_to_save() -> None:
    model = ToyAdapterModel()
    full_state = {name: param.detach().clone() for name, param in model.named_parameters()}

    trainable_names = lora_utils._trainable_parameter_names(model)
    filtered = lora_utils._filter_trainable_state_dict(full_state, trainable_names)

    assert "lora_A" in filtered
    assert "lora_B" in filtered
    assert "modules_to_save.default.weight" in filtered
    assert "base.weight" not in filtered


def test_load_lora_checkpoint_participates_on_nonzero_rank(tmp_path, monkeypatch) -> None:
    model = ToyAdapterModel()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_state = {
        name: torch.full_like(param.detach(), 2.0) for name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(adapter_state, adapter_dir / "adapter_weights.pt")

    calls = []
    barriers = []

    monkeypatch.setattr(lora_utils.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(lora_utils.dist, "barrier", lambda: barriers.append(True))

    import torch.distributed.checkpoint.state_dict as state_dict_mod

    def fake_set_model_state_dict(model_arg, state_dict_arg, options):
        calls.append((model_arg, dict(state_dict_arg), options))

    monkeypatch.setattr(state_dict_mod, "set_model_state_dict", fake_set_model_state_dict)

    lora_utils.load_lora_checkpoint(model, adapter_dir)

    assert len(calls) == 1
    assert calls[0][0] is model
    assert calls[0][1] == {}
    assert calls[0][2].broadcast_from_rank0 is True
    assert barriers == [True]
