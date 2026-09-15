from types import SimpleNamespace

import torch
from torch import nn

from slime.world_model.hidden_encoder import PolicyHiddenEncoder
from slime.world_model.seta_dataset import TerminalTransition


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        prefix = [self.bos_token_id] if add_special_tokens else []
        return prefix + [3 + (ord(char) % 17) for char in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and not return_dict
        ids = [self.bos_token_id]
        for message in messages:
            ids.extend(self.encode(str(message.get("content", "")), add_special_tokens=False))
        if add_generation_prompt:
            ids.append(29)
        return ids


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=4)
        self.calls = 0

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache, return_dict):
        self.calls += 1
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) * self.scale
        return SimpleNamespace(hidden_states=(hidden, hidden + 1.0))


def _transition():
    return TerminalTransition(
        trajectory_id="t",
        task_name="task",
        data_source="unit",
        turn_idx=0,
        context_messages=[{"role": "user", "content": "state"}],
        action_text="act",
        feedback_text="result",
        next_context_messages=[{"role": "user", "content": "state"}, {"role": "tool", "content": "result"}],
        done=False,
        reward=1.0,
        status="completed",
        source_path="unit",
    )


def test_policy_hidden_encoder_uses_one_forward_for_state_and_action():
    model = _Model()
    encoder = PolicyHiddenEncoder(model, _Tokenizer(), hidden_layer=-1, backprop_to_llm=False)

    output = encoder([_transition()])

    assert model.calls == 3  # current state+action, feedback target, next-state target
    assert output["state_hidden"].shape == (1, 4)
    assert output["action_hidden"].shape == (1, 4)
    assert output["target_hidden"].requires_grad is False
    assert output["has_next"].tolist() == [True]


def test_policy_hidden_encoder_backprop_flag_reaches_backbone():
    model = _Model()
    encoder = PolicyHiddenEncoder(model, _Tokenizer(), hidden_layer=-1, backprop_to_llm=True)

    output = encoder([_transition()])
    (output["state_hidden"].sum() + output["action_hidden"].sum()).backward()

    assert model.scale.grad is not None
    assert output["target_hidden"].requires_grad is False
