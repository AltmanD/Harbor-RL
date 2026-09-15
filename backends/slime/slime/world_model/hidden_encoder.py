from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .seta_dataset import TerminalTransition


def _stable_hash_hidden(texts: Sequence[str], hidden_dim: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for text in texts:
        seed_bytes = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(seed_bytes, "little") & ((1 << 63) - 1))
        rows.append(F.normalize(torch.randn(hidden_dim, generator=generator), dim=0))
    return torch.stack(rows) if rows else torch.empty(0, hidden_dim)


def hash_hidden_batch(transitions: Sequence[TerminalTransition], hidden_dim: int) -> dict[str, torch.Tensor]:
    """Deterministic hash-based pseudo-hiddens for model-free smoke tests.

    Each text field seeds a private RNG from its hash, so results are
    bit-stable across processes and machines without loading a policy model.
    The vectors carry NO semantic content; they exist only to exercise the
    training/eval plumbing end to end and must not be used to draw model
    quality conclusions.

    Args:
        transitions: Transitions to encode.
        hidden_dim: Width of every returned hidden tensor.

    Returns:
        Dict with ``state_hidden``, ``action_hidden``, ``target_hidden``, and
        ``next_state_hidden`` of shape ``(B, hidden_dim)``, plus the boolean
        ``has_next`` mask; keys match ``PolicyHiddenEncoder.forward``.
    """

    state_text = [json.dumps(row.context_messages, ensure_ascii=False, sort_keys=True) for row in transitions]
    action_text = [row.action_text for row in transitions]
    feedback_text = [row.feedback_text for row in transitions]
    next_text = [
        json.dumps(row.next_context_messages or row.context_messages, ensure_ascii=False, sort_keys=True)
        for row in transitions
    ]
    return {
        "state_hidden": _stable_hash_hidden(state_text, hidden_dim),
        "action_hidden": _stable_hash_hidden(action_text, hidden_dim),
        "target_hidden": _stable_hash_hidden(feedback_text, hidden_dim),
        "next_state_hidden": _stable_hash_hidden(next_text, hidden_dim),
        "has_next": torch.tensor([row.has_next for row in transitions], dtype=torch.bool),
    }


def _longest_common_prefix(first: list[int], second: list[int]) -> int:
    limit = min(len(first), len(second))
    index = 0
    while index < limit and first[index] == second[index]:
        index += 1
    return index


class PolicyHiddenEncoder(nn.Module):
    """Extract state/action/feedback hidden representations from a policy LLM.

    ``state_hidden`` and ``action_hidden`` come from one causal forward over
    ``h_t + a_t``: state uses the prompt-end position and action pools only the
    action span.  Feedback and next-state targets are evaluated on detached
    target branches.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        hidden_layer: int = -1,
        action_pool: str = "mean",
        max_context_tokens: int = 1536,
        max_action_tokens: int = 512,
        max_feedback_tokens: int = 512,
        backprop_to_llm: bool = False,
    ) -> None:
        super().__init__()
        if action_pool not in {"mean", "last"}:
            raise ValueError(f"Unsupported action_pool={action_pool!r}; expected 'mean' or 'last'")
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_layer = int(hidden_layer)
        self.action_pool = action_pool
        self.max_context_tokens = int(max_context_tokens)
        self.max_action_tokens = int(max_action_tokens)
        self.max_feedback_tokens = int(max_feedback_tokens)
        self.backprop_to_llm = bool(backprop_to_llm)
        self.model.eval()
        self.model.requires_grad_(self.backprop_to_llm)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> "PolicyHiddenEncoder":
        """Load a HuggingFace policy model/tokenizer pair into an encoder.

        Args:
            model_name_or_path: Hub id or local checkpoint directory.
            device: Target device; ``"auto"`` picks CUDA when available.
            dtype: Torch dtype name, or ``"auto"`` to keep checkpoint dtype.
            local_files_only: Skip network access when resolving the model.
            **kwargs: Forwarded to ``PolicyHiddenEncoder.__init__``.

        Returns:
            An encoder in eval mode wrapping the loaded model.
        """

        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        torch_dtype: Any = dtype
        if dtype != "auto":
            torch_dtype = getattr(torch, dtype)
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(torch.device(device))
        return cls(model, tokenizer, **kwargs)

    @property
    def device(self) -> torch.device:
        """Device hosting the wrapped policy model's parameters."""

        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        """Hidden width of the wrapped model.

        Raises:
            AttributeError: If the config exposes neither ``hidden_size`` nor
                ``d_model``.
        """

        value = getattr(self.model.config, "hidden_size", None)
        if value is None:
            value = getattr(self.model.config, "d_model", None)
        if value is None:
            raise AttributeError("Cannot infer hidden size from model config")
        return int(value)

    def _chat_ids(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        max_length: int | None = None,
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": False,
        }
        if max_length is not None:
            kwargs.update({"truncation": True, "max_length": int(max_length)})
        try:
            original_truncation_side = getattr(self.tokenizer, "truncation_side", None)
            if original_truncation_side is not None and max_length is not None:
                self.tokenizer.truncation_side = "left"
            try:
                ids = self.tokenizer.apply_chat_template(messages, **kwargs)
            finally:
                if original_truncation_side is not None and max_length is not None:
                    self.tokenizer.truncation_side = original_truncation_side
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return list(ids)
        except TypeError:
            # Small test/fallback tokenizers may not expose HF truncation
            # kwargs; fall back to the legacy call path in that case.
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
            )
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return list(ids)
        except Exception:
            text = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
            try:
                return list(
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=True,
                        truncation=max_length is not None,
                        max_length=max_length,
                    )
                )
            except TypeError:
                return list(self.tokenizer.encode(text, add_special_tokens=True))

    def _current_ids(self, transition: TerminalTransition) -> tuple[list[int], int, list[int]]:
        prompt = self._chat_ids(
            transition.context_messages,
            add_generation_prompt=True,
            max_length=self.max_context_tokens + self.max_action_tokens + 64,
        )
        full_messages = list(transition.context_messages) + [
            {"role": "assistant", "content": transition.action_text}
        ]
        full = self._chat_ids(
            full_messages,
            add_generation_prompt=False,
            max_length=self.max_context_tokens + self.max_action_tokens + 64,
        )
        prefix = _longest_common_prefix(prompt, full)
        if prefix >= max(1, len(prompt) // 2):
            prompt = full[:prefix]
            action = full[prefix:]
        else:
            action = list(self.tokenizer.encode(transition.action_text, add_special_tokens=False))
        if not action:
            eos = self.tokenizer.eos_token_id
            action = [int(eos if eos is not None else 0)]
        prompt = prompt[-self.max_context_tokens :]
        action = action[-self.max_action_tokens :]
        if not prompt:
            bos = self.tokenizer.bos_token_id
            prompt = [int(bos if bos is not None else 0)]
        combined = prompt + action
        return combined, len(prompt) - 1, list(range(len(prompt), len(combined)))

    def _target_ids(self, text: str) -> list[int]:
        prefix = "<environment_observation>\n"
        # Terminal observations can contain megabytes of logs.  Bound the
        # character payload before tokenization (then enforce the exact token
        # limit as well) so HuggingFace does not materialize 100k-token rows
        # merely to discard all but the final feedback window.
        text = str(text)
        char_limit = max(1024, self.max_feedback_tokens * 12)
        if len(text) > char_limit:
            text = text[-char_limit:]
        try:
            original_truncation_side = getattr(self.tokenizer, "truncation_side", None)
            if original_truncation_side is not None:
                self.tokenizer.truncation_side = "left"
            try:
                ids = list(
                    self.tokenizer.encode(
                        prefix + text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=self.max_feedback_tokens,
                    )
                )
            finally:
                if original_truncation_side is not None:
                    self.tokenizer.truncation_side = original_truncation_side
        except TypeError:
            # Small test/fallback tokenizers may not expose HF truncation
            # kwargs; the character cap still bounds their work.
            ids = list(self.tokenizer.encode(prefix + text, add_special_tokens=True))
        if not ids:
            eos = self.tokenizer.eos_token_id
            ids = [int(eos if eos is not None else 0)]
        return ids[-self.max_feedback_tokens :]

    def _next_ids(self, transition: TerminalTransition) -> list[int]:
        messages = transition.next_context_messages or transition.context_messages
        ids = self._chat_ids(messages, add_generation_prompt=True, max_length=self.max_context_tokens + 64)
        if not ids:
            bos = self.tokenizer.bos_token_id
            ids = [int(bos if bos is not None else 0)]
        return ids[-self.max_context_tokens :]

    def _pad(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(row) for row in rows)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            pad = 0
        input_ids = torch.full((len(rows), max_len), int(pad), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=self.device)
        for index, row in enumerate(rows):
            length = len(row)
            input_ids[index, :length] = torch.tensor(row, dtype=torch.long, device=self.device)
            attention_mask[index, :length] = 1
        return input_ids, attention_mask

    def _forward_hidden(
        self,
        rows: list[list[int]],
        *,
        require_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._pad(rows)
        context = nullcontext() if require_grad else torch.no_grad()
        with context:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("Policy model did not return hidden_states")
            hidden = hidden_states[self.hidden_layer].float()
        return hidden, attention_mask

    def forward(self, transitions: Sequence[TerminalTransition]) -> dict[str, torch.Tensor]:
        """Encode a batch of transitions into state/action/target hiddens.

        State and action come from one causal pass over ``h_t + a_t``:
        ``state_hidden`` is read at the prompt-end position, so the causal
        mask guarantees it never sees action tokens, while ``action_hidden``
        pools only the action span.  Feedback and next-state targets are
        computed on detached no-grad branches.

        Args:
            transitions: Non-empty batch of transitions.

        Returns:
            Dict with ``state_hidden``, ``action_hidden``, ``target_hidden``,
            and ``next_state_hidden`` of shape ``(B, hidden_size)``, plus the
            boolean ``has_next`` mask.  Target/next rows are always detached;
            state/action rows are detached unless ``backprop_to_llm``.

        Raises:
            ValueError: If ``transitions`` is empty.
            RuntimeError: If the policy model returns no hidden states.
        """

        if not transitions:
            raise ValueError("PolicyHiddenEncoder requires at least one transition")
        current_rows: list[list[int]] = []
        state_positions: list[int] = []
        action_positions: list[list[int]] = []
        for transition in transitions:
            row, state_position, action_position = self._current_ids(transition)
            current_rows.append(row)
            state_positions.append(state_position)
            action_positions.append(action_position)

        current_hidden, _ = self._forward_hidden(current_rows, require_grad=self.backprop_to_llm)
        state_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        for index, (state_position, action_position) in enumerate(zip(state_positions, action_positions)):
            state_rows.append(current_hidden[index, state_position])
            action_span = current_hidden[index, action_position]
            action_rows.append(action_span[-1] if self.action_pool == "last" else action_span.mean(dim=0))

        target_rows = [self._target_ids(transition.feedback_text) for transition in transitions]
        target_hidden, target_mask = self._forward_hidden(target_rows, require_grad=False)
        target_pooled = (target_hidden * target_mask.unsqueeze(-1)).sum(dim=1) / target_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)

        next_rows = [self._next_ids(transition) for transition in transitions]
        next_hidden, next_mask = self._forward_hidden(next_rows, require_grad=False)
        next_lengths = next_mask.sum(dim=1).clamp_min(1) - 1
        next_pooled = next_hidden[
            torch.arange(next_hidden.size(0), device=next_hidden.device),
            next_lengths,
        ]
        result = {
            "state_hidden": torch.stack(state_rows),
            "action_hidden": torch.stack(action_rows),
            "target_hidden": target_pooled.detach(),
            "next_state_hidden": next_pooled.detach(),
            "has_next": torch.tensor([row.has_next for row in transitions], dtype=torch.bool, device=self.device),
        }
        if not self.backprop_to_llm:
            result["state_hidden"] = result["state_hidden"].detach()
            result["action_hidden"] = result["action_hidden"].detach()
        return result
