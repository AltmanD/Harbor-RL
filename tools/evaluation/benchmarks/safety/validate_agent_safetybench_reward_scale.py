#!/usr/bin/env python3
from __future__ import annotations

import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTIC_RL_DIR = REPO_ROOT / "harborrl"
for path in (REPO_ROOT, REPO_ROOT / "backends" / "slime"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _install_import_stubs() -> None:
    openai = types.ModuleType("openai")
    openai_types = types.ModuleType("openai.types")
    openai_types_chat = types.ModuleType("openai.types.chat")
    chat_completion = types.ModuleType("openai.types.chat.chat_completion")
    chat_completion.ChatCompletion = object
    message_param = types.ModuleType(
        "openai.types.chat.chat_completion_message_param"
    )
    message_param.ChatCompletionMessageParam = dict
    sys.modules.setdefault("openai", openai)
    sys.modules.setdefault("openai.types", openai_types)
    sys.modules.setdefault("openai.types.chat", openai_types_chat)
    sys.modules.setdefault("openai.types.chat.chat_completion", chat_completion)
    sys.modules.setdefault(
        "openai.types.chat.chat_completion_message_param", message_param
    )

    slime = types.ModuleType("slime")
    slime_rollout = types.ModuleType("slime.rollout")
    sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")
    sglang_rollout.GenerateState = object
    slime_utils = types.ModuleType("slime.utils")
    slime_utils_types = types.ModuleType("slime.utils.types")

    class Sample:
        class Status:
            COMPLETED = "completed"
            FAILED = "failed"
            ABORTED = "aborted"
            TRUNCATED = "truncated"

        def __init__(self, prompt=None, metadata=None):
            self.prompt = prompt
            self.metadata = metadata or {}
            self.reward = None
            self.status = None
            self.remove_sample = False

    slime_utils_types.Sample = Sample
    sys.modules.setdefault("slime", slime)
    sys.modules.setdefault("slime.rollout", slime_rollout)
    sys.modules.setdefault("slime.rollout.sglang_rollout", sglang_rollout)
    sys.modules.setdefault("slime.utils", slime_utils)
    sys.modules.setdefault("slime.utils.types", slime_utils_types)

    agent = types.ModuleType("agent")
    prm_agent = types.ModuleType("harborrl.harnesses.prm.agent")
    prm_agent.TerminalPRMAgent = object
    sys.modules.setdefault("agent", agent)
    sys.modules.setdefault("harborrl.harnesses.prm.agent", prm_agent)

    for name in (
        "harborrl.rollout.backends.sglang",
        "harborrl.rollout.runner",
        "harborrl.environments.client",
    ):
        module = types.ModuleType(name)
        if name == "harborrl.rollout.backends.sglang":
            module.SGLangTurnClient = object
        elif name == "harborrl.rollout.runner":
            module.create_agent_runner = lambda **_kwargs: None
            module.normalize_harness_option = lambda value: value
        elif name == "harborrl.environments.client":
            module.TerminalEnvClient = object
        sys.modules.setdefault(name, module)

    safety_reward = types.ModuleType("harborrl.misc.reward_safety")
    safety_reward.DEFAULT_ZERO_THRESHOLD = 0.0
    safety_reward.broadcast_to_turns = lambda *_args, **_kwargs: {}
    safety_reward.per_turn_score = lambda *_args, **_kwargs: 0.0
    safety_reward.trajectory_score = lambda *_args, **_kwargs: 0.0
    sys.modules.setdefault("harborrl.misc.reward_safety", safety_reward)


_install_import_stubs()

from harborrl.types import Interaction  # noqa: E402
from harborrl.rollout.entrypoint import _build_samples  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


def main() -> None:
    base = Sample(
        prompt=[],
        metadata={"data_source": "agent_safetybench"},
    )
    interaction = Interaction(
        turn_idx=0,
        input_ids=[1, 2],
        output_token_ids=[3],
        output_token_logprobs=[0.0],
        output_text="No.",
        finish_reason="stop",
    )
    samples = _build_samples(
        interactions=[interaction],
        base_sample=base,
        outcome=1.0,
        status=Sample.Status.COMPLETED,
        outcome_is_score=True,
        penalize_short_response=False,
    )
    score = samples[0].reward["score"]
    if score != 1.0:
        raise SystemExit(f"expected direct ASB score 1.0, got {score}")
    print({"score": score, "base_score": samples[0].reward["base_score"]})


if __name__ == "__main__":
    main()
