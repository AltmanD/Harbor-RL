"""Harbor ``claude-code`` agent evaluation harness adapter.

Shares the Harbor job lifecycle with :class:`Terminus2EvalHarness`; only the
``agents`` section and the launching process environment differ: the
``claude-code`` agent speaks the Anthropic protocol, so the model name carries
no ``openai/`` prefix and credentials come as ``ANTHROPIC_BASE_URL`` /
``ANTHROPIC_AUTH_TOKEN``.

NOTE: the exact kwarg/env surface of Harbor's built-in ``claude-code`` agent
varies between Harbor versions — verify ``agent_kwargs`` / ``agent_env``
against the Harbor version you actually run. Anything passed via
``spec.extra["agent_kwargs"]`` / ``spec.extra["agent_env"]`` is forwarded
verbatim.

Harness-specific ``spec.extra`` keys (in addition to the terminus-2 ones):

- ``model_prefix``: model name prefix (default ``""`` — Anthropic protocol).
- ``agent_kwargs``: forwarded as the agent's ``kwargs`` dict.
- ``agent_env``: forwarded into the agent's ``env`` (overrides defaults).
- ``anthropic_auth_token``: default auth token placeholder (default ``"dummy"``).
"""

from __future__ import annotations

from harborrl.harnesses.eval.base import EvalRunSpec
from harborrl.harnesses.eval.terminus2 import _DEFAULT_NO_PROXY, Terminus2EvalHarness


class ClaudeCodeCliEvalHarness(Terminus2EvalHarness):
    """Harbor runner with the built-in ``claude-code`` agent."""

    agent_name = "claude-code"
    default_model_prefix = ""

    @property
    def name(self) -> str:
        return "claude_code_cli"

    def build_agent(self, spec: EvalRunSpec) -> dict:
        extra = spec.extra
        api_base = spec.serving.api_base
        prefix = extra.get("model_prefix", self.default_model_prefix)
        env = {
            "ANTHROPIC_BASE_URL": api_base,
            "ANTHROPIC_AUTH_TOKEN": str(extra.get("anthropic_auth_token", "dummy")),
        }
        env.update(spec.environment)
        env.update({str(k): str(v) for k, v in dict(extra.get("agent_env", {})).items()})
        return {
            "name": self.agent_name,
            "model_name": f"{prefix}{spec.serving.model_name}",
            "kwargs": dict(extra.get("agent_kwargs", {})),
            "env": env,
        }

    def launch_command(self, spec: EvalRunSpec, config_path: str) -> tuple[list[str], dict[str, str]]:
        extra = spec.extra
        harbor_bin = str(extra.get("harbor_bin", "harbor"))
        no_proxy = str(extra.get("no_proxy", _DEFAULT_NO_PROXY))
        env = {
            "ANTHROPIC_BASE_URL": spec.serving.api_base,
            "ANTHROPIC_AUTH_TOKEN": str(extra.get("anthropic_auth_token", "dummy")),
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
        env.update({str(k): str(v) for k, v in dict(extra.get("process_env", {})).items()})
        return [harbor_bin, "run", "--config", config_path], env
