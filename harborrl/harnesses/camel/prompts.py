"""Compatibility entry point for the Camel harness developer prompt."""

from harborrl.harnesses._developer_prompt import build_developer_agent_prompt


def get_developer_agent_prompt(
    current_date: str,
    system: str,
    machine: str,
    is_workforce: bool,
    non_think_mode: bool = True,
):
    """Return the Camel prompt with its historical whitespace preserved."""
    return build_developer_agent_prompt(
        current_date,
        system,
        machine,
        is_workforce,
        non_think_mode,
        strip_trailing_whitespace=True,
    )


__all__ = ["get_developer_agent_prompt"]
