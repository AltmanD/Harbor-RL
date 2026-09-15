from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
AGENTIC_RL = ROOT / "harborrl"


def _launcher_text() -> str:
    """slime_train.sh plus its sourced stage libraries, in execution order."""
    platform_dir = AGENTIC_RL / "platform"
    parts = [platform_dir / "slime_train.sh"] + sorted(
        (platform_dir / "slime_train").glob("lib_*.sh")
    )
    return "\n".join(part.read_text() for part in parts)


def test_rollout_configs_default_to_camel_agent():
    for name in ("rollout_qwen3.yaml", "rollout_qwen3_think.yaml"):
        config = yaml.safe_load((ROOT / "configs" / "rollout" / name).read_text())
        assert config["harness_option"] == "camel-agent"


def test_slime_backend_routes_registered_harnesses():
    script = _launcher_text()
    assert 'HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"' in script
    assert 'cfg["harness_option"] = harness_option' in script
    assert "claude-code|claude_code)" in script
    assert "mcp__terminal_rl__read_file" in script
    assert "mcp__terminal_rl__write_file" in script
    assert "mcp__terminal_rl__list_dir" in script
    assert '\\"HARNESS_OPTION\\": \\"${HARNESS_OPTION}\\"' in script
    assert 'RUNTIME_PYTHONPATH="${MEGATRON_DIR}:${REPO_ROOT}:${SLIME_DIR}"' in script
    assert ':${SCRIPT_DIR}"' not in script


def test_recipe_selects_harness_without_config_composition():
    script = (ROOT / "examples" / "training" / "train_qwen3_8b_seta_dapo.sh").read_text()
    assert 'HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"' in script
    assert "harborrl.platform.cli" not in script
    assert "harborrl/platform/slime_train.sh" in script


def test_dive_po_runtime_accepts_claude_code_alias():
    script = (
        AGENTIC_RL / "algorithms" / "dive_po" / "defaults.sh"
    ).read_text()
    assert "claude-code|claude_code)" in script
    assert "Use: camel-agent|claude_code" in script


def test_camel_harness_factory_imports_complete_package():
    from harborrl.harnesses.factory import create_harness, normalize_harness_name
    from harborrl.harnesses.camel.prompts import get_developer_agent_prompt

    assert normalize_harness_name("camel-agent") == "camel_agent"
    assert callable(create_harness)
    assert callable(get_developer_agent_prompt)
