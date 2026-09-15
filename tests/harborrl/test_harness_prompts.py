import hashlib
import importlib.util
from pathlib import Path

import pytest


def _load_compatibility_entrypoint(module_name: str, relative_path: str):
    """Load a leaf module without importing optional harness dependencies."""
    path = Path(__file__).parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_developer_agent_prompt


get_camel_prompt = _load_compatibility_entrypoint(
    "lightrl_test_camel_prompts",
    "harborrl/harnesses/camel/prompts.py",
)
get_claude_prompt = _load_compatibility_entrypoint(
    "lightrl_test_claude_prompts",
    "harborrl/harnesses/claude_code/prompts.py",
)


@pytest.mark.parametrize(
    ("factory", "system", "non_think_mode", "expected_length", "expected_sha256"),
    [
        (
            get_camel_prompt,
            "Linux",
            False,
            5621,
            "d9ec3b2342b92ac9e3fdc59a50d86c2b80ebc67f1f0606577ab376d898073bf6",
        ),
        (
            get_camel_prompt,
            "Linux",
            True,
            5631,
            "7a37c88ac52304803b22c2cb376f6a4854b4ed6bf8cdb9ab49bb31aac1882822",
        ),
        (
            get_camel_prompt,
            "Linux (in Docker)",
            False,
            5801,
            "830453d22dbb71781313a812e5ca822fb8a8e46236b2918a724c26afd046ef05",
        ),
        (
            get_camel_prompt,
            "Linux (in Docker)",
            True,
            5811,
            "81c942260afe22be59666cd6528698c3644b0acc863c17ffd4286638415d6d23",
        ),
        (
            get_claude_prompt,
            "Linux",
            False,
            5638,
            "42a0e989b0bc4acd50ace7550908852c4fd97e65c9e448af47a4fc2bc9de0840",
        ),
        (
            get_claude_prompt,
            "Linux",
            True,
            5648,
            "e63ea461481bfb795a99a586824ec80537aa387fef43bc8626473761bbd79f74",
        ),
        (
            get_claude_prompt,
            "Linux (in Docker)",
            False,
            5819,
            "a3c0b43699193d149db75a11e0cceb05a6fd8ccc3983afbf34750220bcaa8820",
        ),
        (
            get_claude_prompt,
            "Linux (in Docker)",
            True,
            5829,
            "e7a8b69d1aec1cfaf2f3d62243ce2a48dd5b292a782fe0af7fd9cfcccc81f5bd",
        ),
    ],
)
def test_developer_prompt_is_byte_for_byte_compatible(
    factory,
    system,
    non_think_mode,
    expected_length,
    expected_sha256,
):
    prompt = factory(
        current_date="2026-07-30",
        system=system,
        machine="x86_64",
        is_workforce=False,
        non_think_mode=non_think_mode,
    )

    assert len(prompt) == expected_length
    assert hashlib.sha256(prompt.encode()).hexdigest() == expected_sha256
