from harborrl.harnesses.identity import get_harness_descriptor
from harborrl.harnesses.factory import display_harness_name, normalize_harness_name
from harborrl.harnesses.eval import normalize_eval_harness_name


def test_aliases_share_identity_without_merging_factories():
    assert normalize_harness_name("camel-agent") == "camel_agent"
    assert normalize_eval_harness_name("camel-agent") == "camel_agent"
    assert display_harness_name("claude-code-cli") == "claude-code"

    camel = get_harness_descriptor("camel")
    terminus = get_harness_descriptor("terminus-2", capability="eval")
    assert camel.capabilities == frozenset({"train", "eval"})
    assert terminus.capabilities == frozenset({"eval"})


def test_training_registry_rejects_eval_only_harness():
    try:
        normalize_harness_name("terminus2")
    except ValueError as exc:
        assert "Unsupported harness" in str(exc)
    else:
        raise AssertionError("terminus2 must not be accepted as a training harness")
