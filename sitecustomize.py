"""Local Python startup hooks for HarborRL runs.

This file is imported automatically when the repository root is present on
``PYTHONPATH``.  It keeps GLM-5.1 smoke jobs working with the pinned
Transformers build by registering a minimal config class for
``model_type == "glm_moe_dsa"``.
"""

from __future__ import annotations

try:
    from transformers import AutoConfig, PretrainedConfig
except Exception:  # pragma: no cover - best-effort startup hook
    AutoConfig = None
    PretrainedConfig = None


if AutoConfig is not None and PretrainedConfig is not None:

    class GLMMoEDSAConfig(PretrainedConfig):
        model_type = "glm_moe_dsa"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def _register_config(model_type: str) -> None:
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            if model_type in CONFIG_MAPPING:
                return
        except Exception:
            pass
        try:
            AutoConfig.register(model_type, GLMMoEDSAConfig, exist_ok=True)
        except TypeError:
            try:
                AutoConfig.register(model_type, GLMMoEDSAConfig)
            except Exception:
                pass
        except Exception:
            pass

    _register_config("glm_moe_dsa")
    _register_config("glm51")


# GLM-5.1's tokenizer_config.json declares ``tokenizer_class:
# "TokenizersBackend"`` and a list-shaped ``extra_special_tokens``, both from
# a Transformers newer than the pinned 4.57 build.  Map the class name onto a
# PreTrainedTokenizerFast subclass that translates the list into the 4.57
# ``additional_special_tokens`` form so the normal from_pretrained flow works.
try:
    from transformers import PreTrainedTokenizerFast
    from transformers.models.auto import tokenization_auto as _tokenization_auto
except Exception:  # pragma: no cover - best-effort startup hook
    PreTrainedTokenizerFast = None
    _tokenization_auto = None

if PreTrainedTokenizerFast is not None and _tokenization_auto is not None:

    class _TokenizersBackendCompat(PreTrainedTokenizerFast):
        def __init__(self, *args, **kwargs):
            extra = kwargs.get("extra_special_tokens")
            if isinstance(extra, list):
                kwargs.pop("extra_special_tokens")
                kwargs.setdefault("additional_special_tokens", extra)
            super().__init__(*args, **kwargs)

    _orig_tokenizer_class_from_name = _tokenization_auto.tokenizer_class_from_name

    def _tokenizer_class_from_name(class_name):
        resolved = _orig_tokenizer_class_from_name(class_name)
        if resolved is not None:
            return resolved
        if class_name in ("TokenizersBackend", "TokenizersBackendFast"):
            return _TokenizersBackendCompat
        return None

    _tokenization_auto.tokenizer_class_from_name = _tokenizer_class_from_name


# megatron.bridge overlay: the training image ships megatron-core but not the
# fzyzcjy Megatron-Bridge fork pinned in legacy-requirements.txt.  When
# LIGHTRL_MEGATRON_BRIDGE_OVERLAY points at a directory containing
# ``megatron/bridge/``, append that directory to ``sys.path`` so the
# ``megatron`` namespace package picks up ``megatron.bridge`` from the
# overlay.  (Namespace ``_NamespacePath`` recomputes itself from ``sys.path``
# and silently discards direct ``__path__`` appends, so extending ``sys.path``
# is the reliable mechanism.)
try:
    import os as _os
    import sys as _sys

    _mbridge_overlay = _os.environ.get("LIGHTRL_MEGATRON_BRIDGE_OVERLAY")
    if (
        _mbridge_overlay
        and _os.path.isdir(_os.path.join(_mbridge_overlay, "megatron", "bridge"))
        and _mbridge_overlay not in _sys.path
    ):
        _sys.path.append(_mbridge_overlay)
except Exception:  # pragma: no cover - best-effort startup hook
    pass
