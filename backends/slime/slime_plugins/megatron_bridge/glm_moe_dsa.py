"""GLM-5.1 (glm_moe_dsa) bridge for the fzyzcjy Megatron-Bridge fork API.

Ported from upstream NVIDIA Megatron-Bridge
``src/megatron/bridge/models/glm_moe_dsa/glm5_bridge.py`` to the older fork
API (``register_bridge(source=<str|class>, target=...)`` with no
``provider=`` kwarg, explicit ``provider_bridge``/``mapping_registry``
implementations, no generic CONFIG_MAPPING machinery).

Differences from upstream, forced by the runtime stack:

- Registered by architecture *name* (``"GlmMoeDsaForCausalLM"``) because the
  image pins transformers 4.57, which has no GLM-5 model classes.  A bare
  placeholder class is installed as ``transformers.GlmMoeDsaForCausalLM`` so
  ``AutoBridge._causal_lm_architecture`` can resolve the architecture (the
  fork's dispatcher falls back to ``__name__`` matching).
- The provider is the fork's ``DeepSeekV3ModelProvider`` configured from the
  raw HF config fields (the 4.57 shim config exposes the config.json keys as
  plain attributes).
- The vendored Megatron-Core (0.16.0rc0) only knows the DSA config fields
  ``experimental_attention_variant``, ``dsa_indexer_{n_heads,head_dim,topk,
  loss_coeff,use_sparse_loss}``; the newer ``dsa_indexer_rope_interleaved`` /
  ``topk_freq`` / ``skip_topk_offset`` / ``rotate_activation`` /
  ``k_norm_epsilon`` knobs do not exist and are intentionally not set.
- No FP8 dequantization hook: the fork has no
  ``megatron.bridge.models.conversion.quantization_utils`` and the GLM-5.1
  checkpoint is BF16 (no ``*_scale_inv`` tensors).
- ``_should_map_hf_config_field`` does not exist in the fork's
  ``MegatronModelBridge``; it is an upstream workaround for the generic
  config-mapping path, which the fork does not have.  We read
  ``n_routed_experts`` explicitly, so the num_experts shadowing issue does
  not apply.
"""

import logging
import os
from typing import Any

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekV3ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)


class _GLMExpertAutoMapping(AutoMapping):
    """Load one expert projection shard independently on every ETP rank.

    The fork's generic ``AutoMapping`` gathers the complete HF tensor on ETP
    rank 0 and scatters it.  GLM-5.1 has nearly 20k expert tensors; one slow
    shared-storage read on rank 0 can therefore leave an already-enqueued
    NCCL scatter outstanding for the whole process-group timeout.  Every rank
    already opens the lazy HF tensor, so slicing locally removes that
    serialization without changing the resulting Megatron shard.
    """

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module):
        if self.tp_size == 1:
            return hf_weights

        # Match the generic mapping's expert-name normalization so grouped
        # expert parameters such as ``weight15`` still provide shape metadata.
        from megatron.bridge.models.conversion.utils import get_module_and_param_from_name

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        if hf_weights is None:
            raise ValueError("hf_weights should not be None for an expert projection")

        # Expert down projections are row-parallel (dim 1); grouped/legacy
        # variants can also expose a column-parallel projection, so infer the
        # split from the local parameter shape rather than the class name.
        if hf_weights.ndim == 1:
            return hf_weights.to(device=target_param.device, dtype=target_param.dtype)
        if hf_weights.ndim != 2 or target_param.ndim != 2:
            raise ValueError(
                f"Unsupported expert projection shapes: hf={tuple(hf_weights.shape)} "
                f"target={tuple(target_param.shape)}"
            )

        if hf_weights.shape[0] == target_param.shape[0] * self.tp_size:
            dim = 0
        elif hf_weights.shape[1] == target_param.shape[1] * self.tp_size:
            dim = 1
        else:
            raise ValueError(
                f"Cannot infer expert TP split: hf={tuple(hf_weights.shape)} "
                f"target={tuple(target_param.shape)} tp={self.tp_size}"
            )

        # Slice while still on CPU, then move only this rank's shard to CUDA.
        shard = torch.chunk(hf_weights, self.tp_size, dim=dim)[self.tp_rank]
        return shard.to(device=target_param.device, dtype=target_param.dtype)


class _GLMExpertGatedMLPMapping(GatedMLPMapping):
    """Gated expert load with local TP slicing instead of NCCL scatter."""

    def hf_to_megatron(self, hf_weights, megatron_module):
        if self.tp_size == 1:
            return torch.cat([hf_weights["gate"], hf_weights["up"]], dim=0)

        from megatron.bridge.models.conversion.utils import get_module_and_param_from_name

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        gate = hf_weights["gate"]
        up = hf_weights["up"]
        if gate.shape != up.shape or gate.ndim != 2 or target_param.ndim != 2:
            raise ValueError(
                f"Unsupported gated expert shapes: gate={tuple(gate.shape)} "
                f"up={tuple(up.shape)} target={tuple(target_param.shape)}"
            )
        if gate.shape[0] != target_param.shape[0] * self.tp_size // 2:
            raise ValueError(
                f"Cannot infer gated expert TP split: gate={tuple(gate.shape)} "
                f"target={tuple(target_param.shape)} tp={self.tp_size}"
            )

        gate_shard = torch.chunk(gate, self.tp_size, dim=0)[self.tp_rank]
        up_shard = torch.chunk(up, self.tp_size, dim=0)[self.tp_rank]
        shard = torch.cat([gate_shard, up_shard], dim=0)
        return shard.to(device=target_param.device, dtype=target_param.dtype)


class GlmMoeDsaForCausalLM:  # noqa: D401 - placeholder, never instantiated
    """Dispatch-key placeholder for transformers builds without GLM-5."""


def _ensure_transformers_arch_placeholder() -> None:
    """Install a bare ``GlmMoeDsaForCausalLM`` class where the bridge looks.

    transformers 4.57 has no GLM-5 model classes and the GLM-5.1 config.json
    carries no ``auto_map``, so ``AutoBridge._causal_lm_architecture`` /
    ``_validate_config`` would fail at ``getattr(transformers,
    "GlmMoeDsaForCausalLM")``.  The bridge only ever uses the class object as
    a dispatch key (matched by ``__name__``); the HF model itself is never
    instantiated (weights stream from safetensors).  A plain placeholder
    class is therefore sufficient.  Newer transformers builds that ship the
    real class are left untouched.

    ``import mbridge.core`` (pulled in by the legacy ``slime_plugins.mbridge``
    half of this package) replaces ``sys.modules["transformers"]`` with a
    fresh ``_LazyModule``, so the attribute must be installed on every module
    object the bridge code may hold a reference to: the current
    ``transformers`` module and the ``transformers`` global captured inside
    ``megatron.bridge``'s ``auto_bridge`` module.
    """
    import sys

    candidates = [sys.modules.get("transformers")]
    try:
        from megatron.bridge.models.conversion import auto_bridge as _auto_bridge

        candidates.append(getattr(_auto_bridge, "transformers", None))
    except Exception:
        pass

    installed = set()
    for mod in candidates:
        if mod is None or id(mod) in installed:
            continue
        installed.add(id(mod))
        if not hasattr(mod, "GlmMoeDsaForCausalLM"):
            mod.GlmMoeDsaForCausalLM = GlmMoeDsaForCausalLM


_ensure_transformers_arch_placeholder()


def _hf_dtype(hf_config: Any, default: torch.dtype = torch.float32) -> torch.dtype:
    """Resolve the checkpoint dtype from an HF config.

    transformers 5.x renamed ``torch_dtype`` to ``dtype``; GLM-5.1's
    config.json already uses ``dtype``.  The bare shim config stores it as a
    plain string attribute, so read both spellings.
    """
    value = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        if value in ("float16", "fp16"):
            return torch.float16
        if value in ("bfloat16", "bf16"):
            return torch.bfloat16
        if value in ("float32", "fp32", "float"):
            return torch.float32
    return default


def _hf_rope_theta(hf_config: Any, default: float = 1000000.0) -> float:
    """Read rope_theta from nested ``rope_parameters`` (5.x layout) or top-level."""
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    if rope_parameters is not None and hasattr(rope_parameters, "rope_theta"):
        return rope_parameters.rope_theta
    return getattr(hf_config, "rope_theta", default)


@MegatronModelBridge.register_bridge(source="GlmMoeDsaForCausalLM", target=GPTModel)
class GLMMoEDSABridge(MegatronModelBridge):
    """Megatron Bridge for GLM-5.1 (MoE + MLA + DSA).

    Handles conversion between HuggingFace GlmMoeDsaForCausalLM checkpoints
    and Megatron-Core GPTModel format through ``AutoBridge``::

        bridge = AutoBridge.from_hf_pretrained("<GLM-5.1 snapshot>", trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV3ModelProvider:
        hf_config = hf_pretrained.config

        params_dtype = _hf_dtype(hf_config, default=torch.bfloat16)

        # GLM-5.1 keeps the first ``first_k_dense_replace`` layers dense.
        moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )

        configs: dict[str, Any] = {
            # Base architecture
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "ffn_hidden_size": hf_config.intermediate_size,
            "num_attention_heads": hf_config.num_attention_heads,
            # mcore's MLA path derives everything from qk_head_dim /
            # qk_pos_emb_head_dim / v_head_dim; kv_channels is unused there.
            # Keep the legacy GLM-5.1 megatron-args value (full q head dim).
            "kv_channels": hf_config.qk_head_dim,
            "seq_length": hf_config.max_position_embeddings,
            "vocab_size": hf_config.vocab_size,
            "init_method_std": hf_config.initializer_range,
            "layernorm_epsilon": hf_config.rms_norm_eps,
            # Precision
            "fp16": params_dtype == torch.float16,
            "bf16": params_dtype == torch.bfloat16,
            "params_dtype": params_dtype,
            # Construct parameters on CPU while the colocated SGLang engine
            # is still resident.  The default GPU initialization needs a
            # transient allocation for every grouped expert linear and can
            # fail even after SGLang's memory-saver endpoint returns 200.
            # Weight loading subsequently moves each TP shard to CUDA.
            "use_cpu_initialization": os.getenv("GLM_CPU_INITIALIZATION", "1").lower()
            in {"1", "true", "yes", "on"},
            # The bridge load path overwrites every parameter with checkpoint
            # data immediately after construction, so random initialization
            # is wasted work.  Disabling it avoids minutes of CPU RNG work for
            # the 256-expert model and prevents the repeated RNG-context
            # warnings seen during colocated startup.
            "perform_initialization": os.getenv("GLM_PERFORM_INITIALIZATION", "0").lower()
            in {"1", "true", "yes", "on"},
            # MLA.  NOTE: mcore ``qk_head_dim`` is the *nope* width; the HF
            # ``qk_head_dim`` (256) = nope (192) + rope (64).
            "multi_latent_attention": True,
            "q_lora_rank": hf_config.q_lora_rank,
            "kv_lora_rank": hf_config.kv_lora_rank,
            "qk_head_dim": hf_config.qk_nope_head_dim,
            "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
            "v_head_dim": hf_config.v_head_dim,
            # GLM-5.1 uses default RoPE (no YaRN).
            "rope_type": "rope",
            "rotary_base": _hf_rope_theta(hf_config),
            "rotary_scaling_factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            # Vendored mcore rejects rotary_interleaved with MLA; HF's GLM-5.1
            # modeling applies split-half (NeoX-style) RoPE everywhere, which
            # matches mcore's non-interleaved convention.
            "rotary_interleaved": False,
            # DSA (Dynamic Sparse Attention) indexer.
            "experimental_attention_variant": "dsa",
            "dsa_indexer_n_heads": hf_config.index_n_heads,
            "dsa_indexer_head_dim": hf_config.index_head_dim,
            "dsa_indexer_topk": hf_config.index_topk,
            # Vendored mcore crashes on a None loss coeff (None > 0); keep the
            # upstream GLM5 values.
            "dsa_indexer_loss_coeff": 0.001,
            "dsa_indexer_use_sparse_loss": True,
            # MoE
            "num_moe_experts": hf_config.n_routed_experts,
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size
            * hf_config.n_shared_experts,
            "moe_layer_freq": moe_layer_freq,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "moe_router_num_groups": hf_config.n_group,
            "moe_router_group_topk": hf_config.topk_group,
            "moe_router_topk_scaling_factor": hf_config.routed_scaling_factor,
            "moe_router_score_function": getattr(hf_config, "scoring_func", "sigmoid"),
            "moe_router_enable_expert_bias": True,
            # RL fine-tuning must not mutate e_score_correction_bias.
            "moe_router_bias_update_rate": 0.0,
            # seq_aux_loss with a zero coefficient matches both the upstream
            # GLM5 bridge shape and the GLM-5.1 megatron args (aux coeff 0).
            "moe_router_load_balancing_type": "seq_aux_loss",
            "moe_aux_loss_coeff": 0.0,
            # GLM-5.1 has no shared-expert gate (unlike GLM-4.5).
            "moe_shared_expert_gate": False,
            # MTP is disabled: the megatron model skips the nextn layer, its
            # HF weights are simply not loaded.
            "mtp_num_layers": None,
            # Misc GLM5 provider settings from the upstream bridge.
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
            "add_bias_linear": False,
            "share_embeddings_and_output_weights": False,
            "qk_layernorm": True,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "attention_softmax_in_fp32": False,
            "make_vocab_size_divisible_by": 1280,
            "generation_config": hf_pretrained.generation_config,
        }

        # DSA layer spec.  Newer mcore exposes a dedicated
        # block-spec helper; the vendored 0.16.0rc0 instead routes
        # ``config.experimental_attention_variant`` through
        # ``get_gpt_decoder_block_spec`` (the DeepSeek provider default), so
        # no override is needed there.
        try:
            from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
                get_transformer_block_with_experimental_attention_variant_spec,
            )

            configs["transformer_layer_spec"] = get_transformer_block_with_experimental_attention_variant_spec
        except (ImportError, ModuleNotFoundError):
            pass

        provider = DeepSeekV3ModelProvider(**configs)
        return provider

    def _is_adapter_param_name(self, param_name: str) -> bool:
        # slime's Megatron LoRA marks adapter params as ``slime_lora_A`` /
        # ``slime_lora_B``; the fork's default only knows ``.adapter.``.
        # Adapter params start fresh (they are absent from the HF checkpoint)
        # and must be excluded from conversion tasks — otherwise they leave
        # None holes in the task list that crash
        # ``load_weights_hf_to_megatron`` with
        # ``AttributeError: 'NoneType' object has no attribute
        # 'megatron_module'``.
        return super()._is_adapter_param_name(param_name) or ".slime_lora_" in param_name

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
            # Attention layernorm
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Attention output
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Post-attention layernorm — MoE layers use pre_mlp_layernorm, dense layers use layer_norm_weight
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # MLA weights
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # For non-MLA attention (fallback)
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            # DSA indexer
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight": "model.layers.*.self_attn.indexer.wk.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight": "model.layers.*.self_attn.indexer.k_norm.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias": "model.layers.*.self_attn.indexer.k_norm.bias",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight": "model.layers.*.self_attn.indexer.weights_proj.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            # MoE shared experts.  NOTE: GLM-5.1 has no shared-expert gate, so
            # the upstream ``shared_experts.router.weight`` mapping is dropped.
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            # MoE expert weights (per-expert format: experts.N.down_proj)
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "model.layers.*.mlp.experts.*.down_proj.weight",
        }

        mapping_list = [
            _GLMExpertAutoMapping(megatron_param=k, hf_param=v)
            if ".mlp.experts." in k and "linear_fc2" in k
            else AutoMapping(megatron_param=k, hf_param=v)
            for k, v in param_mappings.items()
        ]

        # Attention (non-MLA fallback: combined QKV)
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Dense MLP gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        # MoE expert weights (per-expert format: experts.N.gate_proj / up_proj)
        mapping_list.extend(
            [
                _GLMExpertGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                _GLMExpertGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
            ]
        )

        # MTP mappings from the upstream GLM5 bridge.  The fork's load path
        # (``load_weights_hf_to_megatron``) never sets ``bridge.hf_config``
        # before calling ``mapping_registry()``; and our provider disables MTP
        # anyway, so these patterns are inert unless both change.
        hf_config = getattr(self, "hf_config", None)
        num_mtp_layers = (getattr(hf_config, "num_nextn_predict_layers", 0) or 0) if hf_config is not None else 0
        num_transformer_layers = getattr(hf_config, "num_hidden_layers", 0) if hf_config is not None else 0
        for mtp_layer in range(num_mtp_layers):
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.shared_head.norm.weight",
                    ),
                ]
            )

            for layer_prefix in ("transformer_layer", "mtp_model_layer"):
                for megatron_param, hf_param in param_mappings.items():
                    megatron_param = (
                        megatron_param.replace(".*", f".*.{layer_prefix}", 1)
                        .replace("decoder", "mtp")
                        .replace(".*", f".{mtp_layer}", 1)
                    )
                    hf_param = hf_param.replace("layers.*", f"layers.{mtp_layer + num_transformer_layers}")
                    mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))
                mapping_list.extend(
                    [
                        QKVMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.weight",
                            q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.weight",
                            k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.weight",
                            v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.weight",
                        ),
                        QKVMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.bias",
                            q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.bias",
                            k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.bias",
                            v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.bias",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.up_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.shared_experts.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                        ),
                        _GLMExpertGatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc1.weight*",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                        _GLMExpertGatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.local_experts.*.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                    ]
                )

        return MegatronMappingRegistry(*mapping_list)
