import re

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("glm_moe_dsa")
class GLMMoEDSABridge(DeepseekV3Bridge):
    """Megatron-Bridge adapter for GLM-5.1 HF checkpoints."""

    _DSA_INDEXER_MAPPING = {
        "core_attention.indexer.linear_wq_b.weight": ["model.layers.{layer_number}.self_attn.indexer.wq_b.weight"],
        "core_attention.indexer.linear_wk.weight": ["model.layers.{layer_number}.self_attn.indexer.wk.weight"],
        "core_attention.indexer.linear_weights_proj.weight": [
            "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
        ],
        "core_attention.indexer.k_norm.weight": ["model.layers.{layer_number}.self_attn.indexer.k_norm.weight"],
        "core_attention.indexer.k_norm.bias": ["model.layers.{layer_number}.self_attn.indexer.k_norm.bias"],
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"
        dsa_match = re.match(r"decoder\.layers\.(\d+)\.self_attention\.(.+)", mcore_weights_name)
        if dsa_match:
            layer_number, rest = dsa_match.groups()
            if rest in self._DSA_INDEXER_MAPPING:
                return [x.format(layer_number=layer_number) for x in self._DSA_INDEXER_MAPPING[rest]]
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _build_config(self):
        return self._build_base_config(
            use_cpu_initialization=False,
            # MLA + DSA
            multi_latent_attention=True,
            experimental_attention_variant="dsa",
            q_lora_rank=self.hf_config.q_lora_rank,
            kv_lora_rank=self.hf_config.kv_lora_rank,
            qk_head_dim=self.hf_config.qk_head_dim,
            qk_pos_emb_head_dim=self.hf_config.qk_rope_head_dim,
            v_head_dim=self.hf_config.v_head_dim,
            dsa_indexer_n_heads=self.hf_config.index_n_heads,
            dsa_indexer_head_dim=self.hf_config.index_head_dim,
            dsa_indexer_topk=self.hf_config.index_topk,
            # MoE
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_shared_expert_intermediate_size=self.hf_config.n_shared_experts
            * self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.0,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.n_routed_experts,
            moe_grouped_gemm=True,
            moe_router_score_function=self.hf_config.scoring_func,
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=True,
            moe_router_topk_scaling_factor=self.hf_config.routed_scaling_factor,
            moe_router_load_balancing_type="none",
            # GLM/DeepSeek-style details
            qk_layernorm=True,
            add_qkv_bias=False,
            add_bias_linear=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Megatron DSA currently routes through multi-latent attention, and
            # TransformerConfig rejects rotary_interleaved with MLA.
            rotary_interleaved=False,
        )
