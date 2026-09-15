import re

from .deepseekv3 import convert_deepseekv3_to_hf


def convert_glm_moe_dsa_to_hf(args, name, param):
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.core_attention.indexer.linear_wq_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", param)]
        if rest == "self_attention.core_attention.indexer.linear_wk.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", param)]
        if rest == "self_attention.core_attention.indexer.linear_weights_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]
        if rest == "self_attention.core_attention.indexer.k_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", param)]
        if rest == "self_attention.core_attention.indexer.k_norm.bias":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", param)]
    return convert_deepseekv3_to_hf(args, name, param)
