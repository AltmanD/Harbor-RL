# GLM-5.1 (744B total / sparse MoE) Megatron model arguments.
# The HF config is the source of truth; these values only provide the
# architecture fields required by slime's Megatron bootstrap.
MOE_ROUTED_EXPERTS=256
MOE_ACTIVE_ROUTED_EXPERTS=8
MOE_SHARED_EXPERTS=1

NHIDDEN=6144
MOE_FFN_HIDDEN=2048
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$((MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS))
FFN_HIDDEN=12288
# HF config: first_k_dense_replace=3, num_hidden_layers=78.
N_DENSE_LAYERS=3
N_MOE_LAYERS=75
NHEADS=64

MODEL_ARGS=(
    --moe-layer-freq [0]*$N_DENSE_LAYERS+[1]*$N_MOE_LAYERS
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    # HF config: routed_scaling_factor=2.5.
    --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --make-vocab-size-divisible-by 16
    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS))
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --no-position-embedding
    --normalization RMSNorm
    --qk-layernorm
    --multi-latent-attention
    --q-lora-rank 2048
    --kv-lora-rank 512
    # mcore qk_head_dim/kv_channels are the *nope* width (192); the HF
    # qk_head_dim (256) additionally includes the 64-wide rope part.
    --qk-head-dim 192
    --v-head-dim 256
    --kv-channels 192
    --qk-pos-emb-head-dim 64
    --vocab-size 154880
    --rotary-base 1000000
    --no-rope-fusion
)
