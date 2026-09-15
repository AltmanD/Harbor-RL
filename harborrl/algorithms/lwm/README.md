# Latent World Model API

`harborrl.algorithms.lwm` is the lightweight Agentic-RL integration boundary
for the optional JEPA latent world model.  It re-exports metadata and isolated
replay helpers lazily; the implementation and offline trainer remain in
`slime/slime/world_model/`.

The package is auxiliary and default-off.  Enabling `--world-model-enable`
attaches transition metadata to rollout samples.  It does not change reward,
advantage, policy logits, or the GRPO/DAPO loss unless a caller explicitly
sets a non-zero `--world-model-loss-coef` and supplies precomputed latent
tensors to the loss hook.
