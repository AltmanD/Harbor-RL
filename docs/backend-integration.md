# Backend integration

HarborRL does not vendor or patch the training backend. The only accepted public
contract is `slime-v0.3.2-native-v1`.

## Pinned combination

| Component | Gate |
| --- | --- |
| Slime | `v0.3.2`, commit `3778dbf6d1a533ab478ecf5ddaa11449a47752b2` |
| Megatron-LM | commit `1dcf0dafa884ad52ffb243625717a3471643e087` |
| SGLang | image tag `v0.5.15.post1-cu129` |

`SLIME_DIR` must contain `train.py`; `MEGATRON_DIR` may be omitted only when the
official Slime image already bundles the pinned Megatron commit. Doctor compares
observed versions and fails on drift.

## Official hooks

| Slime option | HarborRL module |
| --- | --- |
| `--rollout-function-path` | `harborrl.backends.slime_v032.rollout.generate_rollout` |
| `--custom-convert-samples-to-train-data-path` | `...converter.convert_samples_to_train_data` |
| `--custom-advantage-function-path` | `...advantage.compute_advantages_and_returns` |
| `--custom-loss-function-path` | `...loss.loss_function` |
| `--rollout-data-postprocess-path` | `...postprocess.rollout_data_postprocess` |

The adapter exports only Slime standard training fields:
`tokens`, `response_lengths`, `rewards`, `raw_reward`, `truncated`,
`sample_indices`, `rollout_ids`, `loss_masks`, `rollout_log_probs`, and
`rollout_mask_sums`. No `native_*` tensor leaks into the actor.

## Acceptance gate

Before v0.1 is tagged, the pinned combination must complete a GPU one-step run
that demonstrates:

1. official Slime launch with zero HarborRL patches;
2. 0/1 reward contrast and nonzero group advantage;
3. gateway token/logprob evidence matching SGLang semantics;
4. nonzero actor gradient and policy loss;
5. explicit policy-weight change;
6. checkpoint save/reload and a new policy version consumed by rollout;
7. checkpoint, trace, and receipt artifacts that pass post-hoc validation.

`scripts/sglang_semantic_probe.py` compares served output IDs and raw logprobs
with a local HF forward pass and requires an explicit weight version.
