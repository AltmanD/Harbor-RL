# Configuration

Native training uses schema-2 YAML with an exact section and field set. Relative
paths resolve against the YAML file. `harborrl train --set section.field=value`
applies YAML-typed values before a second strict validation.

| Section | Field | Requirement |
| --- | --- | --- |
| `execution.backend` | fixed | `harbor_job` |
| `tasks.catalog` | path | nonempty JSON array of immutable task locks |
| `harness.name` | fixed | `claude_code` |
| `harness.profile` | path | exact Claude CLI/model/timeout profile |
| `harbor.python` | path | external Harbor `0.23.0` interpreter |
| `harbor.workers` | list | nonempty SSH destinations without credentials |
| `gateway.host` / `port` | network | local Messages listener |
| `gateway.advertised_url` | origin | plain HTTP(S) origin, no `/v1`, matching port |
| `gateway.tokenizer_digest` / `template_digest` | SHA-256 | locks serving semantics |
| `gateway.audited_raw_logprobs` | boolean | must be enabled for RL-ready evidence |
| `model.checkpoint` / `reference` | paths | HF actor and reference checkpoints |
| `model.args_file` | name | Slime model preset supplied by `SLIME_DIR` |
| `training.backend_contract` | fixed | `slime-v0.3.2-native-v1` |
| `training.num_rollout` / `save_interval` | positive integers | training length and checkpoint cadence |
| `training.learning_rate` | positive finite number | actor optimizer learning rate |
| `sampling.group_size` | integer ≥2 | trajectories per prompt group |
| `sampling.groups_per_batch` | positive integer | complete groups per rollout batch |
| `sampling.max_attempts` | positive integer | per-slot retry budget |
| `sampling.max_tokens` / `max_context` | positive integers | `max_tokens < max_context` |
| `deployment.layout` | enum | `split` or `colocate` |
| `deployment.num_gpus` / `actor_gpus` / `rollout_gpus` | budget | split layout must not exceed total |
| `deployment.actor_tensor_parallel_size` | divisor | divides actor GPUs |
| `deployment.rollout_gpus_per_engine` | divisor | divides rollout GPUs |
| `output.root` | path | immutable run tree root |

The catalog entry identity is exactly:

```json
{
  "id": "task",
  "revision": "source-revision",
  "path": "relative/or/absolute-task",
  "task_digest": "64-hex-sha256",
  "reward_profile": {"key": "reward", "scale": 1.0, "offset": 0.0, "raw_range": [0, 1]}
}
```

## Commands

```bash
harborrl train --config CONFIG --dry-run
harborrl doctor --config CONFIG
harborrl train --config CONFIG
```

`doctor` checks external Slime/Megatron/SGLang versions, importability of all
five adapter hooks, GPU budget, catalog digests, and whether the semantic
logprob probe remains required. It cannot replace a live GPU one-step acceptance
run.
