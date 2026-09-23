# Architecture

HarborRL v0.1 is a narrow native training boundary rather than a monolithic RL
stack.

```text
strict schema-2 config
        │
        ▼
native group rollout
  ├── Messages gateway → policy tokens + raw logprobs
  ├── external Harbor worker → isolated task + verifier
  └── collector → immutable reward receipt
        │
        ▼
Native IR v2 identity / turns / trace seal / evaluation
        │
        ▼
backend-neutral TrainingBatch
        │
        ▼
Slime v0.3.2 official hooks
  rollout → converter → advantage → loss → postprocess
```

## Core boundaries

- **Configuration:** `harborrl/config/native.py` accepts only schema 2. Paths are
  resolved relative to the YAML file, overrides are explicit, and the only
  backend contract is `slime-v0.3.2-native-v1`.
- **Gateway:** `harborrl/gateway/` exposes a small Anthropic-compatible Messages
  service. It records token IDs, raw model logprobs, policy version, tokenizer
  and template digests, and actual consumption.
- **Rollout:** `harborrl/rollout/native_generate.py` forms complete groups,
  spreads attempts over configured Harbor workers, retries failed attempts with
  new lineage, and drains all slots.
- **Harbor job:** `harborrl/rollout/harbor_job/` dispatches an external runner,
  seals traces, and collects terminal result plus reward artifact evidence.
- **Native IR:** `harborrl/trajectories/native.py` recomputes readiness instead
  of trusting caller flags. Synthetic evidence can be contract-ready but never
  RL-ready.
- **Export:** `harborrl/export/` turns complete IR groups into immutable
  trajectory, turn, mask, logprob, reward, and advantage spans. Each group's
  token weights sum to one.
- **Adapter:** `harborrl/backends/slime_v032/` implements only Slime's official
  hook surface and rejects version or sample drift before actor updates.

## Failure policy

Missing files, changed task digests, nonfinite values, mismatched token/logprob
lengths, stale policy versions, incomplete groups, duplicate turns, mixed
rollouts, or reward artifacts that disagree with verifier results fail closed.
The adapter does not repair or infer missing training evidence.
