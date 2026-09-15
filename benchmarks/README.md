# Benchmark assets

`benchmarks/` is the umbrella for source-controlled benchmark assets. It is
intentionally **not** named `datasets/`: many tasks are executable Docker
environments rather than rows in a dataset, so that name would incorrectly
produce a `datasets/environments/` hierarchy.

This directory stores assets, not Python implementations. Runtime code lives in
`harborrl/environments/`, data preparation in `harborrl/data/`, and scoring
or result export in `tools/evaluation/`.

```text
benchmarks/
├── datasets/       durable inputs: raw sources, JSONL, frozen splits, manifests
└── environments/   durable executable tasks: Dockerfiles, tests, fixtures
```

`metadata.task_path` remains relative to `benchmarks/environments/`; moving the
asset root therefore does not alter task identity or stored trajectories.

Generated views do not belong here. Retry subsets, per-run symlink farms,
temporary conversions, logs, and evaluation outputs must be written below
`runs/<run_id>/` (or a temporary directory). In particular,
`benchmarks/environments/seta_env_retry/` is ignored to prevent historical retry
artifacts from being committed again.
