# HarborRL

<div align="center">

<img src="assets/lightrl_logo_cropped.png" alt="HarborRL Logo" width="100"/>

**A lightweight, efficient, and scalable RL post-training framework for agentic environments.**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [简体中文](README_zh.md)

</div>

## Overview

HarborRL is an RL post-training framework for training language-model agents in
interactive environments. Each experiment is composed explicitly along four
axes, making recipes easy to inspect, reproduce, and extend:

| Axis | Current options | Entry point |
| --- | --- | --- |
| Harness | Training: Camel-Agent, Claude Code CLI; evaluation also supports Terminus-2 | `harborrl/harnesses/`, `harborrl/harnesses/eval/` |
| Environment | SETA, Agent-SafetyBench, AgentHarm, Tau2; SWE-smith / SWE-Verified conversion tools | `harborrl/environments/`, `harborrl/data/` |
| Model | Recipe-defined (maintained recipes currently cover Qwen3-8B and GLM-5.1) | `configs/rollout/` |
| Algorithm | GRPO / DAPO, DIVE-PO | Slime backend and `harborrl/algorithms/` |

HarborRL bundles the Slime and Megatron-LM training backends. Terminal
environments run in isolated Docker workers and are accessed by training
processes over HTTP. A worker may run on a dedicated CPU/Docker host or on the
same host as GPU training; colocated deployments must reserve sufficient CPU,
memory, Docker-network, and port capacity.

## Contents

- [Core capabilities](#core-capabilities)
- [Execution model](#execution-model)
- [Architecture](#architecture)
- [Installation and requirements](#installation-and-requirements)
- [Quickstart](#quickstart)
- [Configuration and outputs](#configuration-and-outputs)
- [Validation status](#validation-status)
- [Development and extension](#development-and-extension)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Core capabilities

- **Recipe-driven training** — every experiment is a reviewable shell script;
  `--dry-run` exposes the resolved data, model, parallelism, and backend command.
- **Agentic environments** — the `EnvSpec` registry supports SETA,
  Agent-SafetyBench, AgentHarm, and Tau2; SWE-smith / SWE-Verified are covered
  by dataset conversion and terminal-worker tooling. Terminal tasks run in
  isolated Docker workers.
- **Explicit algorithm boundaries** — GRPO / DAPO come from the bundled Slime
  backend; HarborRL maintains DIVE-PO and the PRM reward agent under
  `harborrl/algorithms/`.
- **Low-cost extension** — environments, harnesses, and reward post-processors
  have centralized registration points instead of scattered conditionals.
- **Operational observability** — turn-level trajectories, JSONL metrics, W&B
  curves, config snapshots, and dataset manifests share a categorized run
  directory (`runs/training/`, `runs/evaluation/`, or `runs/testing/`).
- **Bounded end-to-end checks** — 4-GPU smoke recipes cover rollout, reward
  shaping, and actor updates without requiring a full training run.

## Execution model

```text
Recipe script
  → Slime launcher: data preparation, worker discovery, command assembly
  → Rollout hook: harness + inference + environment interaction
  → Reward shaping: score construction and optional DIVE-PO post-processing
  → Actor update: training through Slime / Megatron-LM
```

Terminal tasks require a Docker-capable worker exposed through `WORKER_URLS`.
The worker may run on a dedicated CPU/Docker host or on the current GPU training
host. Colocation is suitable for a resource-rich single machine, provided
Docker containers and training processes do not contend for CPU, memory, disk,
or ports. A single worker is contacted directly; multiple workers, or an
explicit `START_ENV_POOL_SERVER=1`, enable the local lease router.

## Architecture

```text
examples/training/<recipe>.sh
  → harborrl/platform/slime_train.sh          # stable public launcher
      ├─ slime_train/lib_*.sh                    # 7 phases: dirs, config, data, worker, args, launch
      └─ backends/slime/train_async.py                    # GRPO / DAPO backend
          → harborrl/rollout/entrypoint.generate
              ├─ environments/registry.py       # source, runtime, and reward policy registry
              ├─ harnesses/factory.py           # Camel-Agent / Claude Code factory
              ├─ rollout/backends/sglang.py     # shared sglang turn client
              ├─ rollout/generate_steps.py      # multi-turn loop, scoring, exploration
              └─ rollout/sample_builder.py      # reward shaping → Sample.reward["score"]
          → algorithms/dive_po/rewards/dual_stream
                                                   # optional group-normalized post-process
```

The launcher loads `lib_bootstrap`, `lib_run_dir`, `lib_rollout_cfg`,
`lib_dataset`, `lib_worker`, `lib_args`, and `lib_launch` in order. Recipes rely
only on the stable `slime_train.sh` entry point, keeping project logic separate
from third-party backend details.

### Repository layout

```text
HarborRL/
├── harborrl/
│   ├── algorithms/
│   │   ├── dive_po/         # DIVE-PO exploration, rewards, and defaults
│   │   └── prm/             # PRM (process reward) agent
│   ├── data/                # conversion, download, and training-data preparation
│   ├── environments/        # EnvSpec registry, protocols, runtimes, rewards, HTTP client
│   ├── harnesses/           # Camel-Agent / Claude Code harnesses and factory
│   ├── misc/                # rollout logs and JSONL sink
│   ├── platform/            # Slime launcher, worker/router, paths, env parsing
│   └── rollout/             # hook, turn loop, serving backends, admission, trajectory store
├── configs/rollout/         # rollout model templates; the retained composition layer
├── examples/
│   ├── training/            # maintained recipes and world_model/WIP entry points
│   └── validation/          # topology-free validation helpers
├── benchmarks/              # benchmark data and task definitions
├── deploy/workers/          # Docker/SETA worker runtime and watchdogs
├── deploy/runtime/          # Worker proxy, image-prep and dependency assets
├── deploy/ops/              # Worker diagnostics, repair and cleanup
├── deploy/archive/          # Historical compatibility-only entry points
├── local/                   # Git-ignored operator workspace (local/README.md)
│   ├── rjob/                # RJob submission and DinD evaluation
│   ├── cluster/             # worker endpoints and watcher scripts
│   ├── ops/                 # local container cleanup actions
│   └── state/               # generated logs, locks and PIDs
├── tools/                   # analysis, evaluation, and developer diagnostics
│   └── evaluation/          # reusable orchestration and benchmark entry points
├── tests/                   # pytest unit and integration tests
├── backends/slime/                   # bundled third-party rollout/training backend
├── backends/Megatron-LM/             # bundled third-party model-training backend
├── runs/                    # git-ignored configs, logs, metrics, trajectories
└── docs/                    # architecture, algorithms, config, evaluation, operations
```

## Installation and requirements

- Python ≥ 3.10.
- Real training requires a prepared runtime with CUDA, Slime, Megatron-LM, and
  model checkpoints.
- SETA and other terminal tasks require a Docker-capable worker, either on a
  dedicated CPU host or on the current GPU training host.
- Training processes must reach the worker service port (default `18081`);
  use `127.0.0.1` for colocation or a training-reachable address across hosts.
- Keep site addresses, credentials, and scheduling parameters in environment
  variables or git-ignored `local/cluster/` files.
- RJob submission and lifecycle helpers are local-only under `local/rjob/`.
- The local workspace layout and state-handling rules are documented in the
  ignored local file `local/README.md` when present; this file is not part of
  the public repository.

Install the Python package from source:

```bash
python3 -m pip install -e '.[rollout,worker,train]'
python3 -c 'import harborrl'
```

This installs the Python package and selected optional dependencies only. It
does not provision CUDA, model weights, or the cluster runtime required by
Slime and Megatron-LM.

## Quickstart

### 1. Start and configure a worker

Start the worker on the selected Docker host, which may be a dedicated CPU node
or the current GPU training node. See the
[Docker worker guide](deploy/workers/README.md) for runtime usage. Worker setup
and repair commands are in [deploy/ops](deploy/ops/README.md), while proxy and
image-preparation assets are in [deploy/runtime](deploy/runtime/README.md). On
an already prepared machine, start the default pool server from the repository root:

```bash
bash deploy/workers/run_pool_server.sh
```

Then configure and check the endpoint in the training shell:

```bash
export WORKER_URLS=http://<WORKER_HOST>:18081
curl --noproxy '*' --fail http://<WORKER_HOST>:18081/healthz
```

Use `127.0.0.1` for `<WORKER_HOST>` when worker and training run on the same
host. For remote deployment, use a reachable IP or hostname. Supply multiple
workers as a comma-separated `WORKER_URLS`, or use `WORKER_URLS_FILE`.

For the maintained Qwen3-8B + SETA fixed12 + Camel-Agent 4-GPU evaluation,
use the [one-click recipe](examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh):

```bash
# Inspect the resolved configuration and command without starting services.
bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh --dry-run

# Start after confirming cleanup of local Ray/SGLang processes.
CONFIRM_LOCAL_CLEANUP=1 \
  bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh
```

The recipe requires 4 GPUs, a reachable `WORKER_URLS`, a Qwen3-8B checkpoint,
and the project runtime dependencies (at minimum PyYAML, Ray, CUDA/sglang). It
checks the SETA worker `/healthz` endpoint before launching
`backends/slime/eval_only.py`. Site-specific RJob/DinD submitters remain under the
git-ignored `local/rjob/` directory and are intentionally not part of public
recipes.

### 2. Inspect a recipe

The maintained entry points are listed below. `examples/training/world_model/`
is WIP and is not a stable training recipe.

| Recipe | Harness | Model | Environment | Algorithm |
| --- | --- | --- | --- | --- |
| `train_qwen3_8b_seta_dapo.sh` | Camel-Agent | Qwen3-8B | SETA | DAPO |
| `train_qwen3_8b_seta_dive_po.sh` | Camel-Agent | Qwen3-8B | SETA | DIVE-PO |
| `train_qwen3_8b_mixed_dapo.sh` | Camel-Agent | Qwen3-8B | SETA + Agent-SafetyBench + AgentHarm | DAPO |
| `train_glm_5_1_seta_dapo.sh` | Camel-Agent | GLM-5.1 | SETA | DAPO |

Run `--dry-run` in the GPU training environment to inspect the resolved data,
model, parallelism, and backend command:

```bash
bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_seta_dive_po.sh --dry-run
bash examples/training/train_qwen3_8b_mixed_dapo.sh --dry-run
bash examples/training/train_glm_5_1_seta_dapo.sh --dry-run
```

### 3. Launch training

```bash
WORKER_URLS=http://<WORKER_HOST>:18081 \
NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/training/train_qwen3_8b_seta_dapo.sh
```

Override the run name with `RUN_ID`. With `BACKGROUND=1`, launcher logs are
written to `runs/training/<RUN_ID>/launcher.log`. The GLM-5.1 recipe additionally needs
valid `HF_CKPT`, `REF_LOAD`, and compatible `MODEL_ARGS_FILE` values. See
[training examples](examples/README.md) for the maintained entry points.

### 4. Run source-level checks

```bash
python3 -m compileall -q harborrl
python3 -m pytest tests/harborrl -q
WORKER_URLS=http://127.0.0.1:18081 \
  bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
```

Generic development smoke checks live under `tools/dev/`. RJob submitters that
contain site topology, credentials, or scheduler parameters live under the
git-ignored `local/rjob/` directory, keeping public recipes portable.

## Configuration and outputs

### Training configuration

Training defaults are defined in the recipe scripts. Environment-variable
parsing is centralized in `harborrl/env.py`; the `ENV_VARS` table
documents the rollout-side variables. Environment and data-source capabilities
are declared in the `EnvSpec` table in `harborrl/environments/registry.py`,
while rollout model templates live in `configs/rollout/`. Environment variables
override recipe defaults; see [configuration](docs/configuration.md) for fields,
precedence, and examples.

Keep site-specific addresses, credentials, proxies, and scheduler capacity out
of public recipes. Provide them through environment variables or git-ignored
`local/cluster/` files.

### Output layout

Each run writes to its lifecycle directory (`training`, `evaluation`, or
`testing`; debug runs use `testing/debug`):

```text
runs/<category>/<RUN_ID>/
├── config/                # resolved config snapshot and dataset manifests
├── logs/                  # train.log, metrics.jsonl, and launcher logs
├── trajectories/          # per-sample traj.json and side-channel index.jsonl
├── metrics/               # W&B and offline analysis artifacts
├── environment_outputs/   # worker/AgentRunner environment artifacts
└── meta.json              # run paths, version, and command metadata
```

`runs/latest` points to the most recent run. Runtime artifacts belong under
`runs/`, not at the repository root. See
[checkpoint and W&B storage](docs/operations/checkpoint-wandb.md) for
storage conventions.

## Validation status

The recorded bounded validation (2026-08-07, 4 GPUs, after the P0–P2 refactor)
reported:

- SETA + DAPO: 3 rollouts, 6 actor training steps, finite non-zero updates, and
  the `TRAINING_METRICS_OK` marker.
- SETA + DIVE-PO: 3 rollouts, 7 actor steps, and 4 non-zero updates; complete
  trajectory artifacts were exported with `EXAMPLE_VALIDATION_OK`.
- Mixed SETA + Agent-SafetyBench + AgentHarm with DAPO: 8 metric records,
  4 actor training steps, 4 non-zero updates, and `EXAMPLE_VALIDATION_OK`.

These are short-horizon correctness checks, not convergence or benchmark
results.

## Development and extension

Common source-level checks:

```bash
python3 -m pytest tests/ -q
python3 -m compileall -q harborrl
```

- **Environment** — register an `EnvSpec` in
  `harborrl/environments/registry.py` and implement the
  `environments/protocol.py:EnvClient` contract. Runtime selection, scoring,
  safety rewards, and trajectory aliases are centralized in the registry.
- **Harness** — add `_HARNESS_ALIASES` / `_HARNESS_TARGETS` entries in
  `harborrl/harnesses/factory.py` and implement the
  `rollout/runner.py:RolloutAgent` protocol. Lazy imports isolate optional
  dependencies.
- **Reward post-processing** — expose `post_process_rewards(args, samples)`
  and set `CUSTOM_REWARD_POST_PROCESS_PATH` to its import path.
- **Training recipe** — reuse the stable launcher in `examples/training/` and a
  model template from `configs/rollout/`; keep site paths and credentials local.

## Documentation

- [Architecture](docs/architecture.md) — boundaries, training path, router, registry
- [Configuration](docs/configuration.md) — recipes, environment variables, precedence
- [Deployment overview](deploy/README.md) — worker/runtime/ops boundaries and local RJob flow
- [DIVE-PO reward math](docs/algorithms/dive_po_dual_stream.md) — dual-stream advantages
- [Harness selection](docs/harnesses/README.md) — Camel-Agent / Claude Code integration
- [Evaluation tools](docs/evaluation/README.md) — orchestration, SETA fixed12, and exports
- [Docker worker](deploy/workers/README.md) — launch, capacity, prewarm, cleanup, recovery
- [Worker operations](deploy/ops/README.md) — diagnostics, repair, cleanup, and provisioning
- [Runtime assets](deploy/runtime/README.md) — proxy, image preparation, dependencies, systemd
- [Checkpoint and W&B storage](docs/operations/checkpoint-wandb.md)
- [Training examples](examples/README.md) — maintained recipes and validation entry points

## Acknowledgements

HarborRL bundles [Slime](https://github.com/THUDM/slime) for rollout/training
runtime and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for model
training. The agentic RL stack was originally developed in **OpenClaw-RL** and
later extracted and refactored into this framework.

## Citation

If HarborRL helps your research, please cite:

```bibtex
@misc{lightrl,
  title={HarborRL: A Lightweight, Efficient, Scalable RL Post-training Framework for Agentic Environments},
  author={Pu, Yuan and Zhang, Shaoang and Zhang, Chenhao and Li, Xueyan and Lu, Yudong and Tang, Jia and Wang, Guanchu and Niu, Yazhe},
  publisher={GitHub},
  howpublished={\url{https://github.com/opendilab/LightRL}},
  year={2026},
}
```

## License

[MIT](LICENSE)
