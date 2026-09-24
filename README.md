# HarborRL

HarborRL is a reinforcement-learning framework for tasks adapted to the Harbor format. It connects model serving, isolated task execution, verifier receipts, trajectory records, and GRPO training, then updates model parameters through a Slime/Megatron actor.

[简体中文](README_zh.md)

## Summary

HarborRL targets training runs with the following properties:

- Model calls, task execution, verifier results, rewards, and policy versions are recorded as one rollout history rather than reconstructed later from logs.
- Harbor tasks run in their environments on external workers, while the trainer keeps scheduling, data validation, and training responsibilities.
- Native trajectory data is validated and exported in a backend-neutral form before Slime/Megatron consumes it.

HarborRL is not a benchmark distribution, model provider, Docker orchestration system, general RL library, or replacement for Slime, Megatron-LM, or SGLang.

The framework is organized around four pluggable dimensions: dataset, harness, model, and train framework/algorithm. Their selections are declared separately in the training YAML, so an experiment can change one dimension without rewriting the rollout trajectory format or training-data export. The current defaults are a Harbor task catalog, a Claude Code harness, Qwen3-8B with SGLang serving semantics, and Slime/Megatron GRPO; this combination has passed CPU tests and a pinned GPU closed-loop run.

## Framework

HarborRL separates evidence-producing rollout from actor updates. The native side generates and validates task trajectories; the adapter side feeds only the accepted training fields to Slime's official hooks; Megatron remains the actor trainer.

```text
Training configuration
    -> native group rollout and Harbor execution
    -> Native IR and backend-neutral export
    -> Slime v0.3.2 official hooks
    -> Megatron actor update and checkpoint
```

| Area | Contents |
| --- | --- |
| Native pipeline | Configuration, Messages gateway, rollout scheduling, Harbor-job runner, trajectory validation, and export |
| Training integration | Slime v0.3.2 hook adapter, version checks, launcher planning, and policy/checkpoint coordination |
| Examples | Offline CPU fixture and a generic Qwen GPU launch template |
| Tooling | Backend bootstrap, SGLang semantic probe, and publication audit |
| Tests | CPU contracts for configuration, gateway, lifecycle, export, and adapter behavior |

The pluggable dimensions are represented at a high level as follows:

| Dimension | Current selection | Primary configuration |
| --- | --- | --- |
| Dataset | Harbor-format task catalog | `tasks` |
| Harness | Claude Code profile | `harness` |
| Model | Configured actor/reference checkpoints and serving semantics | `model` and `gateway` |
| Train framework / algorithm | Slime/Megatron GRPO hooks | `training` |

## Install

Use Python 3.10 or newer. The CPU development install does not require CUDA, Docker, model weights, Slime, Megatron-LM, or SGLang.

Create and enter a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
```

Expected result: `.venv/` is created and the shell prompt shows `(.venv)`. Neither command starts a service or modifies the examples.

Install the editable package with development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Expected result: pip installs HarborRL with `pytest` and `ruff`, but does not install the heavyweight training stack. The command completes without compiling CUDA kernels.

Check the command-line entry point:

```bash
harborrl --help
```

Expected result: argparse prints the training CLI usage and exits with status 0. The public commands are `train` and `doctor`; both require `--config`.

## CPU startup

### Inspect the example launch plan

```bash
python -m harborrl.cli train --config examples/native/train_qwen_native.yaml --dry-run
```

Expected result: the command prints one JSON object with `config`, `command`, `environment`, and `training_command`. Paths in `config` are resolved to their absolute locations, GPU and sampling budgets are normalized, and `training_command` shows the external native-train dispatcher. It exits with status 0 and does not create a run directory, start Ray or Docker, load a model, or contact a worker.

The example deliberately contains generic names such as `worker-1`, `/models/qwen3-8b`, and placeholder serving digests. Seeing those values in dry-run output is expected; they indicate what must be replaced before doctor or training.

### Run the offline contract example

```bash
bash examples/native/cpu_contract_smoke.sh
```

Expected result: the script prints JSON with `"status": "passed"`, a 64-hex `task_digest`, `reward_contrast` of `[0.0, 1.0]`, opposite nonzero `advantages`, and identical rollout IDs for the two group members. It exits with status 0.

The example locks the bundled hello-world task, constructs reward receipts for both outcomes, builds one complete Native IR group, exports a backend-neutral batch, and validates Slime adapter postprocessing. It does not use Docker, a live model, a Harbor worker, or a GPU, so success is CPU contract evidence rather than live training evidence.

## GPU startup

### Check the node and bootstrap pinned backends

Confirm that the node exposes at least the configured number of GPUs:

```bash
nvidia-smi
```

Expected result: `nvidia-smi` lists the expected GPUs and their memory state. The example YAML defaults to eight GPUs, split into four actor GPUs and four rollout GPUs.

On a node with Git, Docker, and network access, install the pinned open-source stack:

```bash
scripts/bootstrap_backends.sh /path/to/backends
. /path/to/backends/harborrl-backend.env
```

Expected result: the script checks out Slime commit `3778dbf6d1a533ab478ecf5ddaa11449a47752b2`, Megatron-LM commit `1dcf0dafa884ad52ffb243625717a3471643e087`, pulls `lmsysorg/sglang:v0.5.15.post1-cu129`, and writes `harborrl-backend.env`. Sourcing that file sets `SLIME_DIR`, `MEGATRON_DIR`, and `SGLANG_IMAGE` in the current shell.

For an offline GPU node, prepare the same repositories and image on a compatible machine, transfer them through shared storage, and export the same three environment variables. Doctor fails if the paths or versions do not match.

### Prepare a private launch configuration

Copy `examples/native/train_qwen_native.yaml` to an untracked private path. Replace the generic worker destinations, model and reference checkpoint paths, gateway origin, tokenizer/template digests, GPU layout, and output location. Also make `tasks.catalog` and `harness.profile` point to the intended catalog and profile; paths are resolved relative to the private YAML.

Expected result: no tracked example file is modified, and the private YAML contains only values that are valid for your machines. Do not commit credentials, proxy settings, private task paths, or cluster-specific values.

Validate the normalized private plan before preparing workers:

```bash
python -m harborrl.cli train --config /path/to/private.yaml --dry-run
```

Expected result: the JSON contains your private values and resolved paths, `training.backend_contract` remains `slime-v0.3.2-native-v1`, and the command still exits without creating a run directory or starting backends.

### Prepare external Harbor workers

Each worker needs Linux, Docker Engine, passwordless SSH from the trainer, Python 3.12 or newer, and Harbor `0.23.0`. The worker must be able to pull or locally build every task image in the catalog and reach the advertised Messages gateway. It does not need a model-provider API key.

Install the matching HarborRL runner and Harbor runtime on each worker. A direct probe looks like:

```bash
ssh WORKER 'cd /path/to/Harbor-RL && /opt/harbor/bin/python -m harborrl.rollout.harbor_job.runner --probe'
```

Expected result: the worker prints JSON containing `harbor_version: 0.23.0`, a Python version of 3.12 or newer, available Claude option fields, and callable Harbor trial interfaces. The command exits with status 0 without creating a trial.

Run workers under unprivileged accounts. Keep API keys, Claude settings, proxy credentials, and private task data off workers; the rollout passes only the sanitized runtime values required for a trial.

### Run doctor

```bash
harborrl doctor --config /path/to/private.yaml
```

Expected result: doctor prints a JSON object containing `checks` and `scope`. Every check has `"ok": true`, the process exits with status 0, and no training run starts.

Doctor checks the private YAML, catalog digests, model files, worker SSH probes, pinned backend paths and commits, required imports, hooks, GPU budget, and SGLang image tag. It does not start the model service or generate tokens, so serving protocol behavior, token IDs, and logprobs are verified by the subsequent live semantic check or actual training rather than by doctor.

### Start training

```bash
harborrl train --config /path/to/private.yaml
```

Expected result: doctor runs again, then HarborRL creates a fresh directory under `output.root/training/` named with a timestamp and run ID. The run records its launch plan, source manifest, materialized task catalog, prompt rows, trajectories, policy history, metrics, and checkpoints according to the configured save interval. A successful run exits with status 0.

If any preflight check fails, training does not start and failed checks are printed to stderr as JSON. If the backend exits unsuccessfully, the child status is propagated and the run directory retains the evidence available at failure for inspection.

## Configuration and task format

Training is configured with one YAML file that has a fixed section and field set. Relative paths resolve against the YAML file. Repeated `--set section.field=value` overrides are parsed as YAML values and validated again.

| Section | Purpose |
| --- | --- |
| `execution` | Selects the fixed `harbor_job` backend |
| `tasks` | Selects a nonempty immutable task catalog |
| `harness` | Selects the Claude CLI, model, and timeout profile |
| `harbor` | Configures Harbor version, interpreter, and SSH workers |
| `gateway` | Configures the Messages listener, origin, serving digests, and raw-logprob audit |
| `model` | Configures actor/reference checkpoints and the Slime model preset |
| `training` | Configures rollout count, learning rate, save cadence, and backend identifier |
| `sampling` | Configures group size, batch groups, retries, context, and generation limits |
| `deployment` | Configures GPU count and actor/rollout parallelism |
| `output` | Configures the run-tree root |

A native task is a content-addressed directory:

```text
task/
├── task.toml
├── instruction.md
├── environment/
│   └── Dockerfile
└── tests/
    └── test.sh
```

The catalog identity is:

```json
{
  "id": "task",
  "revision": "source-revision",
  "path": "relative/or/absolute-task",
  "task_digest": "64-hex-sha256",
  "reward_profile": {
    "key": "reward",
    "scale": 1.0,
    "offset": 0.0,
    "raw_range": [0, 1]
  }
}
```

The verifier writes a shared-environment terminal result and `reward.json` or `reward.txt`. HarborRL resolves both through the reward profile, requires the selected values to agree, and records source bytes and SHA-256 receipts. Boolean and nonfinite rewards are rejected.

The machine-readable Hub inventory is [configs/harbor_hub/manifests.yaml](configs/harbor_hub/manifests.yaml). A dataset is marked `supported` only after its upstream revision, license, task layout, resource requirements, digests, worker-class reward contrast, and a complete native training batch have been recorded.

## Contributing, security, and license

Contributions and vulnerability reports follow [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). HarborRL is released under the MIT license; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for third-party notices.
