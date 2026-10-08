<p align="center">
  <img src="assets/HarborRL-title.jpg" alt="HarborRL" width=800>
</p>

<p align="center">
  <h1>HarborRL</h1>
</p>

<p align="center">
  <strong>A lightweight agentic reinforcement learning framework for everyone</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg" alt="PyTorch 2.4+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
</p>

<p align="center">
  English · <a href="README_zh.md">简体中文</a>
</p>

---

<!-- <p align="center">
  <a href="https://arxiv.org/abs/2608.07346"><img src="https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"/></a>
  <a href="https://github.com/datamllab/A2E"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://colab.research.google.com/github/stevewithjobs/AEP/blob/yuchenyue/notebooks/a2e_quickstart.ipynb"><img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/></a>
  <a href="https://huggingface.co/papers/2608.07346"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/></a>
</p> -->

<!-- <p align="center">
  <a href="#-updates">🎉 News</a> •
  <a href="#1-quick-start">🚀 Quick start</a> •
  <a href="#2-build-experiments">🧪 Build experiments</a> •
  <a href="#3-capture-trajectories">📡 Capture trajectories</a> •
  <a href="#4-score-results">📊 Score results</a> •
  <a href="#5-view-results">👀 View results</a>
</p> -->

<!-- reinforcement-learning framework for tasks adapted to the Harbor format. It connects model serving, isolated task execution, verifier receipts, trajectory records, and GRPO training, then updates model parameters through a Slime/Megatron actor. -->

## News

- **10/2026** 📣📣📣 HarborRL released


## About

HarborRL is a lightweight framework for training tool-using agents on verifiable, sandboxed tasks. It makes model serving, trajectory generation, tool execution, verifier rewards, and policy updates into a single reproducible pipeline.

HarborRL supports flexible combinations of **Models × Algorithms × Tasks × Harnesses (MATH)**:

- **Models:** RL training using Qwen and GLM as base models.
- **Algorithms:** GRPO and DAPO for RL training.
- **Tasks:** RL training on **200+** Harbor-format tasks in isolated workers, including Terminal-Bench, SWE-Bench, Deep-SWE, and more.
- **Harnesses:** RL training with **40+** harness frameworks, including Claude Code, Codex, LangGraph, SWE-agent, and more.

## Quick Start

After preparing a YAML from `examples/native/train_qwen_native.yaml`, start the complete native pipeline with one command:

```bash
bash scripts/launch_native.sh \
  --config /path/to/private.yaml \
  --backends /path/to/harborrl-backends
```

The launcher reuses an existing `harborrl-backend.env` or bootstraps the pinned Slime, Megatron-LM, and SGLang stack; loads those settings; and invokes `harborrl train`. Training reruns doctor, creates a timestamped run under `output.root/training/`, and starts only after configuration, tasks, workers, checkpoints, backend versions, dependencies, and GPU budget pass preflight.

Use `--doctor` for preflight only or `--dry-run` to print the resolved plan. The private YAML must point to reachable Harbor workers, task catalog and harness profile, model/reference checkpoints, gateway URL and serving digests, GPU layout, and output root. Keep credentials and site-specific values out of the repository.



<!-- ## Summary

HarborRL targets training runs with the following properties:

- Model calls, task execution, verifier results, rewards, and policy versions are recorded as one rollout history rather than reconstructed later from logs.
- Harbor tasks run in their environments on external workers, while the trainer keeps scheduling, data validation, and training responsibilities.
- Native trajectory data is validated and exported in a backend-neutral form before Slime/Megatron consumes it.

HarborRL is not a benchmark distribution, model provider, Docker orchestration system, general RL library, or replacement for Slime, Megatron-LM, or SGLang.

The framework is organized around four pluggable dimensions: dataset, harness, model, and train framework/algorithm. Their selections are declared separately in the training YAML, so an experiment can change one dimension without rewriting the rollout trajectory format or training-data export. The current defaults are a Harbor task catalog, a Claude Code harness, Qwen3-8B with SGLang serving semantics, and Slime/Megatron GRPO; this combination has passed CPU tests and a pinned GPU closed-loop run. -->

<!-- ## Framework

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
| Train framework / algorithm | Slime/Megatron GRPO hooks | `training` | -->

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

<!-- ## CPU startup

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

The example locks the bundled hello-world task, constructs reward receipts for both outcomes, builds one complete Native IR group, exports a backend-neutral batch, and validates Slime adapter postprocessing. It does not use Docker, a live model, a Harbor worker, or a GPU, so success is CPU contract evidence rather than live training evidence. -->



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



## License

HarborRL is released under the MIT license.


## Citation

If you use HarborRL for your research, please cite our technical report:
```

```

## Acknowledgement

We thank [Slime](https://github.com/THUDM/slime) and [Harbor](https://github.com/harbor-framework/harbor) for developing the foundational infrastructure for training and task execution.
