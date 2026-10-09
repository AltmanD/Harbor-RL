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
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg" alt="PyTorch 2.4+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
</p>

<p align="center">
  English · <a href="README_zh.md">简体中文</a>
</p>

---

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

After [installation](#install), choose one of the two configuration styles:

```bash
# Compose the four experiment dimensions.
harborrl train -model qwen3-8B -harness claude-code -task terminal-bench -alg grpo

# Load a complete, reproducible experiment definition.
harborrl train config.yaml
```

For a bootstrap-aware YAML launch, use:

```bash
bash scripts/launch_native.sh --config config.yaml --backends /path/to/harborrl-backends
```

The command style is a shorthand over `config.yaml` in the working directory (or the file selected with `--config`) and selects the four experiment dimensions. The YAML style is the complete launch definition: it also pins checkpoints, workers, serving settings, GPU topology, sampling limits, and output locations. Start from [examples/native/train_qwen_native.yaml](examples/native/train_qwen_native.yaml) and keep site-specific files outside the repository.

## Install

Use Python 3.10 or newer. Python dependencies and the `harborrl` command are declared in [pyproject.toml](pyproject.toml). The CPU development install does not require CUDA, Docker, model weights, Slime, Megatron-LM, or SGLang.

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

Expected result: argparse prints the HarborRL CLI usage and exits with status 0. The public commands are `train` and `doctor`; `train` accepts command composition or a YAML file, while `doctor` validates a YAML deployment.

Install the pinned native training stack with bash:

```bash
bash scripts/bootstrap_backends.sh /path/to/harborrl-backends
. /path/to/harborrl-backends/harborrl-backend.env
```

## Configuration and task format

Training is configured with one YAML file that has a fixed section and field set. Relative paths resolve against the YAML file. Repeated `--set section.field=value` overrides are parsed as YAML values and validated again.

HarborRL indexes Harbor tasks directly. If Harbor can run a task, HarborRL can consume it through a locked catalog entry without maintaining a separate task format.

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
