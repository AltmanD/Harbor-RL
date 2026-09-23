# Quickstart

## 1. Install the CPU development tree

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
harborrl --help
```

No GPU, Docker, model, or external backend is needed for this installation.

## 2. Run the native contract smoke

```bash
bash examples/native/cpu_contract_smoke.sh
```

The script:

1. inspects `examples/native/tasks/hello_world` and locks its content digest;
2. constructs immutable verifier result and reward artifacts for rewards 0 and 1;
3. assembles two Native IR v2 trajectories in one complete group;
4. exports a backend-neutral training batch;
5. expands official Slime v0.3.2 turn samples;
6. runs the adapter converter and actor-side postprocess validation.

It is an offline contract fixture. It does not claim live Harbor, SGLang, or
GPU training evidence.

## 3. Inspect a GPU launch

```bash
python -m harborrl.cli train \
  --config examples/native/train_qwen_native.yaml --dry-run
```

The output includes normalized paths, strict sampling and GPU budgets, external
worker identity, gateway settings, and the only accepted backend contract. It
does not run doctor or the final Slime command.

## 4. Prepare an external environment

On a GPU node, bootstrap the pinned open-source components:

```bash
scripts/bootstrap_backends.sh /path/to/backends
. /path/to/backends/harborrl-backend.env
```

Provision Harbor workers according to
[examples/native/worker_setup.md](../examples/native/worker_setup.md). Then copy
the YAML to an untracked private file, replace generic model and worker values,
and run `harborrl doctor --config <private.yaml>`.

## 5. Run the CPU checks

```bash
python -m pytest
python -m ruff check .
bash scripts/audit_public_tree.sh
```
