# HarborRL

HarborRL is a native reinforcement-learning framework for auditable agents in
Harbor tasks. It keeps task execution, serving evidence, verifier receipts, and
training tensors behind strict contracts, then connects them to an external
Slime v0.3.2 backend through a small adapter.

[简体中文](README_zh.md)

## Status

The public v0.1 tree provides the native framework, CPU contract suite,
hello-world example, and official Slime v0.3.2 adapter. CPU contracts pass, but
the release tag is blocked until the pinned GPU one-step acceptance and Harbor
Hub compatibility matrix are recorded. See [release scope](docs/release-scope.md).

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
harborrl --help
```

Training is intentionally external: Slime, Megatron-LM, SGLang, model weights,
and Harbor workers are not Python dependencies of this package. Use
[bootstrap_backends.sh](scripts/bootstrap_backends.sh) on a GPU node or supply
an already-pinned environment.

## Quickstart

Run the offline reward contrast and adapter contract check:

```bash
bash examples/native/cpu_contract_smoke.sh
```

Inspect the complete GPU launch command without starting Ray, Docker, SGLang,
or training:

```bash
python -m harborrl.cli train \
  --config examples/native/train_qwen_native.yaml --dry-run
```

The example uses generic placeholders for workers, model paths, and gateway
names. Replace them in a private untracked copy; do not commit machine-specific
values.

## Repository map

| Path | Purpose |
| --- | --- |
| `harborrl/config/` | Strict native schema-2 loading and launch planning |
| `harborrl/gateway/` | Anthropic-compatible Messages gateway and SGLang evidence |
| `harborrl/rollout/` | Native group generation and external Harbor runner |
| `harborrl/trajectories/native.py` | Native IR v2, identity, reward, and trace validation |
| `harborrl/export/` | Backend-neutral immutable training batches |
| `harborrl/backends/slime_v032/` | Official Slime hook adapter and version gates |
| `examples/native/` | Offline task fixture and Qwen launch example |
| `scripts/` | Backend bootstrap, semantic probe, and publication audit |

## Testing

```bash
python -m pytest
python -m ruff check .
bash -n harborrl/platform/slime_train.sh scripts/*.sh
bash scripts/audit_public_tree.sh
```

CI runs the CPU suite, example, dry-run, package build, and hygiene audit. GPU
semantic probing and one-step training are separate explicit gates.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Task format](docs/task-format.md)
- [Backend integration](docs/backend-integration.md)
- [Dataset compatibility](docs/dataset-compatibility.md)
- [Release scope](docs/release-scope.md)

## Contributing, security, and license

Contributions and vulnerability reports follow [CONTRIBUTING.md](CONTRIBUTING.md)
and [SECURITY.md](SECURITY.md). HarborRL is released under the MIT license; see
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for third-party notices.
