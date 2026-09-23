# External Harbor worker setup

HarborRL treats workers as external machines. The training process never assumes
that it can install packages on them silently.

## Prerequisites

- Linux, Docker Engine, SSH access from the training node, and Python 3.10+.
- Harbor `0.23.0` installed at the path configured in `harbor.python`
  (`/opt/harbor/bin/python` in the example).
- The worker can pull or locally build each task image declared by the catalog.
- The worker can reach the training node's Messages gateway on the configured
  host and port. No provider API key is used on the worker.

## Install

```bash
python3 -m venv /opt/harbor
/opt/harbor/bin/pip install harborrl==0.1.0
/opt/harbor/bin/python -m harborrl.rollout.harbor_job --help
```

Use a pinned installation method provided by your Harbor distribution if it is
already managed by your platform. Record the exact binary path and Harbor
version in the training YAML.

## Runtime identity and health

`harborrl train` passes the worker SSH destination, native catalog, gateway URL,
policy profile, and attempt budget through a sanitized child environment. The
worker runner:

1. creates one immutable Harbor trial per attempt;
2. binds the Claude Code client to the Messages gateway;
3. rejects missing serving evidence or a stale policy version;
4. writes the verifier result and reward artifact under the Harbor trial;
5. closes the trial even when generation or verification fails.

Check a worker without launching training by running its runner help over SSH
and by running `harborrl doctor --config examples/native/train_qwen_native.yaml`
on the training node. Docker must be able to pull `debian:bookworm-slim` for the
bundled hello-world task.

## Security

- Do not copy provider keys, Claude settings, proxy credentials, or private task
  data to workers.
- Keep task catalogs immutable; a changed task digest fails configuration.
- Run each worker as an unprivileged account with access only to its Docker
  socket and task workspace.
