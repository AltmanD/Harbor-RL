# Task format

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

`task.toml` must contain `environment`, `agent`, and `verifier` tables. The
instruction and required files must be nonempty. The verifier runs in the shared
task environment. Native v1 does not enable simulated users, bridges,
multi-step declarations, separate verifier environments, accelerators, compose
files, or non-Linux task operating systems.

The inspector hashes relative paths, file modes, lengths, and bytes. A catalog
must record the resulting SHA-256 digest; later content changes fail
configuration. Symlinks are rejected.

## Reward artifact

The Harbor verifier terminal result contains a shared-environment result and a
`rewards` mapping. The trial directory must also contain `reward.json` (or
`reward.txt`) with the same selected value. HarborRL independently resolves both
through the catalog's reward profile and records source bytes and digests in an
immutable receipt. Boolean and nonfinite values are invalid.

## Minimal example

See [examples/native/tasks/hello_world](../examples/native/tasks/hello_world).
Its instruction asks the agent to create `/hello`; its verifier compares the file
and emits reward 1. The offline smoke constructs the complementary reward-0
receipt so group centering is observable without Docker.
