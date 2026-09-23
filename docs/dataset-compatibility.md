# Dataset compatibility

The machine-readable inventory is
[configs/harbor_hub/manifests.yaml](../configs/harbor_hub/manifests.yaml).
No external dataset is marked `supported` for the public v0.1 release yet; this
avoids inheriting unverified private histories.

| Dataset | Status | Reason |
| --- | --- | --- |
| `hello_world` | `example` | bundled offline task fixture; contract only, not Hub execution |
| `terminal-bench-core` | `planned` | public source; inspector and live worker matrix pending |
| `terminal-bench-test` | `planned` | public source; pinned revision and digest policy pending |

## Promotion process

A dataset may become `supported` only after recording:

1. upstream repository, revision, branch, license, and download command;
2. expected task count and local task-root layout;
3. content digest and repair policy;
4. Docker image, CPU, memory, storage, network, and GPU requirements;
5. static inspector profile;
6. baseline and solution reward-contrast run on every configured worker class;
7. one native training batch through task execution, receipt, IR, export, and
   Slime adapter;
8. owner and acceptance date.

Any transition from `supported` to a lesser status requires a documented blocker
and maintainer approval. Private research datasets and task trees are not added
to this public inventory.
