# Remote worker (pool server)

This directory runs on the **CPU worker**: a pool server that manages Docker containers and executes terminal tasks on behalf of GPU training nodes.

Site-specific proxy credentials must remain outside the repository. Worker
runtime, watchdog, and private-DinD scripts are tracked here; provisioning and
recovery actions live in `../ops/`, while credential-bearing environment files
are still created only on workers.

This directory contains reusable worker-side runtime only. RJob submission
and the private-DinD RJob entrypoint (`start_rjob_dind_worker.sh`) live under
`local/rjob/` because RJob is this site's cluster form and those are
operator-side pu-dev helpers, not deployment payloads.

---

## Active scripts (current workflow)

### Runtime entrypoints

| Script | When to use |
|---|---|
| `local/rjob/start_rjob_dind_worker.sh` | Starts the private RJob dockerd and optional SETA pool (operator-side, lives with the submission scripts). |
| `docker_watchdog.sh` / `start_watchdog.sh` | Monitors dockerd and pool lifecycle. |

### Steady-state (every training run)

| Script / file | Role |
|---|---|
| `run_pool_server.sh` | Hardened pool server launcher. Optionally sources the build-proxy env, sanity-checks dockerd, configures capacity, starts uvicorn. Credential-isolated runtime-proxy deployments must set `SKIP_PROXY_ENV=1`. Pass `--restart` to replace the local server in one command (stops the listener on `ENV_SERVER_PORT`, default 18082 in that mode). |
| `harborrl/platform/worker_app.py` | FastAPI service exposed on port 18081 |
| `harborrl/environments/terminal/runtime.py` | Environment client used by pool server |
| `harborrl/environments/terminal/docker_compose.py` | Helper to build / up / down compose stacks |

### Manual ops

| Script | Role |
|---|---|
| `../ops/docker_worker_doctor.sh` | Diagnose or repair a CPU/Docker worker. |
| `../ops/cleanup_docker.sh` | Remove stopped containers, build cache and dangling state. |
| `../ops/provision_worker.sh` | Provision a new CPU/Docker worker. |
| `../ops/fix_dockerd_and_proxy.sh` | Repair Docker and proxy state. |
| `../ops/restart_docker_force.sh` | Force-restart dockerd when systemd is wedged. |

Runtime setup assets such as proxy units, dependency pins and image prewarm
helpers live in `../runtime/`.

---

## Quick start (assuming machine already set up)

From the repo root:

```bash
bash deploy/workers/run_pool_server.sh
```

Then on the GPU worker:

```bash
export WORKER_URLS="http://<this-cpu-worker-ip>:18081"
bash examples/training/train_qwen3_8b_seta_dive_po.sh
```

For the fixed12 evaluation, edit `deploy/workers/worker_urls.txt` instead of
exporting a permanent URL.  Its local router reloads the file every five
seconds.  New leases use the updated worker; existing leases remain pinned to
the worker that created their Docker session.  Multiple URLs may be written
one per line or comma-separated.

---

## Optional environment variables (read by `harborrl/platform/worker_app.py`)

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | `benchmarks` | Path to the task dataset directory |
| `TBENCH_OUTPUT_ROOT` | `<server-run>/task_outputs` | Root for build/output artifacts under the worker's structured run log directory |
| `ENV_SERVER_PORT` | `18081` | Port the pool server listens on |
| `WORKER_MAX_TASKS` | `16` | Max tasks allocated per worker |
| `WORKER_MAX_RUNS_PER_TASK` | `8` | Max concurrent runs per task |
| `WORKER_MAX_TOTAL_RUNS` | `WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK` | Global reserved run/network ceiling across tasks; counts active leases plus leases whose async Docker cleanup is still retiring |
| `TBENCH_DOCKER_IMAGE_SOURCE` | `build` | `build` or `pull` — build locally or pull from registry |
| `TBENCH_DOCKER_PULL_PREFIX` | — | Image prefix for `pull` mode |
| `COMPOSE_OVERRIDE_PATH` | — | Optional Docker Compose override file |

Capacity sizing rule (from issue #3 §1):
```
The effective capacity is additionally capped by `WORKER_MAX_TOTAL_RUNS`; set it
below the host's available Docker address-pool capacity so retries queue at
`/allocate` instead of failing later during `docker compose up`. Capacity is
released only after close/force-cleanup completes, not when a lease merely
leaves the active map. Monitor `total_active_runs`, `total_retiring_runs`, and
`total_reserved_runs` in `/status`.
```

For 8×4 (current default), 16×8=128 has been observed to saturate dockerd. v2 launcher defaults to a more conservative 64×16=1024 to leave headroom.

---

## Archived scripts

The superseded `start_server_legacy.sh` is retained under `deploy/archive/` for
rollback/reference. It is not used by the current `run_pool_server.sh`
workflow. Other files with no static references remain in their active
functional directories until an operator confirms they are safe to remove.
