# TB2.1 + SPEAR baseline record

Date: 2026-09-09 (Asia/Hong_Kong)

## Intended configuration

The reproducible online baseline uses the Qwen3-8B SETA-compatible terminal
environment, DAPO as the on-policy objective, and the HarborRL SPEAR adapter:

| item | value |
| --- | --- |
| model | `/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B` |
| actor/rollout GPUs | 2 + 2 (4 GPUs total), TP=2 |
| rollout batch / samples | 4 prompts × 4 responses |
| optimizer | AdamW-equivalent Slime Adam, lr `1e-6`, constant schedule |
| DAPO | `eps_clip=0.2`, `eps_clip_high=0.28`, per-token loss, GRPO |
| overlong shaping | enabled, soft buffer `4096` (max response `8192` in SETA recipe) |
| SPEAR buffer | trajectory `2048`, score threshold `1.0`, positive group-relative advantage |
| SPEAR baseline/staleness | historical group-reward p50, baseline history `10240`, tolerate `10` policy steps |
| replay | decoupled policy loss, FIFO-staleness generic buffer, two train iterations/rollout |
| eval cost | `EVAL_N_SAMPLES=1`, `EVAL_TOP_P=0.7`, evaluation trajectory dump enabled |

The command used to validate the final current-branch command line (without
starting Ray, SGLang, or a worker) was:

```bash
DRY_RUN=1 \
RUN_DIR=/tmp/lightrl-spear-dryrun \
WORKER_URLS=http://127.0.0.1:18081 \
HF_CKPT=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
REF_LOAD=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B_torch_dist \
NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES=2 \
EVAL_INTERVAL=100000 SAVE_INTERVAL=100000 MAX_CKPT_KEEP=0 \
EVAL_N_SAMPLES=1 EVAL_TOP_P=0.7 \
SLIME_SAVE_DEBUG_ROLLOUT_DATA=/tmp/lightrl-spear-dryrun/eval_{rollout_id}.pt \
bash examples/training/train_qwen3_8b_seta_spear.sh
```

The dry run exited with `RC=0` and emitted the final `slime/train_async.py`
command containing `--enable-trajectory-replay`,
`--enable-trajectory-posadv`, `--baseline-buffer-size 10240`,
`--trajectory-tolerate-steps 10`, cosine replay warm-up parameters, and
`--n-samples-per-eval-prompt 1`.

The online retry used the following RJob launcher invocation (the launcher
passed the dataset and environment roots into the pod):

```bash
RJOB_NAME=lightrl-tb21-spear-smoke2-20260909 \
RUN_ID=lightrl-tb21-spear-smoke2-20260909 \
START_TRAINING=1 KEEP_ALIVE=0 \
TRAIN_SCRIPT=examples/training/train_qwen3_8b_seta_spear.sh \
DATASET_DIR=/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/envs \
ROLLOUT_PROMPT_DATA=/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/datasets/tb21_smoke4.filtered.jsonl \
NUM_ROLLOUT=3 EVAL_INTERVAL=3 SAVE_INTERVAL=3 MAX_CKPT_KEEP=1 \
bash local/rjob/start_seta_dapo_train_4g_dind.sh
```

For future retries, use the tracked TB2.1 wrapper and set
`USE_BLACKLIST=0` when the input filename already ends in `.filtered.jsonl`:

```bash
TRAIN_SCRIPT=examples/training/train_qwen3_8b_tb21_spear.sh \
TB21_DATASET=/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/datasets/tb21_full89.jsonl \
TB21_DATASET_DIR=/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/envs \
bash local/rjob/start_seta_dapo_train_4g_dind.sh
```

## Online execution status and blocker

Two post-change smoke attempts were made. The first reached the RJob control
plane but failed before container startup because the cluster CNI pool was
exhausted:

```text
Failed to create pod sandbox ... kubebrain-networking ...
no IP addresses available in range set: 100.98.237.1-100.98.237.254
```

The retry `lightrl-tb21-spear-smoke2-20260909` entered `Running`, loaded the
Qwen3-8B checkpoint, initialized SGLang, allocated all 16 TB2.1 rollout
leases, and wrote a trajectory. It then exposed a task-image blocker during
the first rollout:

```text
rjob logs job lightrl-tb21-spear-smoke2-20260909 -n 200
... POST /evaluate ... 500 Internal Server Error
response={"ok":false,"error":"tmux is not installed in the container. Please install tmux in the container to run Terminal-Bench."}
```

The run was stopped after this deterministic infrastructure failure to avoid
further GPU cost. Its measured artifacts are preserved at:

```text
run config: /mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/runs/training/lightrl-tb21-spear-smoke2-20260909/config/run_config.json
train log:  /mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/runs/training/lightrl-tb21-spear-smoke2-20260909/logs/train.log
trace:      .../trajectories/seta_task-adaptive-rejection-sampler-2eace3c2_iter0_rewm1p000_r0_g3_s14_6f76a5ec_20260909_133211
```

There is no valid pass@1/aggregate score for this run, and no estimate is
reported. Before retrying, rebuild the converted TB2.1 task images with
`tmux` (and `asciinema`, if required by the worker), then invoke
`examples/training/train_qwen3_8b_tb21_spear.sh` with the same configuration.

## Measured historical run (reference only)

For continuity, the latest completed HarborRL SPEAR-like terminal run is:

```text
run:  lightrl-seta-spear-s1234-153614-20260908-153614
path: /mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/runs/training/lightrl-seta-spear-s1234-153614-20260908-153614
```

It completed 30 rollouts on 4 GPUs and used the old worktree commit
`e8cf701b`. It is **not** a post-change TB2.1 full-89 result: it used the
SETA converted train set and `seta_fixed48_v2` evaluation, saved 16 responses
per evaluation prompt, and had `enable_trajectory_posadv=false`. The exact
expanded command is preserved in:

```text
/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl/runs/lightrl-seta-spear-s1234-153614-20260908-153614.launcher.out
```

The measured summaries at rollout 30 were:

| evaluation set | pass metric | value |
| --- | --- | ---: |
| `seta_fixed48_exploit` | `pass_at_1` | `0.0625` (3/48) |
| `seta_fixed48_explore` | `pass_at_8` | `0.2291666667` (11/48) |

The corresponding machine-readable files are under
`evaluations/seta_fixed48_{exploit,explore}/step_0030/summary.json` in the run
directory above. These values are provided as an engineering reference only;
they must not be presented as the requested fresh TB2.1 + current-SPEAR score.
