# Native MVP 完整链路真实训练准备（2026-09-21）

> **2026-09-22 状态更新**：本文的 P0 GPU 检查、8×H200 全链路运行、
> reward 0/1 自然对比、非零梯度、显式权重变化与 updated checkpoint
> 重载均已完成。最终证据见
> [native_mvp_gpu_result_zh.md](native_mvp_gpu_result_zh.md) 和
> [native_mvp_reward_contrast_result_zh.md](native_mvp_reward_contrast_result_zh.md)。
> 下文保留为启动前检查清单与运行前事实记录。

本文记录当前启动前状态。今日只做了非 GPU P0 准备和验证：
**没有启动训练、没有启动策略 rollout、没有调用 Messages Gateway/策略模型、
没有运行 Claude agent trial、没有在 GPU Pod 上尝试 Docker。**
两台 worker 上的 Harbor `install_only` 只验证任务装载与容器生命周期。

GPU SSH 指令暂不可用。资源方后续会即时提供；收到后填入 `GPU_SSH`，先执行
第 8.3 节基线检查，不能直接 `train`。

## 1. 运行前结论（历史）

**完整训练仍是 No-Go，剩余阻塞全部依赖 GPU 资源或 GPU 侧事实。**

非 GPU P0 已完成：

1. 源码 dirty tree 已用 Git bundle + binary patch + untracked source archive 固化。
2. 两台 worker 均有 pinned Claude Code `2.1.141`，共享 Harbor `0.23.0` venv 可用。
3. 找回真实原始 Harbor hello-world task，非 materialized 反推版本。
4. 原始 task digest 与 0.5 历史 manifest 完全一致。
5. 原始 Dockerfile 镜像已在 worker-0 完成 baseline=0 / official solution=1。
6. 为避免外网冷启动接近 verifier 900 秒预算，构建了可追溯 runtime 镜像；
   两台 worker 均 baseline=0 / official solution=1，耗时 74s / 65s。
7. 最终执行 task lock 指向 runtime 镜像，并显式配置 worker 代理与内部 PyPI。
8. 两台 worker 的 Harbor `install_only` 均成功，未调用 Claude 或策略模型。
9. 两台 worker 的 Docker 默认 bridge 创建/删除探针均通过。
10. Native/GPU 布局契约回归：**93 passed**。

仍待 GPU：

1. 新 GPU SSH 指令。
2. Qwen3-8B SGLang vs HF raw logits 语义探针。
3. Messages Gateway 在 GPU 可路由地址上的 worker 反向可达性。
4. 依赖完整拓扑的 `doctor + dry-run`。
5. 操作者明确批准后的最小真实训练。

## 2. 固定部署边界

```text
GPU Pod（8×H200，不运行 Docker task / 不做 DinD）
  ├─ Ray / Megatron actor
  ├─ SGLang rollout engine
  ├─ HarborRL Messages Gateway
  └─ native rollout 调度器
        │ SSH + stdin/stdout 生命周期协议
        ▼
两台共享存储 CPU worker
  ├─ Harbor 0.23.0 Runner
  ├─ Claude Code CLI 2.1.141
  └─ Docker / Harbor task / verifier
```

- GPU Pod 只承担 Ray/Megatron/SGLang/Gateway/native 调度。
- task、模型、代码和运行产物必须使用共享存储绝对路径。
- Messages Gateway 监听 GPU Pod 可路由地址，CPU worker 反向访问。
- Harbor Runner 输出路径在 GPU 侧和 worker 侧必须完全一致。
- 0.5 遗留 `/tmp/harborval18net/docker.sock` daemon 保留不动；本 MVP 使用
  系统 Docker daemon。

## 3. Worker 现状

| 角色 | SSH 入口 | 直连 IP | 状态 |
|---|---|---|---|
| worker-0 / `luyd-dev` | `luyd-dev.luyudong+root.ailab-narmodel.ws@h.pjlab.org.cn` | `100.98.161.238` | Ready |
| worker-1 / `dev` | `dev.luyudong+root.ailab-narmodel.ws@h.pjlab.org.cn` | `100.103.145.28` | Ready |

两台均通过：

```text
Python 3.12.13
Harbor 0.23.0
Trial.create callable
ClaudeCodeOptions schema loaded
Claude Code 2.1.141
Docker 29.6.0
default-bridge create/delete probe: ok
```

最终探针证据：

```text
runs/native-mvp-lock/20260921-hello-world/worker-probes/latest.txt
runs/native-mvp-lock/20260921-hello-world/p0-non-gpu-evidence.json
```

### Docker 修复记录

- `luyd-dev`：系统 Docker 元数据里存在默认 bridge，但内核缺 `docker0`。重启
  系统 Docker daemon 后重建，默认 bridge 容器创建/删除通过。
- `dev`：同样重建 `docker0` 后，容器仍出现 `rw layer lease not found` 并卡在
  Dead/removing。系统 Docker 已从 containerd-snapshotter 切回经典 `overlay2`，
  `none` 与默认 bridge 探针均通过。
- 两台机器的旧 `harborval18net` daemon 均未改动。

## 4. 源码快照

当前 base commit：

```text
96df255e46949b439618f12490b6dd49adfc39e4
```

快照目录：

```text
runs/native-mvp-lock/20260921-hello-world/source-snapshot/
```

包含：

- `harbor-rl-head.bundle`：base commit 的 Git bundle。
- `tracked-changes.patch`：当前 tracked dirty diff（binary patch）。
- `untracked-source.tar.gz`：native schema/runner/gateway/tests/docs/probe 等
  必要 untracked 源文件。
- `SHA256SUMS`：三件套校验。
- `snapshot.json`：commit、时间和恢复步骤。

已验证：clone bundle 后 `git apply --check tracked-changes.patch` 通过，archive
可完整列出。最终验收前应以最新 `SHA256SUMS` 为准。

## 5. Task lock 与 runtime 镜像

### 5.1 原始真实 task

来源为历史 manifest 指向的原始 Harbor task，不是 0.5 materialized 反推版本：

```text
runs/native-mvp-lock/20260921-hello-world/task
task_digest: a411e432e80e572143af9f4ba19b04fb209b21d3929024c571296b1e25e4bc3f
```

原始 verifier 与任务文件未修改。直接 Docker 验证：

- worker-0 baseline reward = `0`
- worker-0 official solution reward = `1`
- worker-0 official solution冷启动约 `865s`，暴露外网依赖风险
- worker-1 baseline reward = `0`；official solution 的外网下载被停止，不伪造成
  通过，用于保留风险证据

### 5.2 执行用 task lock

为避免每个 trial 都安装 curl/uv/Python/pytest，执行锁使用可追溯 runtime 镜像：

```text
runs/native-mvp-lock/20260921-hello-world/task-execution
task_digest: 0aa3f3a35066607226924290a90d09af9856925258f4282fdf530aadd155f4e5
catalog:     runs/native-mvp-lock/20260921-hello-world/native-catalog-execution.json
```

相对原始 task 只增加：

- `environment.docker_image = harbornative:hello-world-a411e432-runtime`
- worker HTTP/HTTPS 代理与 `NO_PROXY`
- 内部 PyPI mirror `UV_INDEX_URL`

原始 `instruction.md`、`solution/solve.sh`、`tests/test.sh`、`tests/test_state.py`
保持不变。

### 5.3 Runtime 镜像

```text
tag: harbornative:hello-world-a411e432-runtime
recipe: runs/native-mvp-lock/20260921-hello-world/docker/runtime.Dockerfile
```

构建内容：

- 基于原始 task Dockerfile 镜像。
- 预装 `ca-certificates`、`curl`、`python3`。
- 从 PyPI wheel 固化 `uv 0.9.7` / `uvx`，wheel SHA-256 为
  `8cf6bc2482d1293cc630f66b862b494c09acda9b7faff7307ef52667a2b3ad49`。
- 将官方 uv 安装脚本请求的 GitHub tarball 替换为本地同版本归档；其他 URL 仍走
  系统 curl。
- 使用系统 Python 3.12，避免 uv 下载 CPython 3.14。
- 预热 `pytest==8.4.1` 与 `pytest-json-ctrf==0.3.5` 缓存，运行时 `UV_OFFLINE=1`。

原始 verifier 0/1 验证：

| worker | baseline | official solution | elapsed |
|---|---:|---:|---:|
| `luyd-dev` | 0 | 1 | 74s |
| `dev` | 0 | 1 | 65s |

worker-0 使用 containerd-snapshotter，image inspect 显示 manifest-list ID
`sha256:5a96e118...`；worker-1 使用经典 overlay2，显示 amd64 config ID
`sha256:5123fc27...`。这不是内容漂移，recipe、base digest 和 layer 输入一致。
完整证据见 `p0-non-gpu-evidence.json`。

## 6. Harbor `install_only`

两台均使用最终执行 task lock、系统 Docker 与 `nop` agent 执行 Harbor public
Trial API `install_only`：

| worker | result | trial ID |
|---|---|---|
| `luyd-dev` | success | `611e6665-6771-4ea3-891e-4ba2b084b09d` |
| `dev` | success | `25e98362-1a9c-420e-967a-012e1e46f580` |

该检查只验证 task 装载、prebuilt image、容器启动/停止/删除，不运行 Claude，
不访问 Gateway，不调用策略模型，不执行 verifier。

## 7. GPU 前置占位符

以下值必须等 GPU 机器提供后填写；不要复用旧 SSH 指令：

```bash
GPU_SSH='<GPU_SSH_TO_BE_PROVIDED>'
GPU_IP='<GPU_ROUTABLE_IP>'
GATEWAY_PORT='31999'

RUN_DATE="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="/mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-${RUN_DATE}"
mkdir -p "$(dirname "$RUN_ROOT")"
```

模型仍为：

```text
/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B
tokenizer_digest: a292b49b00aebd1a2bca0ca466e09272157d7e95ecc738aed99186101ec96d23
template_digest:  6df37b4cc9e9ef8b9ebed358b2c32de1c3aa4657a0a89f96af0daf96029bcc7f
```

## 8. Schema-2 配置模板

`tasks.catalog` 必须指向最终执行锁 catalog，而不是原始 task 目录或 0.5
materialized 目录：

```yaml
schema_version: 2
execution: {"backend": "harbor_job"}
tasks:
  catalog: /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-lock/20260921-hello-world/native-catalog-execution.json
harness:
  name: claude_code
  profile: /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-lock/20260921-hello-world/claude.json
harbor:
  python: /mnt/shared-storage-user/luyudong/Harbor-RL/.venv-harbor-0.23/bin/python
  version: "0.23.0"
  workers:
    - luyd-dev.luyudong+root.ailab-narmodel.ws@h.pjlab.org.cn
    - dev.luyudong+root.ailab-narmodel.ws@h.pjlab.org.cn
gateway:
  host: "0.0.0.0"
  port: 31999
  advertised_url: "http://<GPU_ROUTABLE_IP>:31999"
  tokenizer_digest: "a292b49b00aebd1a2bca0ca466e09272157d7e95ecc738aed99186101ec96d23"
  template_digest: "6df37b4cc9e9ef8b9ebed358b2c32de1c3aa4657a0a89f96af0daf96029bcc7f"
  audited_raw_logprobs: false
model:
  checkpoint: /mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B
  reference: /mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B
  args_file: qwen3-8B
training: {"num_rollout": 1, "learning_rate": 0.000001, "save_interval": 1}
sampling:
  group_size: 2
  groups_per_batch: 1
  max_attempts: 2
  max_tokens: 1024
  max_context: 8192
deployment:
  layout: split
  num_gpus: 8
  actor_gpus: 4
  rollout_gpus: 4
  actor_tensor_parallel_size: 4
  rollout_gpus_per_engine: 4
output: {"root": "<NEW_UNIQUE_RUN_ROOT>"}
```

`claude.json`：

```json
{
  "cli_version": "2.1.141",
  "model": "policy",
  "max_turns": 8,
  "agent_timeout_sec": 1800,
  "setup_timeout_sec": 600,
  "verifier_timeout_sec": 900
}
```

## 9. GPU 到位后的检查顺序

### 9.1 CPU 回归

```bash
cd /mnt/shared-storage-user/luyudong/Harbor-RL
python -m pytest -q \
  tests/harborrl/test_native_contracts.py \
  tests/harborrl/test_native_gateway.py \
  tests/harborrl/test_native_lifecycle.py \
  tests/harborrl/test_native_training.py \
  tests/harborrl/test_gpu_layout.py \
  tests/harborrl/test_gpu_layout_review.py \
  tests/harborrl/test_sglang_gpu_id_mapping.py
```

### 9.2 Worker 复验

```bash
WORKERS=(
  luyd-dev.luyudong+root.ailab-narmodel.ws@h.pjlab.org.cn
  dev.luyudong+root.ailab-narmodel.ws@h.pjlab.org.cn
)
for worker in "${WORKERS[@]}"; do
  ssh -CAXY "$worker" '
    set -e
    hostname
    docker info --format "docker={{.ServerVersion}} driver={{.Driver}}"
    claude --version
    cd /mnt/shared-storage-user/luyudong/Harbor-RL
    .venv-harbor-0.23/bin/python -m harborrl.rollout.harbor_job.runner --probe
    docker run --rm harbornative:hello-world-a411e432-runtime true
  '
done
```

### 9.3 GPU 基线

```bash
ssh -CAXY "$GPU_SSH" '
  set -e
  hostname
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  /mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python - <<"PY"
import torch, ray, sglang, transformers
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print(ray.__version__, sglang.__version__, transformers.__version__)
PY
'
```

还必须确认：

- `/mnt/shared-storage-user/luyudong/Harbor-RL` 可见。
- GPU Pod 能 SSH 到两台 worker。
- 两台 worker 能访问 `http://$GPU_IP:$GATEWAY_PORT`。
- 不检查、不启动 GPU Pod 本地 Docker。

### 9.4 Qwen3-8B semantic probe

独立 SGLang 服务显式使用 `--weight-version 0`，然后执行：

```bash
cd /mnt/shared-storage-user/luyudong/Harbor-RL
CUDA_VISIBLE_DEVICES=<empty-gpu> \
/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python \
  tools/verification/sglang_semantic_probe.py \
  --endpoint http://127.0.0.1:<sglang-port> \
  --model /mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
  --tolerance 0.1
```

通过并保存 JSON 后才允许把 `gateway.audited_raw_logprobs` 改为 `true`。

### 9.5 doctor / dry-run

```bash
cd /mnt/shared-storage-user/luyudong/Harbor-RL
PY=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python
$PY -m harborrl.cli doctor --config "$NATIVE_CONFIG"
$PY -m harborrl.cli train --config "$NATIVE_CONFIG" --dry-run
```

doctor 必须全部 `ok: true`，尤其每个 external Harbor worker 均为 true。

## 10. 获得批准后的最小真实训练

只有第 9 节全部通过且操作者明确说“启动”后才执行：

```bash
cd /mnt/shared-storage-user/luyudong/Harbor-RL
PY=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python
$PY -m harborrl.cli train --config "$NATIVE_CONFIG" \
  2>&1 | tee "$RUN_ROOT-launch.log"
```

建议规模：

- 1 个 native task，group size 2，必须出现 reward 0/1 差异。
- 1 个 rollout、1 个 save interval。
- split：4 GPU actor + 4 GPU rollout。
- 最多 2 次 attempt；异常 trial 不进入训练。
- 成功后立即停止，不自动扩大规模。

验收还必须包含：

1. Gateway 每个 attempt 有唯一 credential 与 request/response/consumption 证据。
2. 每个 slot 恰好一个 Claude session；错误 trial、重复 session、缺消费证据均拒绝。
3. reward receipt 来自原始 verifier，且与 task lock 的 0/1 语义一致。
4. advantage/token weight 非零，Slime/Megatron 有非零梯度与参数变化。
5. 权重版本从 0 发布到 1，下一轮 admission 使用版本 1。
6. checkpoint 可重载并继续采样。
7. worker 无残留 container/network/cleanup-required 记录。
8. launch/source/task/worker/GPU/probe/log/checkpoint 全部落入共享 run root。

## 11. Go/No-Go

| 项 | 状态 | 阻塞级别 |
|---|---|---|
| Native 代码契约 | Ready | — |
| 源码快照 | Ready | — |
| 共享 Harbor 0.23 venv | Ready | — |
| Worker Claude CLI | Ready | — |
| 原始 native task lock | Ready | — |
| 执行 task lock + runtime 镜像 | Ready | — |
| Worker Docker / bridge / install-only | Ready | — |
| Qwen3-8B checkpoint/digest | Ready | — |
| GPU SSH | PASS（见 GPU 结果文档） | — |
| Qwen3-8B semantic probe | PASS（见 GPU 结果文档） | — |
| Gateway worker 可达性 | PASS（见 GPU 结果文档） | — |
| doctor/dry-run | PASS（见 GPU 结果文档） | — |
| 完整训练启动 | PASS（见 GPU 与 reward 对比文档） | — |

## 12. 常见误判

- “本机/worker Docker 可用，所以 GPU Pod 可用”：错误。GPU Pod 不跑 Docker。
- “materialized 六任务可直接 native 训练”：错误。当前最小验收使用找回的原始
  task 与执行锁。
- “Qwen2.5-0.5B 探针通过即可开审计标志”：错误。必须用生产 Qwen3-8B 复验。
- “CLI 在 GPU Pod 存在即可”：错误。Claude agent 由 CPU worker 执行。
- “doctor 只看 GPU”：错误。它必须通过每台外部 worker 的 SSH probe。
- “runtime 镜像改了 verifier”：错误。verifier 和测试文件未改；只预置依赖并
  将同版本 uv 归档本地化。
