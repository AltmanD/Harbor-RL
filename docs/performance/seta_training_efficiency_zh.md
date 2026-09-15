# SETA 训练吞吐与 GPU 利用率

## 共享外置 worker 基线

统计对象是两条 8 卡 fixed12 运行，均取前 100 个 rollout-step。
每步为 4 个 prompt × 8 条 trajectory，并进行 2 次 actor update。

| 指标 | DAPO | DIVE-PO |
|---|---:|---:|
| 100 step 的 rollout 累计时长 | 48.85 h | 48.79 h |
| 每 step 均值 / p50 | 29.31 / 28.53 min | 29.27 / 28.05 min |
| p90 / p95 / 最大值 | 41.89 / 48.75 / 76.58 min | 44.91 / 53.54 / 67.24 min |
| actor 每 step 实际训练 | 77.79 s | 96.15 s |
| actor 等待比例 | 95.02% | 93.74% |
| checkpoint 占 rollout 总时长 | 0.138% | 0.139% |
| 按均值外推 1000 step（未计评估/故障） | 488.5 h / 20.35 d | 487.9 h / 20.33 d |

结论：actor 反向和 checkpoint 都不是主瓶颈。4 张 actor GPU 在约 94%–95% 的 step 时间等待 rollout，首先应缩短环境/agent 长尾并让 4 张 rollout GPU 承载更多独立请求。

可复算命令：

```bash
python tools/analysis/analyze_seta_throughput.py RUN_DIR --steps 100
```

结果写入 `RUN_DIR/metrics/analysis/throughput_analysis.{json,md}`。

## 8 卡同 Pod 私有 worker 实测

为避免拿 401.33 秒的单个最快 step 作为长期吞吐，固定比较两条
DAPO 的共同前 66 个 rollout-step：

- 旧配置：共享外置 worker，active cap=6，1×TP4 rollout engine；
- 新配置：同 Pod 私有 worker，active=32/reserved=40，4×TP1
  rollout engine。

其余共同口径为 8×H200、4 actor + 4 rollout GPU、batch=`4×8`、每个
rollout 两次 actor update。

| 指标（前 66 步） | 共享外置 | 同 Pod 私有 | 改善 |
|---|---:|---:|---:|
| rollout 均值 | 1773.56 s | 696.79 s | **2.55×** |
| rollout P50 | 1711.95 s | 650.02 s | **2.63×** |
| rollout P90 | 2577.79 s | 1184.69 s | **2.18×** |
| rollout tokens/GPU/s | 58.40 | 171.09 | **2.93×** |
| actor 实际训练 | 74.59 s | 82.82 s | 无提升 |
| actor 等待比例 | 95% | 86% | -9 pp |
| 100 step 纯 rollout 外推 | 49.27 h | 19.36 h | 省 29.91 h |
| 1000 step 纯 rollout 外推 | 20.53 d | 8.06 d | 省 12.47 d |

结论：长窗口端到端吞吐提升约 **2.5×**。actor 反向未变快，checkpoint 均值也
仍约 19 秒；收益主要来自 4×TP1 推理并发、32 路环境并发、localhost 链路、
私有 NVMe Docker cache、镜像预热/构建去重和消除共享 worker 竞争。由于 worker 位置、
环境并发和 rollout engine 拓扑同时变化，这是正式执行栈的系统级对比，不是
“仅把 Docker 搬入同 Pod”的单变量对比。外推未计 fixed12、启动和故障停顿。

## 已落地的优化

下表是共享外置 worker 的历史/诊断 profile；算法、数据、batch 和 fixed12
协议不变。同 Pod 方案在 `sequential-throughput-v1` 上显式把 client cap
扩到 32，worker reserved 扩到 40。

| profile | 远程环境并发 | rollout engine | 用途 |
|---|---:|---:|---|
| `paired-compat-v1` | 6 | 1 × TP4 | 复现旧的双任务共享配置 |
| `sequential-env12-v1` | 12 | 1 × TP4 | 单变量验证环境并发收益 |
| `sequential-throughput-v1` | 12 | 4 × TP1 | 外置 worker 的单任务吞吐诊断 |

TP1 对 Qwen3-8B/H200 是可容纳的；四个独立 engine 能并行处理不同 trajectory，避免 TP4 为单个解码流占满全部 rollout GPU。该收益必须先用 1–3 step GPU smoke 实测，不能把理论并发倍数当作实际加速比。

同时选择性移植了最新版 slime 的轻量 trace 与 request timing 聚合。AgenticRL 会记录环境创建、SGLang generation、tool call 和 evaluator span；需要离线时间线时，在短 smoke 中设置：

```bash
export SLIME_SAVE_DEBUG_ROLLOUT_DATA="$RUN_DIR/metrics/traces/rollout_{rollout_id}.pt"
python slime/tools/trace_timeline_viewer.py \
  "$RUN_DIR/metrics/traces/rollout_0.pt" --no-serve
```

正式 1000-step 默认不保存每步 debug dump，避免放大存储压力；聚合 timing 指标仍会进入 `perf/request/*`。

## AgentENV 设计对照

对照了 AgentENV `3da36143216d` 的 watermark warm pool、network slot pool、
Firecracker 预热进程和 snapshot/OverlayBD 路径。可直接复用的原则是“有界资源池 +
异步维护 + 生命周期可观测”，而不是直接把当前 Docker 环境替换成 Firecracker。

HarborRL worker 现已在 `/status` 的 `lifecycle_latency_sec` 中提供最近 512 次（可由
`WORKER_LIFECYCLE_HISTORY_SIZE` 调整）以下阶段的 count、success/failure、mean、
P50、P95 和 max：

- `reset_admission_wait`：reset 进入有界并发槽之前的排队时间；
- `reset`：镜像准备、compose 创建和环境初始化的总时间；
- `exec_tool`、`evaluate`：局内工具和 grader 长尾；
- `close`：回收排队与环境关闭总时间。

这些指标只增加观测，不改变 reward、mask、采样或更新逻辑。若
`reset_admission_wait.p95` 高，先扩 worker/减少共享竞争；若 `reset.p95` 高而排队低，
优先预构建镜像和定位 compose；若 `exec_tool.p95` 高，则环境任务本身才是主要长尾。

暂不直接照搬 warm Docker network/container：共享 worker 的 network slot 是硬
容量，预留 warm network 会占住正式 lease 配额；而不同 SETA task 使用不同镜像和
compose，复用容器还必须证明 reset 后文件、进程和网络状态完全隔离。Firecracker
snapshot/OverlayBD 可作为独立 AgentENV backend 评估，但需要 `/dev/kvm`、镜像转换、
grader 语义一致性和完整任务回归，不能在当前正式基线中无验证切换。

## 后续优化顺序

1. 已完成系统级提速验证。若要分解收益，用同一 checkpoint/seed 各跑 3 step：
   先固定 4×TP1/cap6 只改 worker 位置，再在 Pod 内依次改 cap6→12→32；比较
   step wall time、P95 trajectory、request queue/e2e latency 和失败率。
2. 预构建训练集涉及的镜像并清理 `cached_failed`。镜像 build/reset 超时既制造长尾，也会令部分 group 变成 non-trainable。
3. 从 trace 中按阶段拆分 `environment_open`、`sglang_generate`、`environment_tool`、`environment_evaluate`；只针对占 p95 的阶段优化。
4. 若共享 worker 仍受 Docker network 硬上限约束，应在维护窗口调整
   Docker address pool，或增加独立 worker。重启 Docker daemon 会影响共享任务，
   不应在训练中直接操作。
5. fully-async 只作为独立算法/系统实验：它能绕开最慢 trajectory 的 batch barrier，但会引入策略陈旧度与入批选择偏差，不能无声明地用于 DAPO/DIVE-PO 正式对照。
6. 根据 worker `lifecycle_latency_sec` 决定是否值得开发 AgentENV backend；只有 reset/close 明确主导 p95，且 Docker 预构建仍无法改善时，才进入 snapshot/预热池原型。
