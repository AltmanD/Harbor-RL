# Native MVP 8-GPU 真实全链路结果（2026-09-22）

> 后续更新：本文记录 hello-world 首次 8-GPU 全链路运行及其当时遗留的
> reward 对比缺口。该缺口已在同日后续运行
> `20260922-153741-7f74640f` 中关闭，见
> [native_mvp_reward_contrast_result_zh.md](native_mvp_reward_contrast_result_zh.md)。

## 结论

**Native MVP 的真实 GPU 主链路已经跑通。** 8×H200 Pod 上完成了
Ray/Megatron 训练、SGLang rollout、Messages Gateway、Claude Code 工具调用、
CPU worker Harbor 原生任务、原始 verifier reward、RL IR 导出、GRPO 训练步、
checkpoint 保存、策略版本发布以及 checkpoint 重载抽样。

成功运行：

```text
run_id: 20260922-120924-856003d7
path:   /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260922/training/20260922-120924-856003d7
Ray job: terminal_rl_8b_8gpu_20260922_120935
status: succeeded
```

机器学习验收摘要：

```text
/mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260922/mvp-fullchain-acceptance.json
```

本文只陈述已经发生的真实运行证据；`runs/` 产物与 checkpoint 不进入 Git。

## 验收矩阵

| 检查 | 结果 | 说明 |
|---|---:|---|
| Claude Code → Messages Gateway | PASS | request/response/消费确认均闭环 |
| Gateway → SGLang Qwen3-8B | PASS | 每次请求唯一 request id |
| raw-model logprob 审计 | PASS | `logprob_semantics=raw_model` |
| Claude 工具调用 | PASS | 两个 attempt 都实际调用 `Write` |
| Harbor 原始 verifier | PASS | reward 来自任务原始测试 |
| 每 attempt Claude session 数 | PASS | 均为 1 |
| RL IR readiness | PASS | 两条轨迹均为 `RL_READY` |
| trace seal / delivery audit | PASS | 无错误，所有 turn `consumed_confirmed` |
| Megatron 训练步与 checkpoint | PASS_WITH_ZERO_GRADIENT | 任务组内 reward 无差异，零梯度正确 |
| 策略版本发布 | PASS | policy pool `1 → 2` |
| checkpoint 重载与继续采样 | PASS | reload 语义探针通过 |
| worker/GPU 清理 | PASS | 8 卡显清零，容器清理 |
| reward 0/1 对比与非零参数更新 | NOT_PROVEN | 需要更难的真实任务 |

两次独立 policy attempt 的关键证据：

```text
attempt caa36c5f...: 2 turns, 4 request ids, reward 1, RL_READY
attempt 702f4099...: 2 turns, 4 request ids, reward 1, RL_READY
```

完整 JSON 中的 request id、Harbor trial id、trace 路径和 policy version
以 `mvp-fullchain-acceptance.json` 为准。

## 为什么本运行梯度为零

hello-world 任务对 Qwen3-8B 已经饱和：两个独立采样都得到原始 verifier
reward 1。GRPO 组内 advantage 因此为 0：

```text
reward values:        [1.0, 1.0]
advantages:           0.0
grad_norm_pre_clip:   0.0
native_pg_loss:       0.0
native_weight_sum:    0.25
```

这是算法上的正确结果，不是 loss/tensor plumbing 失败。不能通过注入
reward、混入官方 solution、篡改 verifier 或人为让一个样本失败来关闭该
验收项。后续必须选择一个真实原始任务，使策略在自然采样中出现 0/1
对比，再验证非零梯度和参数变化。

## 部署与网络事实

GPU Pod：

```text
hostname: gpu-lg-cmc-h-h200-0668.host.h.pjlab.org.cn
IP:       100.97.196.1
GPU:      8× NVIDIA H200
```

CPU worker：

```text
worker-0: root@100.98.161.238 / luyd-dev
worker-1: root@100.103.145.28 / dev
```

Gateway 使用：

```text
http://100.97.196.1:32123
```

不要使用 31999：该端口被平台 nginx 拦截/保留，会返回 release-center HTML。
最终配置为：

```text
/mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260922/native-config.yaml
```

布局为 actor 4 GPU + rollout 4 GPU，`max_context=32768`，
`audited_raw_logprobs=true`。hello-world 的 Claude 请求渲染约 24k token，
仍在上下文预算内。

## 离线依赖与执行镜像

GPU Pod 不联网。`mbridge==0.15.1` 先在联网 worker 上准备为 wheel，再通过
共享存储给 GPU Pod 离线安装：

```text
.tools/offline-wheels/mbridge-0.15.1-py3-none-any.whl
SHA256 5c7c007c2eb13bb2d0ce9bd6b313537024d0a7fad52179ff367635eafe5cd4cd
```

原任务镜像中 Claude 冷安装会造成 agent setup 超时。执行锁使用预装精确
Claude ELF 的派生镜像：

```text
harbornative:hello-world-a411e432-claude-2.1.141
```

构建上下文与证据：

```text
runs/native-mvp-20260922/image-claude-preinstalled/
runs/native-mvp-20260922/task-execution-claude-preinstalled/
runs/native-mvp-20260922/native-catalog-claude-preinstalled.json
runs/native-mvp-20260922/claude-preinstalled-lock-evidence.json
```

执行锁 task digest：

```text
9003a0be0d29bcb826538c351884b46b72a86ed31e639d46bda114c4dcb191ae
```

`instruction.md`、原环境 Dockerfile、verifier 测试内容未变；`task.toml`
只改变执行镜像和运行时代理资源配置。该派生镜像不改变任务语义。

## Checkpoint 与重载

Megatron torch_dist checkpoint：

```text
runs/native-mvp-20260922/training/20260922-120924-856003d7/checkpoints/20260922-120924-856003d7
```

checkpoint 约 123GB，后续转换得到 HF 权重：

```text
.../checkpoints/20260922-120924-856003d7/reloaded-hf
```

转换与重载证据：

```text
.../checkpoint-to-hf.log
.../reloaded-checkpoint-semantic-probe.json
```

SGLang 以 `weight_version=2` 加载转换后的 HF checkpoint，并完成文本/token
一致性探针。由于共享存储剩余空间有限，不应无意义复制 123GB checkpoint
或 16GB 转换模型。

## 清理状态

运行后检查确认：

- Ray 已停止；
- 8 张 GPU 显存均为 0 MiB；
- Gateway 32123 不再监听；
- worker 上无 Harbor task 容器；
- 无 Harbor runner/Claude/compose 残留进程；
- 成功 run 下无 `cleanup-required.json`、`failure.json` 或
  `trace-unsealed.json`。

## 回归状态

当前 native 相关 CPU 回归：

```bash
python -m pytest -q \
  tests/harborrl/test_native_contracts.py \
  tests/harborrl/test_native_gateway.py \
  tests/harborrl/test_native_lifecycle.py \
  tests/harborrl/test_native_training.py \
  tests/harborrl/test_gpu_layout.py \
  tests/harborrl/test_gpu_layout_review.py \
  tests/harborrl/test_sglang_gpu_id_mapping.py
```

结果为 **99 passed**。

## 剩余工作

唯一未完成的 MVP 验收是：

```text
真实任务自然采样 reward 0/1 对比
→ 非零 advantage
→ grad_norm_pre_clip > 0
→ checkpoint 参数实际变化
→ 下一轮 admission 使用新的 policy version
```

候选任务必须保留原始 instruction、solution、环境构建配方和 verifier。
不得为制造对比而修改任务语义或 reward。
