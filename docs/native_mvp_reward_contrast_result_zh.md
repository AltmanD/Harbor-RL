# Native MVP reward 对比与非零参数更新结果（2026-09-22）

## 结论

**Native MVP 最后一个验收缺口已经关闭。** 在同一真实
HumanEvalFix python-1 任务上，Qwen3-8B 自然采样得到完整有效 4-sample
GRPO 组，原始 Harbor verifier 给出自然 0/1 对比，Slime 导出非零
advantage，Megatron actor 产生非零梯度，checkpoint 权重发生实际变化，
并以 `weight_version=2` 通过 SGLang 重载语义探针。

最终验收文件：

```text
/mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-contrast-20260922/mvp-reward-contrast-final-acceptance.json
```

`all_checks_pass: true`。

成功运行：

```text
run_id:  20260922-153741-7f74640f
path:    /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-contrast-20260922/training/20260922-153741-7f74640f
Ray job: terminal_rl_8b_8gpu_20260922_153752
status:  succeeded
```

## 数据链路澄清

rollout 数据不是绕过 Slime 直接进入 Megatron：

```text
Harbor/Claude rollout
→ native trace / IR
→ Slime native group hook
→ Slime custom converter（token、mask、old logprob、advantage、weight）
→ Slime Ray actor
→ MegatronTrainRayActor（Slime 训练后端）
```

Harbor 原始轨迹和工具结果不直接喂给 Megatron。Megatron 只消费 Slime
转换后的训练张量；日志出现 Megatron 是因为 loss、梯度、optimizer 和
checkpoint 最终在 `MegatronTrainRayActor` 内执行。

关键代码边界：

```text
harborrl/rollout/native_generate.py
harborrl/rollout/exporters/native_slime.py
backends/slime/slime/backends/megatron_utils/loss.py
backends/slime/slime/backends/megatron_utils/model.py
```

## 任务与执行锁

任务：

```text
HumanEvalFix python-1 / separate_paren_groups
source digest: 2e7a5f49ae9566bf6d11ed3c0a13fa913fcb1ffad66ba7779852c407b178dc4d
execution digest: 01fd388b37e627c92bdb87041f7cc89f23de8f6f1645f444db7a870695902eaa
```

执行锁与证据：

```text
runs/native-mvp-contrast-20260922/task-humanevalfix-python1-claude-preinstalled/
runs/native-mvp-contrast-20260922/native-catalog-humanevalfix-python1.json
runs/native-mvp-contrast-20260922/humanevalfix-python1-lock-evidence.json
```

镜像：

```text
harbornative:humanevalfix-python1-claude-2.1.141
```

两台 worker 的原始 verifier 对照均为：

```text
baseline reward = 0
official/reference reward = 1
```

证据：

```text
runs/native-mvp-contrast-20260922/image-validation-humanevalfix-python1/worker-0/result.json
runs/native-mvp-contrast-20260922/image-validation-humanevalfix-python1/worker-1/result.json
```

任务 instruction、buggy workspace、official solution、verifier 测试均与
prepared task 完全一致；派生镜像只预装精确 Claude Code ELF 和运行时网络
适配，不改变 reward 语义。

## 自然 reward 对比

最终导出的 4 个 slot 全部有效：

| slot | reward | readiness | turns | session |
|---:|---:|---|---:|---:|
| 0 | 0 | `RL_READY` | 11 | 1 |
| 1 | 0 | `RL_READY` | 见 acceptance JSON | 1 |
| 2 | 1 | `RL_READY` | 见 acceptance JSON | 1 |
| 3 | 0 | `RL_READY` | 见 acceptance JSON | 1 |

因此：

```text
reward values: [0, 0, 1, 0]
reward std:    0.4330127018922193
```

没有注入 reward、没有混入官方 solution、没有人为篡改失败样本。

## Slime / Megatron 训练信号

Slime rollout 数据：

```text
advantages:        -0.06414986401796341
native_padding:    0
response_lengths:  89.5925925925926
total_lengths:     20405.222222222223
```

Megatron actor update：

```text
grad_norm_pre_clip: 4.6897874464334315
grad_clip_scale:    0.21322927988143617
grad_norm_effective: 1.0
native_pg_loss:     1.8570244450260093e-05
native_weight_sum:  0.037037032621878165
```

这证明 reward 对比已经穿过 Slime converter 并在 Megatron 训练后端产生
真实非零梯度。

## 显式参数更新

Megatron torch_dist checkpoint：

```text
runs/native-mvp-contrast-20260922/training/20260922-153741-7f74640f/checkpoints/20260922-153741-7f74640f/iter_0000000
```

转换后的 HF checkpoint：

```text
runs/native-mvp-contrast-20260922/training/20260922-153741-7f74640f/updated-hf
```

与原始 Qwen3-8B 权重逐张量比较：

```text
tensor_count:       399
element_count:      8,190,735,360
changed_tensors:    310
changed_elements:   121,298,254
max_abs_diff:       1.9073486328125e-06
mean_abs_diff:      1.8684926476282402e-08
rms_diff:           1.6764529032951467e-07
exactly_equal:      false
```

差异量级符合 `lr=1e-6` 和 bf16 checkpoint 的预期。证据：

```text
runs/native-mvp-contrast-20260922/training/20260922-153741-7f74640f/updated-hf-weight-diff.json
```

## Updated checkpoint 重载

转换后的 HF checkpoint 由 SGLang 以：

```text
weight_version = 2
```

实际加载，并通过 raw HF 模型对照探针：

```text
max_abs_logprob_delta:  0.0019657090306282043
mean_abs_logprob_delta: 0.0004981456565857911
tolerance:              0.1
passed:                 true
```

证据：

```text
runs/native-mvp-contrast-20260922/training/20260922-153741-7f74640f/updated-hf-semantic-probe.json
```

## 有界重试与 fail-closed 行为

HumanEvalFix python-1 上，Qwen 有时会在编辑后调用不存在的 `Check` 工具或
生成 malformed tool JSON。Gateway 严格拒绝这些输出，不会猜测或改写成
Bash 命令。为诊断该现象，曾对失败前上下文独立采样 32 次：

```text
valid tool call:              13
unknown / invalid tool call:  19
```

证据：

```text
runs/native-mvp-contrast-20260922/qwen-raw-output-samples.json
```

最终运行使用有界 retry：无效 predecessor attempt 不进入训练；每个 slot
只有拿到完整 sealed trace、单一 Claude session、raw-model logprob 和
verifier receipt 后才导出。最终 4 个有效样本组成完整 group。

## 清理状态

运行和重载探针后确认：

- 8 张 GPU 显存均为 0 MiB；
- Ray / SGLang 无残留进程；
- Gateway 32123 与探针 30100 不监听；
- 两台 CPU worker 无任务容器；
- 无 Harbor runner / Claude 残留进程；
- 无 `cleanup-required.json`；
- 无 `trace-unsealed.json`。

证据：

```text
runs/native-mvp-contrast-20260922/training/20260922-153741-7f74640f/cleanup-verification.json
```

## 最终验收矩阵

| 检查 | 结果 |
|---|---:|
| Ray job | PASS |
| 完整有效 4-sample group | PASS |
| 自然 reward 0/1 对比 | PASS |
| 原始 Harbor verifier | PASS |
| trace integrity / session audit | PASS |
| raw-model logprob | PASS |
| 所有 turn consumption confirmed | PASS |
| 每 attempt 单一 Claude session | PASS |
| 非零 advantage | PASS |
| 非零 gradient / PG loss | PASS |
| policy version `1 → 2` | PASS |
| Megatron checkpoint | PASS |
| 显式权重变化 | PASS |
| updated checkpoint 重载抽样 | PASS |
| worker/GPU/trace 清理 | PASS |

因此，此前 `native_mvp_gpu_result_zh.md` 中唯一遗留的
“reward 0/1 对比与非零参数更新”缺口已经关闭。
