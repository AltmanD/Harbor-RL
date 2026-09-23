# Native MVP clean branch 真实训练验收（2026-09-23）

## 结论

`feat/harborrl-mvp` 在 8×H200 GPU 机上完成两轮真实 GRPO 训练：

- 1、2 号开发机各并行运行 4 个 Harbor trial（每个 group 8 个采样，4+4 分布）；
- 两组 rollout 均出现自然奖励方差（0.5–0.8 分级奖励）；
`grad_norm_pre_clip` 分别为 10.38 / 6.74，`native_pg_loss` 非零；
- policy pool 版本 1→2→3，第二轮 rollout 确认使用版本 2 的更新后权重；
- 最终 Megatron checkpoint（iteration 1）转换为 HF 后与基线 Qwen3-8B 对比：
**399 个张量中 310 个变化，148,482,952 / 8,190,735,360 个参数变化，
max |Δw| = 3.81e-6（≈ 2 × lr），非零参数更新成立**。

完整证据：
`runs/native-mvp-20260923-reward-contrast/mvp-reward-contrast-clean-branch-acceptance.json`。

## 分支与提交

- 验收 HEAD：`f7a81629`（= 基线 `9132f06e` + docs + 两个修复）
- `54964b1d fix(native): merge split Claude session blocks for audit`
- `f7a81629 fix(native): spread first rollout attempts across Harbor workers`
- GPU 机回归：**102 passed**（原 99 项 + 3 个新增回归）

### 修复 1：Claude 会话分块审计适配

策略模型在一条 assistant 消息里并行发出 4 个 Write tool call 时，
Claude Code 2.1.141 会把同一 message id 按内容块写成多行 JSONL。
原审计按“同 id 不同内容即冲突”拒绝，导致所有采样失败。适配器按块合并后
仍与网关不可变 trace 逐一比对（`registry.consume`），未匹配即 fail-closed。

### 修复 2：rollout worker 均衡

原实现用重试序号而非 slot 序号选择 worker，8 个首次尝试全部落在 1 号机、
2 号机空闲，重试也不换机。修复后按 slot 轮询、重试顺延下一台机器；
本轮验收两个 group 均为 4+4 并发。

## 任务与配置

- 任务：`graded-quiz-native-v1-claude-preinstalled`（10 道混合难度题，
  按答对比例给 0–1 的分级奖励；task digest
  `120e00d61450fb59865280f342a1c525f62d44f65607d3add05d50abd9297c72`）
- 先前的 secret-transport 抄写任务对 Qwen3-8B 完全饱和（8/8 全对、零梯度），
  已按证据保留并替换；
- 配置：2 rollout × group 8、max_attempts 2、temp=1/top_p=1、lr=1e-6、
  split 4(actor TP4)+4(rollout TP4)、save_interval=2（只写最终 checkpoint）。

## 训练信号（run `20260923-114103-ab5abb9c`）

| rollout | policy | rewards | grad_norm | rollout 耗时 |
|---|---|---|---|---|
| 0 | v1 | 0.5,0.6,0.7×6 | 10.38 | 68.8s |
| 1 | v2 | 0.7×7,0.8 | 6.74 | 71.4s |

端到端 450s（含 37.5s checkpoint 保存、Ray/SGLang/Megatron 启动）。

## 权重变化证明

Megatron torch_dist `iter_0000001`（107G）→ mbridge 转换 HF（16G）→
与基线逐张量对比：

```text
tensors=399 changed_tensors=310
elements=8,190,735,360 changed=148,482,952 (1.81%)
max_abs_diff=3.81e-06 mean_abs_diff=3.48e-08 rms=2.93e-07
```

报告：
`runs/native-mvp-20260923-reward-contrast/training/20260923-114103-ab5abb9c/updated-hf-weight-diff.json`

## 效率与并行度结论

| 指标 | step 0 | step 1 |
|---|---|---|
| step_time | 133.8s | 109.4s |
| rollout(train_wait) | 70.5s | 74.7s |
| actor_train | 63.3s | 34.7s |
| wait_time_ratio | 52.7% | 68.3% |
| actor tflops | 82.1 | 150.7 |

监控（5s 采样）显示：actor 4 卡利用率均值 9.4%，rollout 4 卡均值 2.3%
（峰显存 89G/90G）；两台开发机并发 4 容器时 load1 仅 1.1–1.7。

**判断：应该增加 rollout 并行度。** 当前瓶颈是等待（53–68%），而两侧资源
都远未饱和：开发机 CPU 余量大（4 并发仅 ~1 load），SGLang 引擎在 8 个并发
会话下利用率 <3%，gateway 并发信号量为 512。建议将
`groups_per_batch × group_size` 提升到 16–64（例如每机 8–16 slots），
rollout 吞吐应接近线性提升；后续瓶颈预计转移到 SGLang prefill/decode
与 Harbor 容器启动速率，再考虑增加 worker 数量。

## 运维记录

- GPU Pod 重建导致到开发机的 root SSH 失效；重新生成 ed25519 密钥并在两台
  worker 授权后 doctor 恢复 0 failed；
- 两次 run 之间需 `ray stop --force`，否则 Ray 6379 冲突导致下一次启动失败；
- checkpoint 预检要求共享卷 ≥128G 空闲；已按既有流程 retire 旧 dirty-branch
  run 的 107G Megatron checkpoint（保留其 updated-hf/日志/验收 JSON）；
- `save_interval=2 + num_rollout=2` 恰好只保存最终 iteration 1，避免 214G 双写；
- 当前 ipfs 卷保留本次 107G checkpoint + 16G updated-hf，剩余 ~90G。
