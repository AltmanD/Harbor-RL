# GPU 布局

统一入口支持固定 `split` 和 `colocate`，默认保持 split。不做自动布局选择或运行中 TP/DP 调整。

```bash
harborrl train --config configs/train_colocate_smoke.yaml --dry-run
harborrl doctor --config configs/train_colocate_smoke.yaml
harborrl train --config configs/train_colocate_smoke.yaml
```

profiles 使用显式环境插值提供模型、reference、catalog、worker；GPU 主机必须安装与当前 CUDA ABI 配套的训练环境。公共 profile 是配置示例，真实小规模训练还需按 `backend_options` 明确采样预算和输出目录。

| 配置 | 含义 |
|---|---|
| deployment.layout | split：不重叠分配；colocate：同一组卡交替使用 |
| deployment.num_gpus | 可用预算 |
| deployment.actor_gpus | 训练卡数 |
| deployment.actor_tensor_parallel_size | 训练TP，DP由训练卡数/TP派生 |
| deployment.rollout_gpus | 推理卡数 |
| deployment.rollout_gpus_per_engine | 单引擎卡数/TP，副本数由推理卡数/单引擎卡数派生 |

split 要求两组卡数之和不超过预算，允许6+2、2+6等不对称布局。colocate 当前要求训练卡数、推理卡数和预算相同。首轮限单节点、同步GRPO，PP/CP/EP=1。还需满足模型注意力头/KV头约束。

初始 global batch 为 `ROLLOUT_BATCH_SIZE × N_SAMPLES / 2`，须为训练 DP 的正整数倍。例如 DP3 可配置 batch=2、N_SAMPLES=6。后续 dynamic history 会按实际 turn 数重新计算 batch，可能裁剪尾部样本；同奖励组过滤后，再用零 loss 占位补齐到完整 batch，使 DP 各 rank 保持相同步数（真实样本保留、占位不贡献梯度）；DP 不同的运行需要核对实际训练样本数，不能只按相同 episode 数比较速度。

旧配置缺省 layout=split、训练TP=训练卡数、单引擎卡数=全部推理卡数，维持原有4+4语义。布局参数不能通过 EXTRA_GRPO_ARGS 追加覆盖。

colocate 强制训练和推理offload，禁用与当前memory saver冲突的expandable_segments。doctor 会实际检查memory saver预加载时能否导入Torch。如果出现CUDA符号缺失，使用backend_options.LD_LIBRARY_PATH显式指向与Torch匹配的CUDA runtime；该路径会记录并传给Ray。不要修改共享环境或猜测系统CUDA目录。

新入口在同步后检查全部推理引擎版本，采样前检查健康及版本，并在每条轨迹前后再次核对策略池。多引擎不接受单一HARBOR_VERSION_ENDPOINT作为整体版本证明。不同副本版本不一致时失败，不静默继续或切回split。

每个run包含launch.json、gpu-mapping.json、policy-pool-history.jsonl、layout-phases.jsonl、原始训练日志和规范IR。GPU映射通过Ray实际探测获得。切换耗时与端到端耗时分开统计，不能把全卡复用视为必然加速。

SIGINT/SIGTERM 会转发到本次启动器并停止本次 Ray job；退出时按 `leases/` 记录关闭本 run 获得的环境租约，结果保存在 `lease-cleanup.json`。硬杀（SIGKILL）仍依赖 worker 的租约回收。

恢复失败时保留运行证据，结束该run；回退使用新的run显式选择split。checkpoint是否包含optimizer取决于保存参数；model-only不能提供精确续训。

2026-09-17 三机实测（Qwen3-8B、8×H200、两台环境 worker）：

| 布局 | Camel | Claude Code |
|---|---|---|
| colocate 8：TP4×DP2，2×TP4 引擎 | 48条 RL_READY，8 rank各5次非零更新 | 24条 RL_READY，8 rank各6次非零更新 |
| split 6+2：TP2×DP3，1×TP2 引擎 | 48条 RL_READY，6 rank各4次非零更新 | 24条 RL_READY，6 rank各4次非零更新 |
| split 2+6：TP2，3×TP2 引擎 | 32条 RL_READY，2 rank各4次非零更新 | 16条 RL_READY，2 rank各4次非零更新 |
| split 4+4：原兼容布局 | 两次各48条 RL_READY，4–5次非零更新/rank | 两次各24条 RL_READY，4–6次非零更新/rank |
| split 4+2：预算8、实际6 | 32条 RL_READY，4 rank各4次非零更新；GPU6/7无模型进程 | 未单独验收 |

这些成功运行均完成后续版本采样、checkpoint与入口退出。DP2/DP3 已在过滤后奇数/不整齐样本的真实场景验证补齐，避免原逻辑中各 rank 步数不一致造成的 collective 停顿。早期一次 Claude colocate 全零奖励、一次 DP 尾批停顿均保留为失败/未达标证据，不计入上述成功结果。

当前不能根据这些小规模运行认定 colocate 必然更快。以三轮训练总耗时为例，Camel 的 split4+4 两次为170–173秒，colocate多次为97–130秒；但各次生成长度、奖励组过滤、真实训练片段及step数不同。修复后整个三轮循环（首次同步结束至最终同步结束，含保存）的单次观察为 Camel colocate约620秒、split4+4约577秒，Claude分别约309秒、276秒。这些是观测值，不是严格控制后的布局加速比。DP3采用不同采样预算，仅用于正确性验收。

默认继续用 split；需要更多训练卡或显存时显式选择 colocate，并先按实际任务预算验证收益。详细实验产物保存在 `runs/gpu-layout-20260917`，包括初始化失败、零更新与停顿记录。

负向验证已确认：真实初始化后注入 KV 恢复失败或一个副本版本落后，均在采样前失败退出，0轨迹、0更新，进程回收后GPU释放。阶段切换时发送 SIGTERM，入口退出143、本次 Ray job 为 STOPPED，Ray 服务仍存活；16个环境租约清理无错误，8张GPU显存全部释放。
