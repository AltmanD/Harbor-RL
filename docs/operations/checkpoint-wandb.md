# 存储：Checkpoint 与 W&B Offline

> 类型：可复用存储约定。适用：训练、评测和 RJob 运行目录的 checkpoint、指标与
> W&B 离线产物布局。

## 默认布局

默认持久化根目录是 `runs/.persistent`；长时任务建议通过
`LIGHTRL_PERSIST_ROOT` 指向站点的持久化存储：

```text
<LIGHTRL_PERSIST_ROOT>/
├── checkpoints/<run-id>/
└── wandb/<run-id>/
```

训练结构化指标和控制台镜像仍在 `runs/training/<run-id>/logs/`（评测对应
`runs/evaluation/`，测试对应 `runs/testing/`）。因此 checkpoint 介质暂时
不可写或空间不足时，训练可以继续，并保留完整训练日志。可用 `CKPT_ROOT`、
`WANDB_DIR` 或 `LIGHTRL_PERSIST_ROOT` 覆盖默认值。

W&B 默认 `WANDB_MODE=offline`、`WANDB_ENABLE=1`。离线模式不需要 API key，启动器
会移除 `WANDB_API_KEY`/`WANDB_KEY`，也不会把 key 放入命令行或 Ray runtime
metadata。训练结束后如需网页查看，在有网络且已登录 W&B 的节点执行：

```bash
wandb sync <LIGHTRL_PERSIST_ROOT>/wandb/<run-id>/wandb/offline-run-*
```

## 保存事务与磁盘满行为

Megatron 只有在更新 `latest_checkpointed_iteration.txt` 后才算提交成功；仅存在
`iter_NNNNNNN/` 目录不能证明保存完整。HarborRL 的清理规则是：

1. 保存前读取 tracker，只删除比 tracker 更新的半成品目录；tracker 缺失时仅当目录
   带 HarborRL per-run 管理标记才清除上次失败的首次保存，否则不删。
2. 按 `--max-ckpt-keep` 清理已提交旧版本，但永不删除 tracker 指向的最后有效版本。
3. 默认要求至少 128 GiB 空闲，或预估 checkpoint 大小的 1.15 倍（二者取大值）。
4. 空间仍不足时打印 `CHECKPOINT_SAVE_SKIPPED_NONFATAL`，跳过本次保存并继续训练。
5. 写盘抛错时打印 `CHECKPOINT_SAVE_FAILED_NONFATAL`；forward hook 和 offload process
   group 会在 `finally` 中恢复，避免一次保存失败破坏后续训练。

8B 模型可用以下环境变量调整估算，但不建议降低到单个完整 checkpoint 大小以下：

```bash
export CHECKPOINT_MIN_FREE_GB=128
export CHECKPOINT_EXPECTED_GB=0        # 0=首次按128 GiB，之后从有效版本估算并缓存
export CHECKPOINT_SPACE_MARGIN_RATIO=1.15
export MAX_CKPT_KEEP=1
```

如某个安全关键任务仍希望保存失败即终止，可显式设 `CHECKPOINT_SAVE_FATAL=1`。

## 排障

```bash
grep -E 'CHECKPOINT_(SAVE|CLEANUP)' runs/training/<run-id>/logs/train.log
cat <checkpoint-dir>/latest_checkpointed_iteration.txt
du -sh <checkpoint-dir>/iter_*
df -h <checkpoint-dir>
```

恢复只能使用 tracker 指向的目录。不要用 shell 按目录名盲删 `iter_*`；失败保存通常
留下“编号最新但不可恢复”的目录，这正是旧机制误删最后有效版本的原因。
