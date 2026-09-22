# Native MVP clean branch verification（2026-09-22）

## 分支与提交

- 基线：`96df255e46949b439618f12490b6dd49adfc39e4`
- 新分支：`feat/harborrl-mvp`
- 未 push；`feat/harborrl-0.5-mvp` 保持不变。
- 代码验证 HEAD：`9132f06e7cc7a5282d2fcbb0614eea60debf248e`（提交序列：`a26f83e0`、`460c1434`、`6cfebbc5`、`57a60206`、`64172205`、`9132f06e`）。

## Clean HEAD 验证

在开发机 1 号、Python `3.12`、Git worktree clean 状态下执行：

```bash
/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python -m pytest -q \
  tests/harborrl/test_native_contracts.py \
  tests/harborrl/test_native_gateway.py \
  tests/harborrl/test_native_lifecycle.py \
  tests/harborrl/test_native_training.py \
  tests/harborrl/test_gpu_layout.py \
  tests/harborrl/test_gpu_layout_review.py \
  tests/harborrl/test_sglang_gpu_id_mapping.py
```

结果：**99 passed in 14.63s**。

补充验证：

- lifecycle 集合在 Python 3.12 下连续 3 次执行，均 **12 passed**；
- 新增/修改 Python 模块 Ruff 通过；
- Slime launcher 及 shell library `bash -n` 通过；
- `git diff --check` 通过；
- `harborrl.cli train --dry-run` 使用 `runs/native-mvp-20260922/native-config.yaml` 生成 schema-2 native 计划，catalog、profile、workers、gateway 与 Slime launcher 环境均正确。

`doctor` 在开发机 1 号实际 SSH 检查两台 external Harbor worker，均返回 Python 3.12.13、Harbor 0.23.0 和完整 trial factory 配置；task lock、catalog、模型路径和依赖版本也通过。开发机 1 号无 GPU / CUDA runtime，因此 GPU budget 与 Megatron import 检查按预期 fail-closed，不能替代 GPU 机 doctor。

## GPU 状态

GPU 机在本轮早期可访问，并确认：

- 共享工作区分支为 `feat/harborrl-mvp`；
- 8×H200 显存均为 0 MiB；
- PyTorch `2.9.1+cu128`、Ray `2.53.0`、SGLang `0.5.6.post2`、Transformers `4.57.3`；
- CUDA available 为 true，device count 为 8。

随后平台 SSH 对同一入口开始持续返回 `Permission denied (publickey)`，因此最终 `9132f06e` 的 99 项集合未能在 GPU 机复跑。GPU 恢复后应先执行：

```bash
cd /mnt/shared-storage-user/luyudong/Harbor-RL
test "$(git rev-parse HEAD)" = 9132f06e7cc7a5282d2fcbb0614eea60debf248e
PY=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python
$PY -m pytest -q \
  tests/harborrl/test_native_contracts.py \
  tests/harborrl/test_native_gateway.py \
  tests/harborrl/test_native_lifecycle.py \
  tests/harborrl/test_native_training.py \
  tests/harborrl/test_gpu_layout.py \
  tests/harborrl/test_gpu_layout_review.py \
  tests/harborrl/test_sglang_gpu_id_mapping.py
$PY -m harborrl.cli doctor --config runs/native-mvp-20260922/native-config.yaml
$PY -m harborrl.cli train --config runs/native-mvp-20260922/native-config.yaml --dry-run
```

上述命令全部通过后，才可认为 clean branch 的 GPU 机复核完成。
