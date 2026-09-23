# Slime v0.3.2 Exporter 重构开发状态

日期：2026-09-23
分支：`refactor/slime-v032-exporter-dev`
基线：`origin/feat/harborrl-mvp`（`999513bb`）
位置：独立 clone `/mnt/shared-storage-user/luyudong/Harbor-RL-refactor`（不占用共享 MVP 工作区）

## 已完成

### 中性 exporter（Phase 2）

- `harborrl/export/contract.py`：`TokenSpan` / `TrajectoryTrainingUnit` / `TrainingGroup` / `TrainingBatch` 不可变纯数据契约。
- `harborrl/export/native.py`：Native IR v2 → `TrainingBatch`；group mean/std/advantage、turn 展开、`trajectory_weight = 1 / (group_size * token_count)`。
- `harborrl/export/validate.py`：schema、identity、turn 覆盖、长度、mask、权重、rollout id、policy version fail-closed 校验。
- 该包不 import `slime` / `torch` / `megatron`，并有 AST 测试约束。

### Slime v0.3.2 adapter（Phase 3，CPU 契约层）

- `harborrl/backends/slime_v032/rollout.py`：`--rollout-function-path` hook；生成前锁定 policy version，生成后复验版本；同 group 共享 rollout id，展开 turn 不拆 group。
- `converter.py`：`--custom-convert-samples-to-train-data-path` hook；重建并复验完整 `TrainingBatch`；只输出 Slime 标准字段（`rewards`/`raw_reward`/`rollout_ids`/`rollout_mask_sums` 等），`rollout_mask_sums[i]` = 所属 trajectory 的有效 token 总数；无 `native_*` 私有 key。
- `advantage.py`：`--custom-advantage-function-path` hook；把已中心化的 trajectory advantage 复制到 response token，不在 DP 切分后重新分组。
- `loss.py`：`--custom-loss-function-path` hook；标准字段 + `rollout_mask_sums` 重建 token weight 的 clipped PG loss；纯 Python 数值参考 + Torch 训练实现。
- `postprocess.py`：`--rollout-data-postprocess-path` hook；actor 侧标准字段 / 长度 / mask sum / 权重预算防御校验。
- `versions.py`：`slime-v0.3.2-native-v1` contract、Slime `v0.3.2`（`3778dbf6…`）、Megatron `1dcf0daf…`、SGLang `v0.5.15.post1-cu129` 版本门禁与 policy-pool 锁读取。
- `launcher.py`：五个官方 hook path 的组装、assignment 校验与 dry-run 命令渲染。

### 接线与门禁（Phase 4 部分）

- `training.backend_contract` 配置（默认 `slime-legacy`，可选 `slime-v0.3.2-native-v1`），映射到环境变量 `HARBORRL_NATIVE_SLIME_CONTRACT`。
- `slime_train.sh` 参数组装支持 `legacy` / `slime-v032` 选择器，未知值 fail-closed；legacy 路径保持不变。
- doctor 在非 legacy contract 时静态校验五个 hook 模块可导入。

### CPU 契约测试（Phase 5 部分）

`tests/harborrl/test_slime_v032_exporter.py` 覆盖：golden 导出（0/1 reward、zero std、变长 turn、权重和）、混合 policy version / 缺 turn / 证据篡改 / 重复 turn fail-closed、标准字段完整性、advantage/loss/postprocess 数值与防御、rollout 原子性与版本锁、版本 doctor、launcher dry-run、export 包无后端依赖。`test_native_training.py` 增加 shell dry-run 的 legacy/v032 参数化断言。

## 未完成 / 后续门槛

1. **官方镜像 feasibility spike（Phase 1）**：在 Slime v0.3.2 官方 Docker 组合中验证五个 hook 实际加载与调用；本分支仅完成源码契约核对（commit `3778dbf6…`）与 CPU fake 测试。
2. **GPU 单步验收（Phase 6）**：0/1 reward contrast、非零 advantage/gradient/PG loss、权重变化、checkpoint 重载、新 policy version 生效。
3. **custom loss 缩放核对**：v0.3.2 会在 hook 外按 microbatch / step rollout 数 / DP 缩放；`LOSS_SCALE_CORRECTION` 当前为 1.0，必须在 GPU 验收中确认是否需要抵消。
4. **真实 rollout 集成**：`rollout.py` 默认 runner 复用现有 `native_generate.generate_group` 与 `policy-pool.json` 发布机制；v0.3.2 unpatched 环境下需要确认该发布时机（external rollout engine / adapter 侧显式发布）。
5. **Harbor Hub dataset matrix**：inspect / catalog / Native smoke 兼容回归。
6. 按 `docs/branch_management_plan_zh.md` §5.3，本分支验收后生成受限 patch 移植到 `feat/native-mvp-clean` 子分支，不直接 merge。

## 测试记录（本分支，CPU）

- `tests/harborrl/test_slime_v032_exporter.py`：12 passed。
- `tests/harborrl/test_native_training.py`：10 passed（含 legacy / slime-v032 shell dry-run）。
- `tests/harborrl/test_native_contracts.py` + `test_native_gateway.py` + `test_native_lifecycle.py` + 新测试：81 passed, 1 skipped。
- 存量环境问题：`test_static_hygiene.py::test_slime_hook_modules_importable` 因缺少 `pybase64` 失败；在基线 `999513bb` worktree 复现，非本分支引入。
