# HarborRL Exporter 重构方案（Slime v0.3.2 评估版）

日期：2026-09-23

评估对象：`THUDM/slime` tag `v0.3.2`，commit `3778dbf6d1a533ab478ecf5ddaa11449a47752b2`

结论：**Slime v0.3.2 明显降低 HarborRL 对 Slime 源码的侵入需求，值得作为 exporter 重构的首选目标版本。** 它已经提供 custom rollout function、custom converter、actor 侧 rollout-data postprocess、custom advantage 和 custom loss hook，并把多 turn 展开样本按 rollout 保持在一个 training step 内。理论上，当前 6 个文件、约 50 行的 Native Slime patch 大部分可以删除，改为 HarborRL 侧 adapter；最终仍需 CPU 契约测试和 GPU 单步验收确认。

## 1. 背景与目标

### 1.1 当前问题

当前 HarborRL Native MVP 直接修改 vendored Slime，主要解决五类问题：

1. 将 Slime 默认 generation group 替换成 HarborRL Native group。
2. 在 policy pool 更新后向 HarborRL 发布版本事实。
3. 阻止 Slime 按 fixed batch 裁剪 Native 多 turn 样本。
4. 让 `native_advantages`、`native_token_weights`、`native_padding` 穿过 Ray / Megatron 边界。
5. 让一个 Native group 在 DP 下完整对齐和消费。

这导致 HarborRL 与 Slime 内部实现强耦合，后续升级和抽象 training backend 都比较困难。

### 1.2 重构目标

重构后的边界应为：

```text
HarborRL Native rollout / IR / reward
  → backend-neutral TrainingBatch contract
  → Slime v0.3.2 adapter
  → Slime + Megatron training
```

具体目标：

1. `harborrl` 核心不 import Slime 类型，也不依赖 Slime 内部 batch 字段。
2. Slime 相关逻辑只存在于 `harborrl/backends/slime/`。
3. 优先使用 Slime v0.3.2 官方扩展点，避免维护源码 patch。
4. 保留 HarborRL 的训练语义：完整 group、组内中心化 advantage、逐 token loss mask、轨迹等权 reduction、policy version 证据。
5. 为未来接入第二个 training backend 留出稳定接口。

### 1.3 非目标

1. 不在本次重构中实现第二个真实训练 backend。
2. 不重写优化器或 Megatron 训练器。
3. 不承诺多轮长程训练稳定性；验收标准仍是有界单步 / 少步训练。
4. 不继续扩展旧 interactive exporter；新接口以 Native IR v2 为唯一输入。

## 2. Slime v0.3.2 能力评估

### 2.1 与 HarborRL 需求的对应关系

| HarborRL 需求 | Slime v0.3.2 能力 | 结论 |
| --- | --- | --- |
| 替换 rollout 生成逻辑 | `--rollout-function-path` 加载自定义 `generate_rollout(args, rollout_id, data_source, evaluation=False)` | 可消除对 `sglang_rollout.submit_generate_tasks` 的源码修改 |
| 多 turn 展开后不裁剪 group | `build_dp_schedule` 按 `rollout_id` 分 step，保证同一 rollout 展开的多个 training sample 留在同一 step | 可消除当前 Native 专用 trim patch |
| DP / microbatch 对齐 | v0.3.2 使用 pack-first、distribute-second 调度，并为 PP 对齐 microbatch 数 | 可删除当前复制 padding 样本的 `dp_batch` 路径 |
| 自定义样本到训练数据转换 | `--custom-convert-samples-to-train-data-path` | 可承载 HarborRL IR 到 Slime batch 的映射 |
| 训练前修正数据 | `--rollout-data-postprocess-path`，actor 收到完整 rollout 数据后调用 | 可用于 DP 切分后的本地字段校验和补充 |
| 自定义 advantage | `--custom-advantage-function-path`，原地写 `advantages` / `returns` | 可承载 Native GRPO advantage 语义 |
| 自定义 loss | `--loss-type custom_loss` + `--custom-loss-function-path` | 可保留 HarborRL clipped weighted PG loss |
| 外部 rollout 引擎 | `--rollout-external-engine-addrs`，支持外部 SGLang、router 和磁盘/NCCL 权重同步 | 更符合 HarborRL Gateway 与训练进程解耦的部署形态 |
| policy version | SGLang response `meta_info.weight_version`，`Sample.weight_versions` 会自动记录 | 可用于 rollout 证据；更新完成通知仍需 adapter 明确校验 |

### 2.2 v0.3.2 已解决的关键问题

#### 2.2.1 自定义 rollout 入口

v0.3.2 将 rollout function 定义为一级扩展点，签名如下：

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False):
    ...
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)
```

HarborRL 可以提供：

```text
harborrl.backends.slime.rollout.generate_rollout
```

该函数内部完成：

1. 从 `data_source` 获取 Native task groups；
2. 查询并锁定当前 policy weight version；
3. 调用 Messages Gateway / Harbor Runner；
4. 收集 reward 和 IR；
5. 将每个 turn 转成 Slime `Sample`；
6. 返回 `RolloutFnTrainOutput`。

这样不再需要修改 Slime 的 `GenerateState.submit_generate_tasks`。

#### 2.2.2 Group 原子调度

v0.3.2 的 `slime.utils.dp_schedule.build_dp_schedule` 明确保证：

- 按 `rollout_id` 分 training step；
- 一个 rollout 展开的多个 training sample 保留在同一 step；
- DP rank 拥有相同 microbatch 数以满足 PP 同步；
- 不需要复制 loss-bearing padding sample。

HarborRL adapter 只需保证：

1. 同一 prompt group 的所有轨迹和 turn 使用相同 `rollout_id`；
2. `rollout_batch_size` 表示 prompt group 数，而不是展开后的 turn 数；
3. `global_batch_size` 语义与 Slime v0.3.2 对齐。

#### 2.2.3 标准字段承载 Native 语义

v0.3.2 的训练侧 batch 字段仍有白名单，不提供任意 tensor 字段透传。但足够用标准字段表达 Native 语义：

| HarborRL 语义 | v0.3.2 表达方式 |
| --- | --- |
| prompt / response token | `tokens`、`response_lengths` |
| 可训练 token mask | `loss_masks` |
| rollout logprob | `rollout_log_probs` |
| 原始 reward | `raw_reward` |
| 已中心化 advantage 标量 | `rewards`，再由 custom advantage 展开为 `advantages` / `returns` |
| 单条轨迹 token 总数 | `rollout_mask_sums`，由 HarborRL converter 按轨迹填充 |
| clipped PG loss | `custom_loss` |

其中 `rollout_mask_sums` 是关键：custom loss 可以基于每个样本所属轨迹的 token 总数重建 `1 / (group_size * trajectory_token_count)` 权重，从而保留“每条轨迹等权、token 内均匀分摊”的 Native reduction 语义。

#### 2.2.4 Policy version

推荐放弃“修改 Slime publish hook”，改为在 HarborRL adapter 的 rollout 入口显式执行版本屏障：

1. rollout 开始前调用 rollout engine 的 weight-version / readiness 接口；
2. 确认所有可更新 engine 返回同一版本；
3. 将该版本写入 Native identity；
4. Gateway 校验每个生成 response 的 `meta_info.weight_version` 与 identity 一致；
5. exporter 拒绝混合版本 batch。

Slime v0.3.2 的 external rollout engine 模式天然更适合这个方案：训练进程负责权重更新，HarborRL rollout adapter 负责版本事实检查。

### 2.3 仍未解决的问题与风险

| 问题 | 影响 | 处理 |
| --- | --- | --- |
| v0.3.2 不透传任意 tensor 字段 | 不能继续携带 `native_advantages` / `native_token_weights` 私有 key | exporter 改用标准字段和 custom loss 重建语义 |
| custom loss 只能看到被请求的标准 batch 字段 | `rollout_id`、trajectory metadata 不能直接进入 loss | converter 在 `rewards`、`rollout_mask_sums` 中提前表达必要信息 |
| policy 更新完成没有 HarborRL 专用 listener | 需要显式版本屏障 | adapter rollout 前查询并锁定 weight version |
| v0.3.2 API 与当前 vendored 版本差异较大 | 现有 launcher / rollout 代码不能直接替换 | 单独 migration 分支重写 adapter |
| GPU 语义未验证 | 理论方案不等价于验收完成 | 必须复跑 Native MVP GPU 验收 |
| Megatron 依赖复杂 | 版本漂移会破坏训练 | 使用 Slime v0.3.2 官方镜像或锁定 bootstrap |

## 3. 推荐版本策略

### 3.1 版本基线

以 Slime v0.3.2 官方 Docker 组合为第一基线：

| 组件 | 版本 / commit | 来源 |
| --- | --- | --- |
| Slime | `v0.3.2`，commit `3778dbf6d1a533ab478ecf5ddaa11449a47752b2` | official tag |
| Megatron-LM | commit `1dcf0dafa884ad52ffb243625717a3471643e087` | Slime v0.3.2 `docker/Dockerfile` |
| SGLang 基础镜像 | `slimerl/sglang:v0.5.15.post1-cu129` | Slime v0.3.2 `docker/Dockerfile` |
| sglang-router | v0.3.2 官方镜像构建使用的 release wheel | Slime v0.3.2 `docker/Dockerfile` |

短期优先使用 Slime 官方镜像或完整镜像 digest，避免在 HarborRL 仓库中复制依赖树。若必须裸机安装，应提供独立 lockfile / bootstrap，而不是把 Slime 和 Megatron 源码 vendor 进 HarborRL。

### 3.2 版本规则

1. HarborRL 配置显式声明 backend contract version，例如 `training.backend_contract: slime-v0.3.2-native-v1`。
2. doctor 检查：
   - Slime package version；
   - Slime source commit（可从安装环境或 lock 记录读取）；
   - Megatron import path / commit；
   - SGLang router version；
   - 必要 hook 参数是否可用。
3. 不跟随 mutable branch；升级必须变更 contract 并跑回归。
4. 若发现仍需 patch，先判断能否通过 upstream extension 解决；只有 GPU 语义必需且无官方 hook 时才保留 patch。

## 4. 新模块设计

### 4.1 目标目录

```text
harborrl/
├── export/
│   ├── __init__.py
│   ├── contract.py          # backend-neutral schemas
│   ├── native.py            # Native IR → TrainingBatch，纯 Python
│   └── validate.py          # schema / digest / weight-sum validation
└── backends/
    └── slime_v032/
        ├── __init__.py
        ├── rollout.py       # Slime rollout_function_path hook
        ├── converter.py     # TrainingBatch → Slime train data
        ├── advantage.py     # custom_advantage_function_path hook
        ├── loss.py          # custom_loss_function_path hook
        ├── postprocess.py   # rollout_data_postprocess_path hook
        ├── launcher.py      # 参数组装与 fail-closed 校验
        └── versions.py      # weight version / backend version checks
```

`harborrl/export` 不得 import `slime`、`torch`、`megatron`。所有 Slime 类型只在 `harborrl/backends/slime_v032/` 中出现。

### 4.2 中性数据契约

引入三个核心对象：

```python
@dataclass(frozen=True)
class TokenSpan:
    tokens: tuple[int, ...]
    prompt_length: int
    response_length: int
    loss_mask: tuple[bool, ...]
    old_logprobs: tuple[float, ...]
    policy_version: str


@dataclass(frozen=True)
class TrajectoryTrainingUnit:
    trajectory_id: str
    rollout_id: int
    group_key: str
    reward: float
    advantage: float
    token_count: int
    turns: tuple[TokenSpan, ...]


@dataclass(frozen=True)
class TrainingBatch:
    schema_version: str
    groups: tuple[TrainingGroup, ...]
    reduction: str
    policy_versions: frozenset[str]
```
```

约束：

1. `TrainingBatch` 是不可变纯数据对象。
2. 每个 group 必须包含 `n_samples_per_prompt` 条完整轨迹。
3. 每个 trajectory 的 advantage 由 `export/native.py` 统一计算。
4. 所有 token 长度、logprob 长度、mask 长度必须一致。
5. 所有 policy version 必须唯一，否则构建失败。
6. 权重公式固定为 `trajectory_weight = 1 / group_size`，每个有效 token 的权重为 `trajectory_weight / token_count`。

### 4.3 Export pipeline

```text
Native IR groups
  → validate_ready / identity checks
  → compute group mean/std/advantage
  → expand trajectory turns
  → TrainingBatch
  → backend adapter
```

拆分后的职责：

- `export/native.py`：只输出中性 `TrainingBatch`。
- `backends/slime_v032/converter.py`：把 `TrainingBatch` 输出为 Slime v0.3.2 期望的标准字段。
- `backends/slime_v032/loss.py`：基于 `rollout_log_probs`、`advantages`、`loss_masks`、`rollout_mask_sums` 计算 clipped PG loss。

## 5. Slime v0.3.2 Adapter 设计

### 5.1 Rollout hook

`rollout.py` 提供符合 v0.3.2 contract 的入口：

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise NotImplementedError("Native evaluation rollout is not part of v1")
    groups = data_source(args.rollout_batch_size)
    policy_version = lock_policy_version(args)
    ir_groups = asyncio.run(run_native_groups(args, groups, policy_version))
    batch = export_training_batch(ir_groups)
    return RolloutFnTrainOutput(
        samples=to_slime_samples(batch),
        metrics=collect_metrics(batch),
    )
```

实现要求：

1. `lock_policy_version` 在任何生成请求前执行。
2. 同一 prompt group 的全部 turn 共享同一 `rollout_id`。
3. 所有 turn 的 `weight_versions` 与 Native IR identity 一致。
4. 失败 group 不允许部分进入训练，除非显式启用 partial rollout，v1 默认关闭。
5. 返回 metrics 包含 group 数、轨迹数、turn 数、reward mean/std、policy version。

### 5.2 Converter hook

`converter.py` 实现 `convert_samples_to_train_data(args, samples)`：

1. 从 samples metadata 或外部 IR store 还原完整 `TrainingBatch`；
2. 复核 identity、turn coverage、tokens、logprobs、mask；
3. 输出 Slime 标准字段：
   - `tokens`
   - `response_lengths`
   - `rewards`：已中心化的 trajectory advantage；
   - `raw_reward`：原始 training reward；
   - `truncated`
   - `sample_indices`
   - `rollout_ids`
   - `loss_masks`
   - `rollout_log_probs`
   - `rollout_mask_sums`
4. `rollout_mask_sums[i]` 填充 sample i 所属 trajectory 的有效 token总数，而不是 Slime 默认的 prompt group 总 token 数；该字段由 HarborRL custom loss 消费，并在文档中明确声明。

禁止输出：

- `native_advantages`
- `native_token_weights`
- `native_padding`
- 其他 Slime 不识别的私有训练 key。

### 5.3 Advantage hook

`advantage.py` 实现官方签名：

```python
def compute_advantages_and_returns(args, rollout_data):
    ...
    rollout_data["advantages"] = [...]
    rollout_data["returns"] = [...]
```

由于 converter 已把每条 turn 的 `rewards` 设置为所属 trajectory 的中心化 advantage，此 hook 只需：

1. 校验 response length 与 advantage 数量；
2. 将每条 turn 的标量 advantage 复制到 response token；
3. 设置 `returns`；
4. 拒绝 NaN / Inf。

不在 actor 侧重新按 reward 分组，避免 DP 切分后缺少全组上下文。

### 5.4 Custom loss hook

`loss.py` 使用标准字段计算：

```text
loss = -E_clip(ratio, advantage) * token_weight
token_weight = 1 / (n_samples_per_prompt * trajectory_token_count)
trajectory_token_count = rollout_mask_sums[i]
```

关键点：

1. `loss_masks` 仍表示有效 response token。
2. `rollout_mask_sums` 携带 trajectory token 总数，用于重建 token weight。
3. 权重总和在完整 rollout batch 上应接近 1。
4. Slime v0.3.2 会按 step rollout 数缩放 custom loss；adapter 测试需明确该校正是否需要在 loss 内抵消。
5. DP / microbatch 只改变求和位置，不改变全局权重和。
6. 输出 metrics：`native_pg_loss`、`native_weight_sum`、`native_token_count`、`native_group_count`。

### 5.5 Actor postprocess hook

`postprocess.py` 用于防御性校验，而不是重新计算训练语义：

1. 校验标准字段存在；
2. 校验 response / mask / logprob / advantage 长度；
3. 校验 `rollout_mask_sums > 0`；
4. 校验本地样本权重可以重建；
5. 发现缺失字段 fail-closed。

如果 GPU 验证显示 custom loss 无法获得足够的全局归一化信息，再评估在此 hook 补充本地标准字段；仍不得恢复对 Slime 源码的白名单 patch。

## 6. 实施计划

### Phase 1：只读 feasibility spike（1–2 天）

1. 在独立环境安装或拉起 Slime v0.3.2 官方镜像。
2. 写最小 fake Native rollout，返回 2 个 group、每个 group 4 条轨迹、每条轨迹 1–3 turn。
3. 验证：
   - `--rollout-function-path` 可加载 HarborRL hook；
   - `build_dp_schedule` 不拆 rollout group；
   - custom converter / advantage / loss 均被调用；
   - custom loss 在 DP > 1 时可正确归约。
4. 输出结论：是否仍需要 Slime patch。

### Phase 2：定义中性 exporter（2–3 天）

1. 新增 `harborrl/export/contract.py`。
2. 将 `rollout/exporters/native.py` 的 group advantage / turn expansion 逻辑迁入 `export/native.py`。
3. 增加纯 Python golden tests：
   - 完整 / 缺失 turn；
   - 混合 policy version；
   - reward 0/1；
   - std 为 0；
   - token / logprob / mask 长度；
   - 全局权重和。
4. 确认核心模块无 Slime / Torch import。

### Phase 3：实现 Slime v0.3.2 adapter（3–5 天）

1. 实现 rollout、converter、advantage、loss、postprocess 五个 hook。
2. 修改 launcher，只生成 v0.3.2 官方参数：
   - `--rollout-function-path`
   - `--custom-convert-samples-to-train-data-path`
   - `--custom-advantage-function-path`
   - `--custom-loss-function-path`
   - `--rollout-data-postprocess-path`
3. 删除对以下源码修改的依赖：
   - `slime/rollout/sglang_rollout.py`
   - `slime/ray/rollout.py`
   - Megatron `actor.py` / `data.py` / `loss.py` / `model.py`
4. 保留可追踪的 adapter 版本和 hook contract 检查。

### Phase 4：版本锁定与 doctor（1–2 天）

1. 在配置中声明 Slime / Megatron / SGLang contract。
2. doctor 检查 backend 版本和 hook 参数。
3. 支持 `--dry-run` 输出完整 Slime 命令。
4. 对官方镜像 digest或 lockfile 做 hash 校验。

### Phase 5：CPU 契约测试（2–3 天）

必须覆盖：

1. Native IR → `TrainingBatch` golden output。
2. `TrainingBatch` → Slime batch 字段完整性。
3. custom advantage shape / finite 校验。
4. custom loss 与现有 `clipped_loss` 数值一致性。
5. DP 调度下权重总和不因样本切分改变。
6. rollout_id 语义：同 group 同 id，展开 turn 不拆 step。
7. policy version 混合时 fail-closed。
8. v0.3.2 launcher dry-run。

### Phase 6：GPU 验收（0.5–1 天，需资源）

最小验收：

1. Slime v0.3.2 官方版本组合启动成功。
2. Native 0/1 reward contrast group 生成成功。
3. rollout logprob 与 Gateway 审计一致。
4. 非零 advantage、gradient、PG loss。
5. 显式参数变化。
6. checkpoint 保存并重载。
7. 新 policy version 生效。
8. 与当前已验证 MVP 的关键指标差异在允许误差内。

未通过 GPU 验收前，不删除旧 adapter；可以并存 `slime-legacy` 和 `slime-v032`，但默认入口必须只有一个。

## 7. 测试与验收矩阵

| 层级 | 测试 | 通过标准 |
| --- | --- | --- |
| Pure exporter | golden / malformed input | 全部拒绝非法输入，输出 schema version |
| Numerics | 0/1 reward、zero std、turn weights | 与当前公式一致 |
| Slime contract | fake rollout + fake batch | 只使用 v0.3.2 官方 hook，不 patch 源码 |
| DP scheduler | 1/2/4 DP、变长 turn | group 不拆 step，权重总和稳定 |
| Policy version | mixed / stale version | fail-closed |
| Launcher | dry-run | 命令完整、路径无内部默认值 |
| GPU | 单步 Native GRPO | 非零梯度与权重变化 |
| Compatibility | version doctor | Slime/Megatron/SGLang 不匹配时拒绝启动 |

## 8. 预期收益

1. 删除当前 Slime 源码 patch，降低版本升级成本。
2. HarborRL 核心只暴露中性 `TrainingBatch`，不再绑定 Slime `Sample`。
3. Slime 成为第一个 training backend adapter，后续可接 verl / OpenRLHF / 自定义 Megatron runner。
4. CPU 测试可以覆盖 exporter 数值语义，不必依赖 GPU。
5. Slime v0.3.2 external rollout engine 模式与 HarborRL Gateway 的部署边界更一致。
6. 公开仓库可以只保留 HarborRL 核心和薄 adapter，不再 vendor 第三方整仓。

## 9. 决策建议

1. **采用 Slime v0.3.2 作为 exporter 重构目标版本**，先做 feasibility spike。
2. **重构目标是零 Slime patch**；只有 GPU 验收发现官方 hook 无法表达训练语义时才降级为最小 patch。
3. **先建立中性 exporter，再迁移 adapter**，避免把 Slime 字段继续扩散到核心代码。
4. **GPU 验收通过前保持旧 MVP 路径只读保留**，通过后在新 release 中删除 legacy adapter。
5. **公开版本锁定官方版本组合**：Slime `v0.3.2`、Megatron `1dcf0dafa...`、SGLang `v0.5.15.post1-cu129`，并以实际镜像 digest 或 lockfile 补齐最终 reproducibility 信息。
