# HarborRL 分支管理、清理与公开发布方案

日期：2026-09-23

适用前提：`main` 是可信安全基线，不需要历史清理或改写；`feat/harborrl-mvp` 是首个完成 MVP 验收的功能分支，但同时包含大量实验性代码；exporter 重构的目标是切割 Slime 依赖。

配套文档：目录与文件清理范围继续以 `docs/public_release_cleanup_plan_zh.md` 为准；exporter 技术设计见 `docs/exporter_refactor_plan_zh.md`。本文档只负责分支、迁移、保留和同步策略。

## 1. 目标与原则

### 1.1 目标

1. 保留 `main` 作为可信、受保护的集成基线，不做历史改写。
2. 保留 `feat/harborrl-mvp` 的完整验收成果和实验性内容。
3. 从 MVP 中提取公开版需要的 Native 核心功能，避免把实验代码整体带入发布分支。
4. 在干净的 Native MVP 基线上完成 Slime v0.3.2 exporter 重构。
5. 按 `docs/public_release_cleanup_plan_zh.md` 清理发布分支，并合入 exporter 重构。
6. 保证已经通过 Harbor Hub 适配的数据集在公开发布版本中仍可检查、下载或引用、生成 catalog，并能进入 Native rollout。
7. 不删除任何不用提交的实验性文档、代码和记录；将它们保存在本地私有归档分支或继续留在 ignored 工作区。

### 1.2 原则

- `main` 只通过 reviewed PR 更新，不直接提交，不 force push，不重写历史。
- `feat/harborrl-mvp` 不直接作为公开发布分支整体合入 `main`。
- “清理”只作用于发布分支的 tracked tree；不使用 `git clean -fdx` 物理删除本地实验内容。
- Git 分支只自动保存 committed tracked 内容；ignored/untracked 内容必须显式加入私有归档分支，否则不会随分支保存。
- Harbor Hub 数据兼容性是发布阻断项，不能为了目录简洁移除必要 adapter。
- orphan branch 只是备用方案，不是默认路径。

## 2. 分支角色

| 分支 / 标签 | 起点 | 权限 | 用途 | 是否同步 GitHub |
| --- | --- | --- | --- | --- |
| `main` | 现有可信基线 | protected | 公开集成基线；只接收发布 PR | 是 |
| `feat/harborrl-mvp` | 当前 MVP 分支 | 本地 / 私有 | 首个 MVP 验收成果和继续验证的工作区；不作为公开清理基线 | 默认否 |
| `archive/harborrl-mvp-accepted` 或 tag `mvp/native-v0.1-accepted` | 当前 MVP HEAD | 本地 / 私有 | 固化首个验收完成时的 tracked tree | 否 |
| `archive/harborrl-local-experiments` | 当前 MVP HEAD | 本地 / 私有 | 保存 ignored/untracked 的实验性文档、代码、配置和文本记录 | 否 |
| `feat/native-mvp-clean` | `main` | feature | 从 `main` 选择性移植 Native MVP 核心代码、测试和 Harbor Hub 兼容层 | 可选私有 PR |
| `refactor/slime-v032-exporter` | `feat/native-mvp-clean` | feature | 实现 backend-neutral exporter 和 Slime v0.3.2 adapter | 可选私有 PR |
| `release/public-v0.1` | `feat/native-mvp-clean` | release | 按清理方案收敛 tracked tree，再合入 exporter 重构 | 是，最终 PR 回 `main` |
| `orphan/public-v0.1` | 无父提交 | emergency | 仅当普通分支迁移造成不可控污染时使用 | 只有完成审计后使用 |

`feat/native-mvp-clean` 是发布工作的关键缓冲层：它从可信 `main` 出发，只移植 MVP 中公开需要的代码，而不是把 `feat/harborrl-mvp` 的全部差异带入。

## 3. 推荐分支图

```text
main
  └── feat/native-mvp-clean
        ├── refactor/slime-v032-exporter
        └── release/public-v0.1
              ├── apply public_release_cleanup_plan
              ├── merge refactor/slime-v032-exporter
              ├── Harbor Hub compatibility gate
              └── PR back to main

feat/harborrl-mvp
  ├── archive/harborrl-mvp-accepted
  └── archive/harborrl-local-experiments
```

推荐顺序：

1. 冻结并归档当前 `feat/harborrl-mvp`。
2. 从 `main` 创建 `feat/native-mvp-clean`。
3. 从 MVP 选择性移植 Native 核心、测试和 Harbor Hub 兼容层。
4. 从 `feat/native-mvp-clean` 创建 `refactor/slime-v032-exporter`。
5. 从 `feat/native-mvp-clean` 创建 `release/public-v0.1`，先完成目录清理。
6. 将 `refactor/slime-v032-exporter` 合入 `release/public-v0.1`。
7. 通过 CPU、Hub dataset、doctor/dry-run 和 GPU 验收。
8. 将 `release/public-v0.1` PR 回 `main`。
9. `main` 合入后再打公开 tag，例如 `v0.1.0-public`。

## 4. 当前 MVP 与实验内容保留

### 4.1 固化 MVP 验收点

在改动前先记录当前状态：

```bash
git switch feat/harborrl-mvp
git status --short --branch
git tag mvp/native-v0.1-accepted
git bundle create /tmp/Harbor-RL-mvp-native-v0.1.bundle --all
```

如果倾向使用分支而非 tag：

```bash
git branch archive/harborrl-mvp-accepted feat/harborrl-mvp
```

tag 和 bundle 只保存 committed 对象；不包含 ignored/untracked 文件。

### 4.2 保存本地实验内容

创建本地私有归档分支：

```bash
git switch -c archive/harborrl-local-experiments feat/harborrl-mvp
git add -f docs/HarborRL_0.5MVP_20260920.md
git add -f docs/HarborRL_next_steps_20260920_zh.md
git add -f <其他明确要保存的 ignored 文档 / 代码 / 配置 / 文本记录>
git commit -m "archive: preserve local MVP experiments"
```

要求：

1. 必须逐路径 `git add -f`，不要用宽泛通配符一次性加入未知内容。
2. 该分支永不 push 到公开远端。
3. 敏感凭据、私钥、内部 IP、内部主机名和 token 不应新增到任何未来可能公开的分支；私有归档分支中也应尽量避免。
4. 大体积 checkpoint、镜像、trajectory、模型和 168 GiB `runs/` 产物不适合直接放入 Git 分支；继续作为 ignored 本地数据保存，或另用外部私有存储归档。
5. 文本性实验记录、历史方案、验证脚本和未公开代码可以放入归档分支。
6. 归档完成后返回工作分支时，不执行 `git clean`。

### 4.3 “清理”和“删除”的边界

发布分支中的清理动作包括：

- `git rm` 移除不该公开的 tracked benchmark 快照、vendored backend、历史研究代码和内部运维脚本。
- 重写 README 和公开 docs。
- 调整 `pyproject.toml` 和 package allowlist。
- 将必要能力改成外部依赖、下载器或 manifest。

清理动作不包括：

- 删除本地 `runs/`。
- 删除 ignored 的历史文档。
- 删除私有实验分支。
- 对 `main` 做历史重写。
- 在当前工作区执行无差别 `git clean -fdx`。

如切换分支时 ignored 文件造成冲突，优先使用单独 worktree 或手动移动到已声明的私有归档路径，不做删除。

## 5. Native MVP 迁移策略

### 5.1 创建干净移植层

```bash
git switch main
git switch -c feat/native-mvp-clean
```

然后按清单从 `feat/harborrl-mvp` 选择性移植，而不是 merge 整个 MVP 分支：

```bash
git checkout feat/harborrl-mvp -- \
  harborrl/config/native.py \
  harborrl/gateway \
  harborrl/rollout/native_generate.py \
  harborrl/rollout/harbor_job \
  harborrl/rollout/exporters/native.py \
  harborrl/trajectories/native.py \
  tests/harborrl/test_native_contracts.py \
  tests/harborrl/test_native_gateway.py \
  tests/harborrl/test_native_lifecycle.py \
  tests/harborrl/test_native_training.py
```

上述命令是示例，实际 allowlist 必须结合 `docs/public_release_cleanup_plan_zh.md` 和 import graph 调整。每次 checkout 后都要：

1. 检查 `git diff --cached --name-status`。
2. 确认没有带入 benchmark、vendored backend、deploy 运维和旧研究模块。
3. 按小提交拆分：核心契约、Gateway、rollout、training dispatch、测试、文档。
4. 运行 Native CPU 测试。
5. 保留 Harbor Hub 兼容层。

### 5.2 为什么不建议直接 merge 当前 MVP

`feat/harborrl-mvp` 相对 `main` 的 tree 差异包含：

- Native MVP 核心功能；
- 两个 vendored 后端和大量历史重构；
- 旧 interactive 栈、算法实验和评测工具；
- 内部部署、RJob、运维和历史记录；
- 部分 benchmark / dataset 变更。

直接 merge 会把无用实验性 tracked 内容一并带入 `main`，后续清理面更大，也容易误删仍被 Harbor Hub 兼容层依赖的模块。选择性移植能把“MVP 功能”和“实验遗留”分开。

### 5.3 如果坚持先在当前 MVP 分支重构

该路线可以作为工程加速路径，但必须增加一道“patch 移植”边界：

1. 在 `feat/harborrl-mvp` 上创建 `refactor/slime-v032-exporter-dev`。
2. 完成重构和验证，但不要删除实验代码。
3. 将重构差异限制在新 exporter 模块、Slime adapter、launcher、配置和测试内。
4. 生成受限 patch：

   ```bash
   git diff feat/harborrl-mvp..refactor/slime-v032-exporter-dev -- \
     harborrl/export harborrl/backends tests configs docs pyproject.toml
   ```

5. 将该 patch 应用到从 `main` 建立的 `feat/native-mvp-clean` 或其子分支。
6. 在干净分支重新跑 CPU 和 GPU 验收。
7. 发布分支只接收受限 patch，不直接 merge `refactor/slime-v032-exporter-dev`。

这样既满足“先在当前 MVP 分支重构”的工作习惯，又避免把 MVP 分支的实验性 tree 带进公开版本。

## 6. Exporter 重构分支

### 6.1 推荐起点

```bash
git switch feat/native-mvp-clean
git switch -c refactor/slime-v032-exporter
```

重构范围：

- `harborrl/export/`
- `harborrl/backends/slime_v032/`
- Native training launcher 的参数组装。
- 版本 doctor 和 dry-run。
- CPU 契约测试。
- 最小 GPU 验收配置。

不在该分支处理：

- benchmark 数据清理；
- README 大改；
- 历史实验代码删除；
- Harbor Hub 之外的评测工具迁移。

### 6.2 合并条件

`refactor/slime-v032-exporter` 合入 `release/public-v0.1` 前必须满足：

1. Native CPU contract / gateway / lifecycle / training 测试通过。
2. fake backend 数值测试与旧 exporter 一致。
3. Slime v0.3.2 官方 hook 均能加载。
4. launcher dry-run 输出完整且无内部默认地址。
5. 版本 doctor 能识别 Slime / Megatron / SGLang contract。
6. 至少完成一次 GPU 单步或等效有界训练验收。
7. Harbor Hub 已适配数据集的 catalog / rollout 兼容测试通过。

## 7. 公开发布分支

### 7.1 创建与清理

```bash
git switch feat/native-mvp-clean
git switch -c release/public-v0.1
```

在 `release/public-v0.1` 上执行 `docs/public_release_cleanup_plan_zh.md` 中定义的 tracked tree 清理。清理后的目标不是“删除所有数据能力”，而是：

- 不提交大数据快照和私有任务资产；
- 保留数据格式检查、下载/引用、manifest、digest、catalog 和 Native runner 兼容层；
- 保留公开最小任务和测试 fixture；
- 将外部 backend 改为 pinned dependency / bootstrap；
- 删除与研究主线无关的算法、评测、部署和历史记录。

### 7.2 合入 exporter 重构

清理完成后：

```bash
git merge --no-ff refactor/slime-v032-exporter
```

如果重构是在当前 MVP 上开发的 `refactor/slime-v032-exporter-dev`，不要直接 merge；使用第 5.3 节的受限 patch 移植。

### 7.3 同步 GitHub

同步顺序：

1. push `release/public-v0.1` 到私有远端或 GitHub 私有分支。
2. 运行 CI 和本地验收。
3. 发起 `release/public-v0.1` → `main` 的 PR。
4. 审查 PR diff 只包含本次预期新增和清理。
5. merge 到 `main`。
6. 在 `main` 上打 `v0.1.0-public` tag。
7. push tag。

默认不 force push `main`。`archive/*` 和 `mvp/*` tag 不 push 到公开远端。

## 8. Harbor Hub 数据兼容策略

### 8.1 发布定义

“已适配过的 Harbor Hub 数据集仍能使用”必须定义为：

1. 用户可通过外部 task root、Harbor Hub 下载或已有本地缓存获得数据。
2. `harborrl hub inspect` 或等价 CLI 能检查本地 task。
3. inspector 能识别该 dataset 已适配的 task profile。
4. catalog 能记录 dataset、revision、task path 和 digest。
5. Native runner 能按 catalog 执行任务。
6. reward receipt 能进入 Native IR。
7. exporter 重构后训练 batch 仍保持相同 group / trajectory / turn 语义。
8. 公开仓库不包含私有数据内容，也不依赖个人绝对路径。

### 8.2 必须保留或替代的模块

清理时不允许简单删除以下能力，除非同名 PR 提供替代：

| 能力 | 当前相关入口 | 发布要求 |
| --- | --- | --- |
| Harbor task 静态检查 | `harborrl/tasks/cli.py`、`harborrl/data/harbor/inspector.py` | 保留或重构为 native hub CLI |
| Native task preflight | `harborrl/data/harbor/native_inspector.py` | 保留，不允许破坏 task digest 和 profile |
| reward receipt 解析 | `harborrl/data/harbor/receipt.py` | 保留并纳入兼容测试 |
| 数据集下载 / 本地缓存 | `harborrl/data/download.py` 或后续 Hub downloader | 支持显式 task root 和 upstream revision |
| task materialize / probe | `harborrl/data/harbor/materializer.py`、`probe.py` | 若删除旧 `TerminalEnv` 依赖，必须先提供 Native runner 等价路径 |
| catalog schema | Native config `tasks.catalog` | 字段兼容或提供显式 migration |
| dataset manifest | 公开配置或文档 | 记录 dataset 名、upstream ref、license、digest 策略，不提交数据本体 |

特别注意：`harborrl/data/harbor/probe.py` 目前依赖旧 `TerminalEnv` 路径。若公开清理方案移除 `harborrl/environments/terminal/`，必须二选一：

1. 保留一个最小、无内部站点逻辑的 probe 兼容层；或
2. 先把 probe 改为直接使用 Native Harbor Runner / Docker 语义。

在完成替代前，Harbor Hub 兼容性是清理分支的阻断项。

### 8.3 数据集兼容清单

发布前建立 `docs/dataset_compatibility.md` 或 `configs/harbor_hub/manifests.yaml`，每个已适配 dataset 记录：

- dataset 名称；
- upstream repo / revision / branch；
- license 或使用限制；
- 下载方式；
- 本地 task root 约定；
- expected task count；
- task digest 策略；
- inspector 预期状态；
- 是否需要 Docker 网络、GPU 或特殊镜像；
- 已通过的 Native smoke；
- 负责人。

数据本体、私有缓存和内部路径不进入 Git。

### 8.4 兼容测试门禁

至少三层：

1. **Fixture CI**：仓库内置小型合成 task，验证 inspect / receipt / catalog / runner / exporter，不需要外部网络。
2. **本地矩阵**：对每个已适配 Harbor Hub dataset 指向外部 task root，运行 inspect 和 catalog validation。
3. **代表任务端到端**：每个 adapter 类别至少一个真实任务跑 Native rollout；发布前至少一个进入 GPU dry-run 或单步训练。

任何 dataset 在新分支从 `SUPPORTED` 变为 `UNSUPPORTED` 且无解释，都不能合入发布分支。

## 9. orphan branch 备用方案

### 9.1 触发条件

只有出现以下情况才使用 orphan branch：

1. `feat/harborrl-mvp` 到 `feat/native-mvp-clean` 的选择性移植无法拆干净。
2. refactor 分支与 clean 分支冲突严重，直接 merge 会持续引入无关文件。
3. 发布树需要完全重新初始化，而普通 `git rm` 提交历史会让 review 不可控。
4. 误将不希望公开的增量提交进了 feature 分支，且无法用安全 patch 重建。

在当前假设下，`main` 可信，因此 orphan 不是安全必需品，只是树切割工具。

### 9.2 使用方式

```bash
git switch --orphan orphan/public-v0.1
git rm -rf . 2>/dev/null || true
# 从 release/public-v0.1 或明确 allowlist 复制最终文件
git add <allowlist>
git commit -m "release: initialize public v0.1 tree"
``+
```

然后：

1. 与 `release/public-v0.1` 的 tracked file list 和内容 hash 对比，确认只保留预期文件。
2. 通过 CPU / Hub dataset / doctor / dry-run 测试。
3. 以普通 PR 将 orphan tree 合入可信 `main`。
4. 不需要改写 `main` 历史。

注意：orphan branch 不应直接从 `feat/harborrl-mvp` 创建后再整体 merge；那样容易把无父提交 tree 和 MVP 实验 tree 混在一起，失去切割意义。

## 10. 操作流程总览

### Step 1：保护现场

1. 记录 `feat/harborrl-mvp` HEAD 和 status。
2. 创建 accepted tag / archive branch。
3. 创建本地 bundle。
4. 将 ignored 的文本型实验内容显式加入 `archive/harborrl-local-experiments`。

### Step 2：建立干净 MVP 层

1. 从 `main` 创建 `feat/native-mvp-clean`。
2. 选择性移植 Native MVP 核心。
3. 移植并验证 Harbor Hub 兼容层。
4. 拆成小提交。
5. 通过 Native CPU tests 和 representative Hub dataset smoke。

### Step 3：重构 exporter

1. 从 `feat/native-mvp-clean` 创建 `refactor/slime-v032-exporter`。
2. 如果为加速开发，可在当前 MVP 上先做 dev 分支，但最终只移植受限 patch。
3. 完成 Slime v0.3.2 adapter 和测试。
4. 完成 GPU 有界验收。

### Step 4：整理公开树

1. 从 `feat/native-mvp-clean` 创建 `release/public-v0.1`。
2. 执行 public cleanup plan。
3. 保留 Harbor Hub downloader / inspector / catalog / runner 兼容能力。
4. 重写 README 和公开 docs。
5. 合入 `refactor/slime-v032-exporter`。

### Step 5：验收与同步

1. clean clone install。
2. CPU CI。
3. Harbor Hub dataset matrix。
4. doctor / dry-run。
5. GPU 单步验收。
6. package build。
7. 公开 hygiene scan。
8. push release branch。
9. PR 回 `main`。
10. merge 后打公开 tag。

## 11. 验收清单

发布分支合入 `main` 前必须确认：

- [ ] `main` 未被 force push 或历史改写。
- [ ] MVP accepted tag / archive branch 已保留。
- [ ] ignored 实验内容未被删除，且需要入库的部分已进入本地归档分支。
- [ ] `release/public-v0.1` 的 tracked tree 符合 public cleanup plan。
- [ ] 未直接 merge `feat/harborrl-mvp` 到公开分支。
- [ ] exporter refactor 已合入并通过 CPU / GPU 门禁。
- [ ] Slime / Megatron 版本由 doctor 校验。
- [ ] Harbor Hub 已适配 dataset 的 inspect / catalog / Native smoke 通过。
- [ ] 私有数据、模型 checkpoint、内部地址、个人路径和内部 registry 未出现在新增 diff。
- [ ] `harborrl` package install 和 CLI help 正常。
- [ ] README 只声明已验证能力。
- [ ] 公开 tag 只打在最终 merge 后的 `main`。

## 12. 决策结论

推荐采用 **“MVP 归档 + main 选择性移植 + clean 分支重构 + release 分支清理后合入 refactor”**。

当前 `feat/harborrl-mvp` 可以作为短期重构开发环境，但公开集成只接收受限 patch 或基于 `feat/native-mvp-clean` 的 refactor 分支。

orphan branch 保留为不可控情况下的备用切割工具；在 `main` 可信的前提下，不应默认使用。
