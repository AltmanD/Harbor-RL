# HarborRL 公开发布整理方案（现状修订版）

日期：2026-09-23（修订）  
适用对象：私有集成仓 `AltmanD/Harbor-RL` 及其 `main` / `feat/harborrl-mvp` / `refactor/slime-v032-exporter-dev` 分支  
配套文档：`docs/branch_management_plan_zh.md`（分支与迁移策略）、`docs/exporter_refactor_plan_zh.md`（exporter 技术设计）、`docs/exporter_v032_dev_status_zh.md`（dev 分支验收状态）  
结论：**采用“私有仓继续集成 + 新公开仓无历史发布”的双仓模型**。公开发布树以 Native MVP 核心与 Slime v0.3.2 零 patch adapter 为目标，经 Harbor Hub 兼容门与 GPU 验收门后，从 `release/public-v0.1` 的 tracked tree 提取白名单文件，在新公开仓库创建无父提交。

## 0. 执行摘要

1. `main`、`feat/harborrl-mvp` 和 refactor dev 分支的 **Git 历史均不可公开**：`benchmarks/`（10,337 个文件 / 139.6 MiB）和 SETA fixture 私钥早已进入 `main` 历史（引入提交 `6e5f9cb6`）。
2. 因此 `branch_management_plan_zh.md` 中“main 是可信基线”应理解为**私有集成基线**：继续受保护、不改写；公开发布则必须使用**新公开仓库 + 无历史根提交**。
3. 内容管线维持分支管理方案的缓冲层设计：`main → feat/native-mvp-clean → refactor/slime-v032-exporter → release/public-v0.1`；最后一跳把 release tree 提取到公开仓，而不是把私有 refs 推公开。
4. 公开 v0.1 的训练后端目标版本是 **Slime v0.3.2 官方组合 + HarborRL 薄 adapter（零 patch）**；refactor dev 分支已完成 CPU 契约层（91 项目标测试通过），GPU 单步验收仍是发布阻断项。
5. Harbor Hub 已适配数据集的 inspect / catalog / Native rollout 兼容能力是清理阻断项；`probe.py` 对 `environments/terminal` 的依赖必须在清理时给出替代，不允许静默删除。

## 1. 当前状态盘点（2026-09-23）

### 1.1 分支与规模

| 分支 / 引用 | 提交 | tracked 文件 | 大小 | 状态 |
| --- | --- | ---: | ---: | --- |
| `origin/main` | `dce4cfaa` | 13,128 | 182.1 MiB | 私有可信集成基线；历史含 benchmarks 与私钥 |
| `origin/feat/harborrl-mvp` | `999513bb` | 13,198 | 182.4 MiB | 已验收 MVP；相对 main 22 个提交 |
| 本地 `feat/harborrl-mvp` | `8af8a478` | — | — | 领先 origin 1 个提交，工作区有暂存/未跟踪内容，且被其他进程占用 |
| `refactor/slime-v032-exporter-dev` | `2a9b2352` | 13,216 | 182.5 MiB | 基于 `999513bb`；新增中性 exporter 与 v0.3.2 adapter；CPU 目标套件 91 passed |

MVP 树主要构成：`benchmarks/` 10,337 文件 / 139.59 MiB；`backends/`（vendored Slime + Megatron）2,466 文件 / 37.59 MiB；`harborrl/` 150 文件 / 1.48 MiB；`tools/` 87；`tests/` 48；`docs/` 33；`deploy/` 31。

### 1.2 refactor dev 分支相对 MVP 的增量

- 新增 `harborrl/export/`：backend-neutral `TrainingBatch` 契约、Native IR 导出与 fail-closed 校验，无 Slime/Torch/Megatron import。
- 新增 `harborrl/backends/slime_v032/`：rollout / converter / advantage / loss / postprocess / versions / launcher 七个模块，对应 Slime v0.3.2 五个官方扩展 hook。
- 接线：`training.backend_contract`（默认 `slime-legacy`）、`HARBORRL_NATIVE_SLIME_CONTRACT` 选择器、doctor 静态 hook 检查。
- 测试：`tests/harborrl/test_slime_v032_exporter.py`（12 项）与 shell dry-run 参数化断言。
- 未完成：官方镜像 feasibility spike、GPU 单步验收、custom loss 缩放核对、真实 rollout 集成、Harbor Hub matrix（详见 `docs/exporter_v032_dev_status_zh.md`）。

### 1.3 必须先处理的发布阻断

#### P0-1：敏感资产已在私有仓历史中

- `benchmarks/environments/seta_env/1198/ssh_keys/id_rsa` 为 OpenSSH 私钥，`main` 与 MVP 树中均存在（同目录还有 `id_rsa.pub`）。
- 多个验收文档包含内部 SSH alias、主机名、IP、个人用户名和集群域名。
- `deploy/ops`、`deploy/runtime`、`deploy/workers`、`examples`、`runs/rjob` 含内部代理、私有 registry、RJob 镜像、个人绝对路径和集群网络信息。
- 本地 `runs/` 约 168 GiB 未跟踪内容必须继续禁止入库。

处理原则：

1. **先轮换/作废暴露过的密钥与凭据**（benchmark fixture 私钥、集群 SSH、代理、registry token、API token），再谈公开发布。
2. 不把任何私有分支或 refs 推到公开远端。
3. 公开发布使用新仓库、新根提交，只复制白名单文件。
4. 发布前扫描公开仓的**全部 Git 对象**，而不只是工作区快照。

#### P0-2：vendored 第三方整仓

`backends/slime`（464 文件）与 `backends/Megatron-LM`（2,002 文件）是修改版整仓，公共用户难以判断 HarborRL 自有代码边界，第三方升级与安全修复也难以追踪。公开仓改为“官方 pinned 版本 + bootstrap 校验 +（仅在必要时）最小 patch”。

#### P0-3：文档与 README 状态矛盾

`README.md` / `README_zh.md` 仍有 “native 尚未验收” 的过期表述，与 `docs/README.md` 中 8×H200 全链路、reward 对比、checkpoint 重载验收记录冲突；同时存在 LightRL 旧命名、旧 logo、旧 citation 与临时 `cleanup_status.md` 链接。公开仓 README 必须重写。

#### P0-4：Harbor Hub 兼容是清理阻断项

以下模块在 MVP 树中均存在，公开清理时不允许无替代删除：`harborrl/tasks/cli.py`、`harborrl/data/harbor/{inspector,native_inspector,receipt,materializer,probe}.py`、`harborrl/data/download.py`。其中 `probe.py` import `harborrl.environments.terminal.runtime.TerminalEnv`；若公开树移除 `environments/terminal/`，必须二选一：

1. 保留一个无内部站点逻辑的最小 probe 兼容层；或
2. 先把 probe 改为直接使用 Native Harbor Runner / Docker 语义。

#### P0-5：共享工作区被占用

`/mnt/shared-storage-user/luyudong/Harbor-RL` 当前有其他进程使用且工作区不干净。所有发布整理操作必须在独立 clone / worktree 中进行（现有 `/mnt/shared-storage-user/luyudong/Harbor-RL-refactor` 可继续作为 refactor 开发环境），禁止在共享工作区执行 `git clean -fdx` 或切换分支。

## 2. 发布架构决策：双仓模型

### 2.1 为什么必须新公开仓

旧版方案与新分支方案的矛盾点在这里统一：

- 分支管理方案的“main 可信、不重写历史”适用于**私有集成仓**：main 继续作为受保护基线，只接收 reviewed PR。
- 清理方案的事实核查表明 **main 历史同样包含私钥与私有数据资产**，因此任何携带 main 父历史的分支都不能作为公开仓库的主线。
- 结论：公开仓必须新根提交；这不是备用方案，而是公开发布的必要出口。但**不需要也不应该改写私有 main**。

### 2.2 仓库角色

| 仓库 / 分支 | 角色 | 历史 | 可见性 |
| --- | --- | --- | --- |
| 私有仓 `main` | 长期集成基线 | 保留，不改写 | 私有 |
| 私有仓 `feat/harborrl-mvp` + `archive/*` + tag | MVP 验收与实验留档 | 保留 | 私有 / 本地 |
| 私有仓 `feat/native-mvp-clean` | 从 main 选择性移植的干净内容层 | 正常提交 | 私有 PR |
| 私有仓 `refactor/slime-v032-exporter` | 干净分支上的 v0.3.2 adapter | 正常提交 | 私有 PR |
| 私有仓 `release/public-v0.1` | 清理后的候选发布树 | 正常提交 | 私有 PR |
| **新公开仓 `public/v0.1`** | 最终公开发布 | **无父提交** | 公开 |

### 2.3 内容流水线

```text
私有仓:
  main
    └── feat/native-mvp-clean        # 选择性移植 MVP 核心 + Hub 兼容层
          ├── refactor/slime-v032-exporter   # 接收 dev 分支受限 patch
          └── release/public-v0.1            # 执行本清理方案，合入 refactor

公开发布:
  release/public-v0.1 --allowlist--> 新公开仓 orphan root --> PR --> v0.1.0-public tag
```

受限 patch 边界（来自 `branch_management_plan_zh.md` §5.3）：dev 分支差异只允许落在 `harborrl/export/`、`harborrl/backends/`、launcher、配置、测试与文档内；不直接 merge `refactor/slime-v032-exporter-dev`。

## 3. 公开 v0.1 目标树

### 3.1 目录骨架

```text
├── LICENSE / NOTICE.md / CONTRIBUTING.md / SECURITY.md
├── README.md / README_zh.md
├── pyproject.toml
├── .gitignore
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── quickstart.md
│   ├── configuration.md
│   ├── backend-integration.md
│   ├── task-format.md
│   ├── dataset-compatibility.md
│   └── release-scope.md
├── examples/native/
│   ├── cpu_contract_smoke.sh
│   ├── train_qwen_native.yaml
│   ├── worker_setup.md
│   └── tasks/hello_world/
├── configs/harbor_hub/manifests.yaml      # 只含元数据，不含数据本体
├── scripts/
│   ├── bootstrap_backends.sh
│   ├── sglang_semantic_probe.py           # 若仍是发布门禁
│   └── audit_public_tree.sh
├── harborrl/
│   ├── cli.py
│   ├── config/native.py
│   ├── gateway/
│   ├── data/harbor/                       # inspector / native_inspector / receipt / materializer / probe
│   ├── data/download.py
│   ├── tasks/cli.py
│   ├── platform/native_train.py
│   ├── platform/slime_train*.sh            # native-only 重写
│   ├── rollout/native_generate.py
│   ├── rollout/harbor_job/
│   ├── export/                             # 中性 TrainingBatch 契约（来自 refactor 分支）
│   ├── backends/slime_v032/                # Slime v0.3.2 官方 hook adapter
│   └── trajectories/native.py
└── tests/harborrl/
    ├── test_native_contracts.py
    ├── test_native_gateway.py
    ├── test_native_lifecycle.py
    ├── test_native_training.py
    └── test_slime_v032_exporter.py
```

### 3.2 `harborrl` 包收敛原则

| 现有模块 | 处理 | 原因 |
| --- | --- | --- |
| `config/native.py` | 保留，入口整理 | Native schema 2 与 `backend_contract` 是首版核心 |
| `gateway/` | 保留 | Messages Gateway、SGLang backend、token/logprob 审计 |
| `trajectories/native.py` | 保留 | reward、identity、IR、原子发布契约 |
| `rollout/native_generate.py`、`rollout/harbor_job/` | 保留并审计依赖 | Native rollout 与 Harbor Runner 主链路 |
| `export/`（新） | 保留 | backend-neutral `TrainingBatch`，纯 Python 可 CPU 测试 |
| `backends/slime_v032/`（新） | 保留 | 零 patch 官方 hook adapter；GPU 验收门通过后为默认 |
| `rollout/exporters/native.py` | 保留为中性导出的兼容薄层或并入 `export/` | 数值语义与 golden 测试基准 |
| `rollout/exporters/native_slime.py`（legacy patched 路径） | **不进公开 v0.1**（GPU 门通过后）；迁移期内只留在私有分支 | 依赖 vendored patch 与私有张量 key |
| `data/harbor/{inspector,native_inspector,receipt}.py` | 保留 | task/receipt 校验与 Hub 兼容 |
| `data/harbor/{materializer,probe}.py`、`data/download.py`、`tasks/cli.py` | 保留或提供等价替代 | Harbor Hub 兼容阻断项 |
| `environments/terminal/` | 默认移除；若 probe 未改造则保留最小无内部站点层 | 旧 TerminalEnv 栈不属 Native MVP |
| `platform/worker_*`、`router*`、`run_leases` | 移除或另建内部包 | 旧远程环境池与站点运维 |
| `environments/`、`harnesses/`、`algorithms/`、`data/convert_*`、旧 `rollout/` interactive 栈 | 移除 | SETA/AgentHarm/Tau2/SWE/DIVE-PO/LWM 等研究与旧栈 |
| `misc/` | 逐项检查 | 只保留 Native smoke 必需函数 |

迁移规则：先建 import 边界，再按调用图搬迁；禁止整目录复制后再删。

## 4. 逐目录处置方案

### 4.1 `benchmarks/`（10,337 文件 / 139.6 MiB）

**整体不进公开仓。** 替代：

1. `examples/native/tasks/hello_world/` 提供可离线运行、无真实凭据的最小任务。
2. Harbor Hub / task manifest 提供外部下载命令，不提交数据快照。
3. 每个外部 benchmark 单列 upstream、license、下载方式、hash 与安全边界（进 `docs/dataset-compatibility.md` 或 `configs/harbor_hub/manifests.yaml`）。
4. 私有任务继续留在内部存储 / 私有数据仓，通过显式 task root 注入。

### 4.2 `backends/`（2,466 文件 / 37.6 MiB）

**不 vendored。** 公开仓后端策略：

1. 基准版本：Slime `v0.3.2`（commit `3778dbf6d1a533ab478ecf5ddaa11449a47752b2`）、Megatron `1dcf0dafa884ad52ffb243625717a3471643e087`、SGLang `v0.5.15.post1-cu129`。
2. `scripts/bootstrap_backends.sh --check` 校验版本；默认目标是**零 patch**。
3. 只有 GPU 验收证明官方 hook 无法表达训练语义时，才降级为最小 patch，并把 patch 与 hash 纳入门禁（见 §5 决策门）。
4. 运行时只接受 `SLIME_DIR` / `MEGATRON_DIR` 显式路径；无后端时 fail-closed 且报错清晰。

### 4.3 `docs/`

从 33 个文件收敛到约 7 个公开文档：`quickstart`、`architecture`、`configuration`、`backend-integration`、`task-format`、`dataset-compatibility`、`release-scope`。内部验收、逐日记录、主机拓扑、SSH 复核、清理过程文档全部留在私有分支。

### 4.4 `deploy/`、`examples/`、`tools/`

- `deploy/`：整体移除；`worker_setup.md` 用无内部默认值的公开文档替代。
- `examples/`：只保留 `examples/native/`；旧模型、旧环境、LWM/RJob 示例移除。
- `tools/`：历史评测/分析/开发工具移除；`tools/verification/sglang_semantic_probe.py` 若仍是门禁则迁至 `scripts/` 并审计路径。

### 4.5 `tests/`

核心集合：`test_native_contracts.py`、`test_native_gateway.py`、`test_native_lifecycle.py`、`test_native_training.py`、`test_slime_v032_exporter.py`。需补齐：最小任务 inspector/digest/receipt、clean install + CLI help、bootstrap 版本校验、public hygiene（禁内部域名/个人路径/私钥/大二进制）。删除依赖旧环境、LWM、SPEAR、Agent57、RJob、internal worker 的测试。

### 4.6 根目录与元数据

保留并更新：README（双语）、LICENSE、NOTICE、CONTRIBUTING、SECURITY、pyproject、.gitignore、CI workflow。移除：`legacy-requirements.txt`、`sitecustomize.py`、`runs/rjob/`、旧 logo 资产、本地 probe 脚本。`pyproject.toml`：只打包 `harborrl*`；删除 `lightrl-eval` 兼容入口；extras 分 `native` / `gateway` / `train` / `dev`；基础安装不拉 Torch/Megatron 全家桶。

## 5. Exporter 决策门（发布前必须裁决）

| 状态 | 条件 | 公开 v0.1 动作 |
| --- | --- | --- |
| **目标路径** | v0.3.2 官方镜像 spike + GPU 单步验收全部通过，零 patch 成立 | 公开树只含 `export/` + `backends/slime_v032/`；不含 legacy exporter 与 Slime patch |
| **降级路径** | GPU 验收证明必须最小 patch | 公开树保留 v032 adapter + `patches/slime/*.patch` + hash 门禁；patch 范围与理由写入 `backend-integration.md` |
| **阻塞路径** | GPU 验收未完成或 loss 缩放未确认 | **推迟公开发布**；不把未验证的 legacy patched 路径当作公开默认能力发布 |

GPU 验收最低要求（沿用 `exporter_refactor_plan_zh.md` Phase 6）：官方组合启动、0/1 reward contrast、rollout logprob 与 Gateway 审计一致、非零 advantage/gradient/PG loss、显式权重变化、checkpoint 保存重载、新 policy version 生效、关键指标与已验收 MVP 误差在允许范围。

## 6. Harbor Hub 数据兼容门

“已适配数据集仍可用”的定义（发布阻断）：

1. 外部 task root / Hub 下载 / 本地缓存三种来源可用。
2. `harborrl hub inspect`（或等价 CLI）可检查本地 task。
3. inspector 识别已适配 task profile；catalog 记录 dataset、revision、path、digest。
4. Native runner 按 catalog 执行；reward receipt 进入 Native IR。
5. exporter 重构后 batch 保持相同 group / trajectory / turn 语义。
6. 公开仓不含私有数据本体与个人绝对路径。

发布前建立 `docs/dataset-compatibility.md` 或 `configs/harbor_hub/manifests.yaml`：dataset 名、upstream repo/revision/branch、license、下载方式、task root 约定、expected task count、digest 策略、inspector 预期、Docker/GPU 需求、已通过的 Native smoke、负责人。任何 dataset 从 `SUPPORTED` 变 `UNSUPPORTED` 且无解释，不得合入 release。

## 7. 实施步骤（现状版）

### Phase 0：冻结与保护（0.5 天）

1. 在独立 clone 中操作；不占用共享 MVP 工作区。
2. 以 `origin/feat/harborrl-mvp`（`999513bb`）为验收固化点：打 `mvp/native-v0.1-accepted` tag、建 `archive/harborrl-mvp-accepted`、生成持久目录 bundle。
3. 将需要入库的 ignored 文本实验记录显式加入 `archive/harborrl-local-experiments`（先列 allowlist，排除 cache/`node_modules`/`.tools`/大数据）。
4. 轮换/作废暴露凭据；确定 public release owner 与公开仓库名。

### Phase 1：干净内容层（1–2 天）

1. 从 `main` 建 `feat/native-mvp-clean`。
2. 按 §3 白名单从 MVP 选择性移植 Native 核心、测试与 Hub 兼容层；按小提交拆分。
3. 处理 `probe.py` → TerminalEnv 依赖（最小兼容层或 Native Runner 改造）。
4. 通过 Native CPU 测试与代表性 Hub dataset smoke。

### Phase 2：合入 exporter 重构（1 天 + GPU 等待）

1. 从 `feat/native-mvp-clean` 建 `refactor/slime-v032-exporter`。
2. 将 `refactor/slime-v032-exporter-dev` 的差异以受限 patch 应用（边界见 §2.3）。
3. 在干净分支重跑 CPU 套件；完成官方镜像 spike 与 GPU 单步验收。
4. 按 §5 决策门确定零 patch / 最小 patch / 阻塞。

### Phase 3：候选发布树（1–2 天）

1. 从 `feat/native-mvp-clean` 建 `release/public-v0.1`，执行本方案清理。
2. `git rm` 移除 benchmarks、vendored backends、deploy、旧研究模块与内部运维脚本。
3. 制作 hello-world 任务、native train YAML、worker setup 文档。
4. 合入 `refactor/slime-v032-exporter`。
5. 通过 Hub dataset matrix（inspect + catalog + 代表任务 Native smoke）。

### Phase 4：新公开仓初始化（0.5 天）

```bash
# 在新公开仓库中，不携带任何私有 refs
git switch --orphan public-v0.1
git rm -rf . 2>/dev/null || true
# 从 release/public-v0.1 复制白名单文件
git add <allowlist>
git commit -m "release: initialize HarborRL native MVP"
```

要求：无 `main` / MVP 父历史；首提交即最终结构；不复用私有远端 refs；如必须留在原仓库则需专业 history rewrite 工具全量清 refs（优先级低于新仓库）。

### Phase 5：文档与 CI（1 天）

1. 重写双语 README（§8 大纲）与 7 个核心文档。
2. GitHub Actions CPU matrix；package build、ruff、pytest、shell syntax、public hygiene。
3. optional GPU workflow 或发布前手工 checklist。

### Phase 6：发布审计（0.5–1 天）

至少执行：

```bash
git rev-list --all --count
git log --oneline --decorate --graph --all
git fsck --full --no-dangling
git grep -I -n -E 'luyudong|puyuan|pjlab|kubebrain|registry\.h\.pjlab|/mnt/shared-storage-user' $(git rev-list --all)
python -m pytest -q tests/harborrl
python -m harborrl.cli train --config examples/native/train_qwen_native.yaml --dry-run
bash scripts/bootstrap_backends.sh --check
bash scripts/audit_public_tree.sh
python -m build
```

另需可信 secret scanner 扫描全部历史对象。

## 8. README 重写大纲

双语同构：What is HarborRL（三句话）→ Status（已验证边界与未承诺项）→ Architecture（Config → Doctor → Gateway → Harbor Runner → Reward/IR → Neutral Export → Slime v0.3.2 → Checkpoint）→ Install（基础 + extras + backend bootstrap）→ Quickstart（hello-world CPU smoke）→ GPU Training（dry-run 前置、外部 worker、pinned backend）→ Repository map（≤20 行）→ Extension points（task / gateway backend / training backend）→ Testing → Acknowledgement / Citation / License。

必须修正：`cleanup_status.md` 临时链接、“native 尚未验收”过期陈述、LightRL 旧命名/旧 logo/旧 citation、backend 前置条件缺失。

## 9. 验收标准

公开仓同时满足：

1. **历史干净**：新根提交；secret scanner 无 high/critical；内部域名、IP、SSH alias、个人路径、私钥不在任何历史对象。
2. **结构简洁**：tracked files < 300；主分支 < 20 MiB；无大数据、checkpoint、环境快照、第三方整仓。
3. **安装可用**：clean clone 后 `pip install -e .[dev]` 与 `harborrl --help` 成功。
4. **CPU 可测**：Native contract/gateway/lifecycle/training + slime_v032 exporter 测试在 CI 通过。
5. **Smoke 可跑**：hello-world CPU contract smoke 无内部网络。
6. **Dry-run 可解释**：GPU train dry-run 输出完整命令、backend 版本、worker 配置与缺失项。
7. **后端可复现**：bootstrap pin 官方版本；若降级 patch，hash 校验通过。
8. **Exporter 门**：§5 决策门有明确结论；默认入口唯一。
9. **Hub 兼容**：§6 matrix 通过；`SUPPORTED` 集合无未解释缩减。
10. **文档一致**：README、quickstart、架构图、release scope 无失效链接与矛盾状态。
11. **许可清晰**：HarborRL MIT；第三方依赖与 patch 的 license/notice 明确。

## 10. 发布切分

- **v0.1.0-public**：Native MVP framework + 零 patch（或门禁过的最小 patch）Slime v0.3.2 adapter + hello-world smoke + Qwen GPU dry-run 配置 + 核心 CI。
- **v0.2.0**：checkpoint/resume 文档、多任务 manifest、backend 升级测试、GPU smoke workflow。
- **v0.3.0**：独立 benchmark adapters、更多 harness/gateway backend、任务下载与缓存管理。

研究算法、内部部署、长周期实验记录继续留在私有分支。

## 11. 明确不做的事情

1. 不把 `main`、MVP 或任何私有 refs 推到公开远端。
2. 不在私有分支上 `git rm` 后直接公开（历史仍含敏感对象）。
3. 不把私钥改成占位文件后沿用同一历史。
4. 不用 README 链接被 Git 忽略的本地验收文档。
5. 不把内部集群默认值作为公共 fallback。
6. 不把未验证的多环境/多算法包装成稳定能力。
7. 不为“功能全”保留 1 万+ 任务资产与两个第三方整仓。
8. 不在 GPU/Hub 门未过时抢发公开版本。
