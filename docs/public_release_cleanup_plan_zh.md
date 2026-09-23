# HarborRL 公开发布整理方案

日期：2026-09-23  
分支：`feat/harborrl-mvp`  
结论：**当前分支和完整 Git 历史不能直接推送到公开 GitHub**。应先保留私有研究分支，再以白名单方式创建无敏感历史的新发布分支，把仓库收敛为 Native MVP 的核心框架、最小示例、后端适配说明和可复现测试。

## 1. 发布目标

### 1.1 公共版本定位

公开发布的 HarborRL 应该是一个结构清晰、边界明确、便于从业者学习和复用的智能体 RL 训练框架，而不是内部实验仓库的压缩版。

保留的主线能力：

1. **Native agentic RL 训练链路**：配置校验、doctor、Messages Gateway、Harbor task rollout、reward/trajectory 契约、GRPO/Slime 导出和单步训练闭环。
2. **可复现实验入口**：一个最小公开任务、一个 CPU contract smoke、一个可选 GPU 单步训练示例。
3. **清晰扩展点**：任务、harness、gateway backend、training backend、reward/trajectory schema 的替换方式。
4. **可测试性**：公开 CI 能在无 GPU/无内部服务的情况下验证核心契约；GPU 检查作为可选门禁。

不随首版公开的内容：

- 多套内部 benchmark 数据和环境快照。
- DIVE-PO、LWM、SPEAR、Agent57 等研究分支和历史算法实验。
- SETA/AgentHarm/Agent-SafetyBench/Tau2/SWE 全家桶适配。
- 内部集群 RJob、代理、节点修复和运维记录。
- 逐日验收报告、主机拓扑、IP、SSH alias、个人路径和实验日志。

### 1.2 非目标

首次公开版本不应声称支持当前私有分支中出现过的所有环境、算法和部署形态。README 中应明确：

- 首版只承诺 Native MVP 及其测试过的最小链路。
- benchmark 结果和环境扩展属于后续 release 或独立 package。
- 多轮稳定训练、故障恢复和集群运维不是首版默认能力。

## 2. 当前仓库体检

### 2.1 规模与结构

当前跟踪对象约 **13,197 个文件 / 182.4 MiB**，其中主要分布如下：

| 目录 | 文件数 | 大小 | 判断 |
| --- | ---: | ---: | --- |
| `benchmarks/` | 10,337 | 139.6 MiB | 私有任务资产和数据快照，不应公开 |
| `backends/` | 2,466 | 37.6 MiB | vendored Slime/Megatron 第三方整仓，应外部化 |
| `harborrl/` | 150 | 1.5 MiB | 混合 Native MVP、旧 interactive 栈、算法实验和 worker 系统 |
| `tools/` | 87 | 0.5 MiB | 历史评测、分析和站点诊断工具 |
| `tests/` | 48 | 0.4 MiB | 只有 Native 相关测试是首版核心 |
| `docs/` | 32 | 1.3 MiB | 混有论文草稿、内部验收、历史记录和公开文档 |
| `deploy/` | 31 | 0.4 MiB | 大量内部代理、节点和 RJob 运维细节 |
| `examples/` | 24 | 0.1 MiB | 旧模型、旧环境、LWM/RJob 示例占多数 |

当前分支相对 `origin/main` 有 21 个提交，并且继承了主线中的大量实验代码、benchmark 资产和第三方后端。仅在当前分支上做删除提交，不能移除 Git 历史中的内容。

### 2.2 必须先处理的发布阻断

#### P0-1：敏感资产已经在 Git 历史中

检查发现以下类型的内容会随历史公开：

- `benchmarks/environments/seta_env/1198/ssh_keys/id_rsa` 是 OpenSSH 私钥文件。
- 多个 Native MVP 验收文档包含内部 SSH alias、主机名、IP、个人用户名和集群域名。
- `deploy/ops`、`deploy/runtime`、`deploy/workers`、`examples`、`runs/rjob` 中存在内部代理、私有 registry、RJob 镜像、个人绝对路径和集群网络信息。
- 本地 `runs/` 目录约 168 GiB，虽大多未跟踪，但包含大量 trajectory、日志、模型和路径信息，必须继续禁止进入发布历史。

处理原则：

1. **先轮换/作废暴露过的密钥和凭据**，包括 benchmark fixture 私钥、集群 SSH 凭据、代理凭据、registry token 和 API token。
2. **不要把 `feat/harborrl-mvp` 直接推到公开远端**。
3. 使用新 public 分支或新仓库，从无历史根提交开始；只复制白名单文件。
4. 发布前扫描全部 Git 对象，而不是只扫描工作区当前文件。

#### P0-2：vendored 第三方后端带来维护和合规噪音

`backends/slime` 是 464 个文件、约 27 MiB 的修改版整仓；`backends/Megatron-LM` 是 2,002 个文件、约 37.6 MiB 的整仓。它们还带着自己的 CI、issue 模板、测试资产和发布流程。

这会造成：

- 公共用户难以判断 HarborRL 自有代码边界。
- 第三方升级和安全修复难以追踪。
- 发布仓库体积和审计成本偏高。
- HarborRL patch 与上游行为差异被整仓噪音淹没。

应改为“外部 pinned backend + HarborRL patch/bootstrap”的模式。

#### P0-3：文档状态互相矛盾

当前 `README.md` 和 `README_zh.md` 开头仍写着 “native 尚未验收”，而 `docs/README.md` 和 Native MVP 结果文档已经声明 8×H200 全链路、reward 对比、checkpoint 重载和显式权重变化验收完成。首段还指向 `docs/cleanup_status.md`，不是正常开源首页入口。

公开前必须重写 README，统一状态为：Native MVP 单步/有界训练链路已验证，但不承诺长期稳定训练和多 benchmark 复现。

## 3. 目标代码结构

### 3.1 首版目标树

```text
HarborRL/
├── README.md
├── README_zh.md
├── LICENSE
├── NOTICE.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── .gitignore
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── quickstart.md
│   ├── configuration.md
│   ├── backend-integration.md
│   ├── task-format.md
│   └── release-scope.md
├── examples/native/
│   ├── cpu_contract_smoke.sh
│   ├── train_qwen_native.yaml
│   ├── worker_setup.md
│   └── tasks/hello_world/
├── patches/
│   └── slime/harborrl-native-mvp.patch
├── scripts/
│   ├── bootstrap_backends.sh
│   └── audit_public_tree.sh
├── harborrl/
│   ├── cli.py
│   ├── config/native.py
│   ├── gateway/
│   ├── data/harbor/
│   ├── platform/native_train.py
│   ├── platform/slime_train_native.sh
│   ├── rollout/native_generate.py
│   ├── rollout/harbor_job/
│   ├── rollout/exporters/native.py
│   ├── rollout/exporters/native_slime.py
│   └── trajectories/native.py
└── tests/harborrl/
    ├── test_native_contracts.py
    ├── test_native_gateway.py
    ├── test_native_lifecycle.py
    └── test_native_training.py
```

### 3.2 `harborrl` 包收敛原则

| 现有模块 | 处理 | 原因 |
| --- | --- | --- |
| `config/native.py` | 保留并重命名入口整理 | Native schema 2 是首版核心 |
| `gateway/` | 保留 | Messages Gateway、SGLang backend 和 token/logprob 审计属于主链路 |
| `trajectories/native.py` | 保留 | reward、identity、IR 和原子发布契约属于主链路 |
| `rollout/native_generate.py` | 保留 | Native rollout group 调度核心 |
| `rollout/harbor_job/` | 保留并审计依赖 | Harbor Runner、collector、audit、offline smoke 属于主链路 |
| `rollout/exporters/native.py`、`native_slime.py` | 保留 | GRPO 训练样本导出边界 |
| `data/harbor/{inspector,native_inspector,receipt}.py` | 保留 | task/receipt 校验 |
| `platform/native_train.py` | 保留并拆掉硬编码后端路径 | doctor/launcher 核心 |
| `platform/slime_train.sh` | 重写为 native-only launcher | 当前混合 interactive/native 分支，难以维护 |
| `platform/worker_*`、`router*`、`run_leases` | 移除或另建内部包 | 属于旧远程环境池和站点运维，不是 Native MVP 必需 |
| `environments/` | 移除 | 旧 SETA/AgentHarm/Tau2/SWE runtime 栈 |
| `harnesses/` | 移除，保留 Native Runner 的外部 CLI 约定 | 旧 Camel/Claude adapter 混合大量可选依赖 |
| `algorithms/` | 移除 | DIVE-PO/LWM/PRM/Agent57 研究分支 |
| `data/convert_*` | 移除 | 私有 benchmark 转换器 |
| `rollout/{entrypoint,generate_steps,runner,admission,...}` | 移除 | 旧 interactive rollout 栈 |
| `tasks/cli.py`、`misc/` | 逐项检查，只保留 Native smoke 需要的函数 | 避免为了一个工具拖入旧体系 |

移动代码时不要整目录复制后再删；先建立目标 import 边界，再按调用图迁移，最后用测试和 `ruff` 清理 unused import。

## 4. 逐目录处置方案

### 4.1 `benchmarks/`

**处置：整体移出公共仓库。**

原因：

- 10,048 个 SETA environment 文件和大量 JSONL 混合了任务、数据、fixture、密钥和历史转换产物。
- 公共仓库会暴露数据授权、任务安全和内部资产边界问题。
- 学习者只需要一个最小任务格式示例和外部任务下载说明。

替代方案：

1. 增加 `examples/native/tasks/hello_world/`，必须是可公开、可离线运行、无真实凭据的最小任务。
2. 提供 Harbor Hub/task manifest 的下载命令，不在仓库中提交数据快照。
3. 对每个外部 benchmark 建立独立说明，列 upstream、license、下载方式、hash 和安全边界。
4. 私有任务保留在内部存储或私有数据仓库，通过 `HARBORRL_TASK_ROOT` 注入。

### 4.2 `backends/`

**处置：不 vendored，改为外部依赖和 patch。**

建议步骤：

1. 以当前验证过的 Slime upstream commit 为 pin 基准。
2. 从当前 vendored 树中提取 HarborRL 必需差异，形成 `patches/slime/harborrl-native-mvp.patch`。
3. `scripts/bootstrap_backends.sh` 支持：
   - `--slime-ref <commit>`
   - `--megatron-ref <commit>`
   - `--apply-patch`
   - `--check`
4. 运行时只接受 `SLIME_DIR` / `MEGATRON_DIR`，doctor 中检查版本和 patch 状态。
5. README 不再声称 “bundles Slime/Megatron”，改为 “integrates with pinned Slime/Megatron”。

如果最小 patch 仍然覆盖大量文件，短期替代方案是维护一个只含 HarborRL 变更的私有 fork，公共仓库 pin fork commit；不要继续复制第三方整仓。

### 4.3 `docs/`

**处置：从 32 个文件收敛到 6 个左右的公开文档。**

保留/重写：

| 文档 | 内容 |
| --- | --- |
| `quickstart.md` | 安装、backend bootstrap、任务准备、CPU smoke、GPU dry-run、训练入口 |
| `architecture.md` | Native rollout、Gateway、Harbor Runner、Slime/Megatron 边界和数据流 |
| `configuration.md` | schema 2、必填项、路径、worker、模型、安全和 dry-run |
| `backend-integration.md` | pinned commit、patch、环境变量、doctor 和升级边界 |
| `task-format.md` | task manifest、digest、receipt、reward profile 和最小 hello-world |
| `release-scope.md` | 已验证/未验证能力，避免过度宣传 |

移除或保留在私有 archive：

- `docs/algorithms/`
- `docs/evaluation/`
- `docs/performance/`
- `docs/operations/`
- `docs/native_mvp_*_zh.md`
- `docs/native_cpu_development_zh.md`
- `docs/records/`
- 所有内部 Go/No-Go、主机名、IP、日志和验收证据。

内部验收可压缩为公开 release note 中的一句话级证据，不包含环境标识和路径。

### 4.4 `examples/` 与 `configs/`

**处置：只保留 Native MVP 一条学习路径。**

保留：

- `examples/native/cpu_contract_smoke.sh`
- `examples/native/train_qwen_native.yaml`
- `examples/native/worker_setup.md`
- 最小公开任务。

移除：

- GLM/Qwen 多套旧 rollout 配方。
- Math、SPEAR、ALFWorld、LWM、RJob、SETA 评测脚本。
- `configs/rollout/` 历史模型模板。
- `configs/train_*_smoke.yaml` 中旧的 `lightrl_interactive` 入口。

配置命名统一为 `native_*`，避免 `lightrl_interactive` 与 HarborRL Native MVP 混淆。

### 4.5 `deploy/`

**处置：删除站点运维树，重写一页通用 worker setup。**

移除：

- `deploy/ops/`
- `deploy/runtime/`
- `deploy/archive/`
- 现有 `deploy/workers/` 的内部 Docker pool/watchdog 体系。

新增 `examples/native/worker_setup.md` 只描述：

- 前置依赖：Python、Docker、Harbor Runner、Claude Code CLI。
- 网络和安全要求。
- 环境变量注入。
- 健康检查命令。
- 常见故障的 fail-closed 行为。

不提供内部代理默认值、节点修复脚本、私有 registry、SSH alias 或 root 操作脚本。

### 4.6 `tools/`

**处置：删除历史工具，只保留必要验证工具或迁入 `scripts/`。**

- `tools/evaluation/`、`tools/analysis/`、`tools/dev/` 全部移除。
- `tools/verification/sglang_semantic_probe.py` 若仍是发布门禁，迁到 `scripts/sglang_semantic_probe.py`，并确保无内部路径。
- 其他工具由内部研究分支保留。

### 4.7 `tests/`

**处置：测试按公开能力重建，而不是保留 48 个历史测试。**

首版核心集合：

- `test_native_contracts.py`
- `test_native_gateway.py`
- `test_native_lifecycle.py`
- `test_native_training.py`

需要补齐：

- 最小任务 inspector/digest/receipt 测试。
- clean install 后 import 和 CLI help 测试。
- backend bootstrap patch 的 hash/dry-run 测试。
- public hygiene 测试：禁止内部域名、个人路径、私钥文件和超大二进制进入 Git。

删除所有依赖旧环境、旧评测、LWM、SPEAR、Agent57、RJob 和 internal worker 的测试。

### 4.8 根目录与元数据

保留并更新：

- `README.md`、`README_zh.md`
- `LICENSE`
- `NOTICE.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `pyproject.toml`
- `.gitignore`
- `.github/workflows/ci.yml`

移除：

- `legacy-requirements.txt`
- `sitecustomize.py`
- `runs/rjob/`
- `assets/lightrl_logo*`
- `probe.sh`、本地 scripts 和 update probe。

`pyproject.toml` 调整：

1. 只打包 `harborrl*`，不再打包 `tools*`。
2. 删除 `lightrl-eval` 兼容入口。
3. optional extras 分为 `native`、`gateway`、`train`、`dev`。
4. 锁定可运行的 Python 和核心依赖范围，不在基础安装中拉入 Torch/Megatron 全家桶。

## 5. README 重写大纲

英文和中文 README 应保持同一信息结构：

1. **What is HarborRL**：三句话说明 Native agentic RL MVP 的范围。
2. **Status**：明确已验证的单步/有界训练链路和未承诺的能力。
3. **Architecture diagram**：Config → Doctor → Gateway → Harbor Runner → Reward/IR → Slime GRPO → Checkpoint。
4. **Install**：基础库、optional extras、backend bootstrap。
5. **Quickstart**：hello-world CPU contract smoke。
6. **GPU training**：dry-run 前置条件、外部 worker、pinned backend、最小配置。
7. **Repository map**：不超过 20 行。
8. **Extension points**：task、gateway backend、training backend。
9. **Testing**：CPU CI 和 optional GPU gate。
10. **Acknowledgement / Citation / License**。

必须修正：

- 首行指向 `cleanup_status.md` 的临时说明。
- “native 尚未验收”的过期陈述。
- LightRL 命名、旧 logo、旧 citation URL 与 HarborRL 身份不一致的问题。
- 未说明 backend 前置条件和资源边界的问题。

## 6. 实施步骤

### Phase 0：冻结与保护（0.5 天）

1. 确认当前 `feat/harborrl-mvp` HEAD 和测试状态，不再向该分支加入无关实验。
2. 为当前分支建立私有 tag 和本地 bundle。
3. 轮换/作废所有可能进入历史的凭据和私钥。
4. 明确 public release owner 和 GitHub 目标仓库。

### Phase 1：创建干净历史（0.5 天）

建议新建发布仓库，或在新仓库中执行：

```bash
git switch --orphan public-release-v0
git rm -rf .
# 只复制白名单文件；不复制 .git、runs、benchmarks、backends、deploy 内部资产
git add <allowlist>
git commit -m "release: initialize HarborRL native MVP"
```

要求：

- 新分支没有 `origin/main` 和 `feat/harborrl-mvp` 的父历史。
- 首个提交即最终结构，避免先复制敏感文件再删除。
- 不复用当前远端的 refs。
- 若必须使用原仓库，应使用专业 history rewrite 工具并强制清理所有 refs；优先级低于新仓库方案。

### Phase 2：抽取 Native MVP 核心（1–2 天）

1. 建立新的 `harborrl` 包骨架。
2. 迁移配置、Gateway、trajectory、task inspector、runner/coordinator/audit/exporter。
3. 简化 CLI：`inspect`、`doctor`、`train`、`offline-smoke`。
4. 重写 native-only Slime launcher，删除旧 interactive 配置分支。
5. 用 import graph 检查，确保核心包不依赖 `environments`、`harnesses`、`algorithms`、`tools`。

### Phase 3：外部化训练后端（1–2 天）

1. 记录当前验证使用的 Slime/Megatron commit 和运行环境。
2. 提取并最小化 Slime patch。
3. 实现 bootstrap/check 脚本。
4. 修改 doctor 和 launcher 的路径解析。
5. 在无 backend 的基础安装下，错误信息必须清晰 fail-closed。

### Phase 4：最小任务与示例（1 天）

1. 制作公开 hello-world task。
2. 校验 task digest、reward 0/1、IR 和 offline smoke。
3. 提供 native train YAML 和 worker setup 文档。
4. 不提交模型 checkpoint、私有镜像名、内部代理或个人路径。

### Phase 5：文档与 CI（1 天）

1. 重写 README 和 6 个核心文档。
2. 增加 GitHub Actions CPU matrix。
3. 增加 package build、ruff、pytest、shell syntax、public hygiene 检查。
4. 增加 optional GPU workflow 或发布前手工 checklist。

### Phase 6：发布审计（0.5–1 天）

审计命令至少包括：

```bash
git rev-list --all --count
git log --oneline --decorate --graph --all
git fsck --full --no-dangling
git grep -I -n -E 'luyudong|puyuan|pjlab|kubebrain|registry\\.h\\.pjlab|/mnt/shared-storage-user' $(git rev-list --all)
python -m pytest -q tests/harborrl
python -m harborrl.cli train --config examples/native/train_qwen_native.yaml --dry-run
bash scripts/bootstrap_backends.sh --check
bash scripts/audit_public_tree.sh
python -m build
```

另需运行一个可信 secret scanner，并检查所有历史对象，而不是只检查工作区。

## 7. 验收标准

发布分支必须同时满足：

1. **历史干净**：新根提交，无旧研究对象；secret scanner 无 high/critical；内部域名、IP、SSH alias、个人路径和私钥不在任何历史对象中。
2. **结构简洁**：建议 tracked files 小于 300，仓库主分支小于 20 MiB；除 logo和文档示意图外无大体积数据、checkpoint、环境快照或第三方整仓。
3. **安装可用**：clean clone 后 `pip install -e .[dev]` 成功，`harborrl --help` 成功。
4. **CPU 可测**：Native contract/gateway/lifecycle/training 测试在 GitHub Actions 通过。
5. **Smoke 可跑**：hello-world CPU offline/contract smoke 不需要内部网络。
6. **Dry-run 可解释**：GPU train dry-run 输出完整命令、backend 路径、worker 配置和缺失项。
7. **后端可复现**：bootstrap 脚本能 pin backend commit 并校验 patch hash。
8. **文档一致**：README 状态、quickstart、架构图和 release scope 一致，无失效链接。
9. **无过度声明**：不声称多 benchmark、多轮稳定训练或内部集群生产可用。
10. **许可清晰**：HarborRL MIT、第三方依赖和 patch 的 license/notice 明确。

## 8. 建议的发布切分

### v0.1.0-public：Native MVP framework

- Native schema、doctor、Gateway、task rollout、IR/reward、Slime GRPO exporter。
- hello-world CPU smoke。
- Qwen Native GPU dry-run 和已验证单步配置。
- 核心测试与 CI。

### v0.2.0：Usability

- 更完善的 checkpoint/resume 文档。
- 多任务 manifest。
- backend 版本升级测试。
- GPU smoke workflow。

### v0.3.0：Ecosystem

- 独立 benchmark adapters。
- 更多 harness/gateway backend。
- 任务下载与缓存管理。

研究算法、内部部署和长周期实验记录继续留在私有分支，不进入首版公共主线。

## 9. 明确不做的事情

1. 不在当前分支上直接 `git rm` 后推公开，避免历史泄漏。
2. 不把 benchmark 私钥改成占位文件后继续沿用同一历史。
3. 不用 README 链接到 Git 忽略的本地验收文档。
4. 不把内部集群默认值作为公共 fallback。
5. 不把未验证的多个环境/算法包装成稳定能力。
6. 不为了“功能全”保留 1 万多个任务资产和两个第三方整仓。
