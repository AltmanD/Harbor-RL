# HarborRL 公开发布整理方案（单仓修订版）

日期：2026-09-23（修订）  
适用对象：当前仓库 `AltmanD/Harbor-RL` 的 `main`、`feat/harborrl-mvp`、`refactor/slime-v032-exporter-dev` 及后续发布分支  
配套文档：`docs/branch_management_plan_zh.md`、`docs/exporter_refactor_plan_zh.md`、`docs/exporter_v032_dev_status_zh.md`  
结论：**继续使用当前仓库发布**。从 `refactor/slime-v032-exporter-dev` 建新分支 `release/public-v0.1`，在该分支完成 tracked tree 清理、example、README 与全部验收后，以普通 PR 合入 `main`，再从 `main` 打 `v0.1.0-public` 公开。不新建公开仓库、不改写历史、不 force push。

## 0. 执行摘要

1. 发布路径：`refactor/slime-v032-exporter-dev → release/public-v0.1 → PR 合入 main → main 打 v0.1.0-public`。
2. 发布前必须完成：tracked tree 清理（benchmarks / vendored backends / deploy / 旧研究栈）、hello-world example、双语 README 重写、CPU CI、Harbor Hub 兼容 matrix、GPU 单步验收、package build 与发布审计。
3. 训练后端目标：Slime v0.3.2 官方组合 + `harborrl/backends/slime_v032/` 薄 adapter（零 patch 优先）；GPU 决策门见 §8。
4. 已接受的存量风险（owner 决策，2026-09-23）：`main` 与 MVP 历史中的 SETA fixture 私钥及内部信息**不作为本次发布阻断项，不做历史清理**。但发布分支的新增内容必须通过 hygiene 扫描，不得引入新的密钥、内部地址、私有数据或个人路径。
5. Harbor Hub 已适配数据集的 inspect / catalog / Native rollout 兼容能力仍是清理阻断项；`probe.py` 对 `environments/terminal` 的依赖必须给出替代。

## 1. 当前状态盘点（2026-09-23）

| 分支 / 引用 | 提交 | tracked 文件 | 大小 | 状态 |
| --- | --- | ---: | ---: | --- |
| `origin/main` | `dce4cfaa` | 13,128 | 182.1 MiB | 受保护集成基线；等待接收发布 PR |
| `origin/feat/harborrl-mvp` | `999513bb` | 13,198 | 182.4 MiB | 已验收 MVP；相对 main 22 个提交 |
| 本地 `feat/harborrl-mvp` | `8af8a478` | — | — | 领先 origin 1 个提交；共享工作区被其他进程占用 |
| `refactor/slime-v032-exporter-dev` | `c16a8f6e` | ~13,216 | ~182.5 MiB | 基于 `999513bb`；含中性 exporter、v0.3.2 adapter 与三份方案文档；CPU 目标套件 91 passed |

refactor 分支增量：`harborrl/export/`（纯 Python TrainingBatch 契约）、`harborrl/backends/slime_v032/`（五个官方 hook + versions/launcher）、`training.backend_contract` 配置、launcher 选择器、doctor 静态检查、`tests/harborrl/test_slime_v032_exporter.py`。未完成：官方镜像 spike、GPU 单步验收、loss 缩放核对、真实 rollout 集成、Hub matrix。

## 2. 发布决策（单仓模型）

### 2.1 Owner 决策

1. **单仓发布**：继续使用 `AltmanD/Harbor-RL`，不新建公开仓库，不使用 orphan root。
2. **存量历史风险接受**：`main` 历史中的 benchmark 私钥与内部信息不处理、不作为阻断；本方案只保证**发布分支的新增 diff 与最终 tracked tree** 不引入新的敏感内容。
3. **main 保护规则不变**：只通过 reviewed PR 更新，不直接 push，不 force push，不改写历史。
4. **共享工作区规则**：发布整理在独立 clone / worktree 中进行，不动被占用的 `Harbor-RL` 工作区，不执行 `git clean -fdx`。

### 2.2 分支流水线

```text
refactor/slime-v032-exporter-dev (c16a8f6e)
  └── release/public-v0.1          # 新建：清理 + example + README + 验收
        ├── cleanup tracked tree
        ├── hello-world example
        ├── bilingual README + docs
        ├── CPU CI / Hub matrix / GPU gate
        └── PR → main
main
  └── v0.1.0-public tag            # PR 合入后打 tag
```

说明：release 分支继承 MVP + refactor 历史，合入 main 的 PR 同时包含“新增 Native/v0.3.2 内容”与“删除 benchmarks/backends/deploy 等 tracked 资产”两类变更，review 时必须按 §4 白名单与 §5 删除清单双重核对。

## 3. 发布阻断项

| 编号 | 阻断项 | 处理 |
| --- | --- | --- |
| P0-1 | README 与 docs 状态矛盾（“native 尚未验收”过期表述、LightRL 旧身份、临时链接） | 按 §7 重写双语 README |
| P0-2 | vendored Slime/Megatron 整仓（2,466 文件 / 37.6 MiB） | 从 release tree 移除，改为官方 pin + bootstrap |
| P0-3 | `probe.py` 依赖 `environments/terminal.TerminalEnv`，而公开树计划移除旧环境栈 | 最小 probe 兼容层或 Native Runner 改造，二选一 |
| P0-4 | v0.3.2 GPU 单步验收未完成 | §8 决策门通过前不得打 v0.1 tag |
| P0-5 | Harbor Hub matrix 未跑 | §9 通过前不得合入 release → main |
| P0-6 | 无公开 example | §6 hello-world 四件套必须可用 |

## 4. 公开 v0.1 目标树

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
├── configs/harbor_hub/manifests.yaml
├── scripts/
│   ├── bootstrap_backends.sh
│   ├── sglang_semantic_probe.py
│   └── audit_public_tree.sh
├── harborrl/
│   ├── cli.py
│   ├── config/native.py
│   ├── gateway/
│   ├── data/harbor/
│   ├── data/download.py
│   ├── tasks/cli.py
│   ├── platform/native_train.py
│   ├── platform/slime_train*.sh
│   ├── rollout/native_generate.py
│   ├── rollout/harbor_job/
│   ├── export/
│   ├── backends/slime_v032/
│   └── trajectories/native.py
└── tests/harborrl/
    ├── test_native_contracts.py
    ├── test_native_gateway.py
    ├── test_native_lifecycle.py
    ├── test_native_training.py
    └── test_slime_v032_exporter.py
```

## 5. tracked tree 清理清单

### 5.1 `harborrl` 包收敛

| 现有模块 | 处理 | 原因 |
| --- | --- | --- |
| `config/native.py` | 保留 | Native schema 2 + backend_contract 核心 |
| `gateway/`、`trajectories/native.py` | 保留 | Gateway / IR / reward 主链路 |
| `rollout/native_generate.py`、`rollout/harbor_job/` | 保留 | Native rollout 主链路 |
| `export/`（新） | 保留 | 中性 TrainingBatch，CPU 可测 |
| `backends/slime_v032/`（新） | 保留并设为默认 | Slime v0.3.2 官方 hook adapter |
| `rollout/exporters/native.py` | 保留为兼容薄层或并入 `export/` | 数值基准 |
| `rollout/exporters/native_slime.py` | GPU 门通过后从 release tree 删除 | legacy patched 路径，不进公开 v0.1 |
| `data/harbor/{inspector,native_inspector,receipt}.py` | 保留 | task/receipt 校验 |
| `data/harbor/{materializer,probe}.py`、`data/download.py`、`tasks/cli.py` | 保留或等价替代 | Hub 兼容阻断项 |
| `environments/terminal/` | 移除；probe 未改造前保留最小层 | 旧 TerminalEnv 栈 |
| `environments/`、`harnesses/`、`algorithms/`、`data/convert_*`、旧 interactive rollout 栈 | 移除 | SETA/AgentHarm/Tau2/SWE/DIVE-PO/LWM 等研究栈 |
| `platform/worker_*`、`router*`、`run_leases`、`misc/` 大部 | 移除或仅保留 Native smoke 必需 | 旧运维与可选依赖 |

### 5.2 顶层目录

| 目录 | 处理 | 替代 |
| --- | --- | --- |
| `benchmarks/`（10,337 文件 / 139.6 MiB） | 整体 `git rm` | hello-world example + Hub manifest/下载说明 |
| `backends/`（2,466 文件 / 37.6 MiB） | 整体 `git rm` | `scripts/bootstrap_backends.sh` pin 官方版本 |
| `deploy/`（31 文件） | 整体移除 | `examples/native/worker_setup.md`（无内部默认值） |
| `tools/`（87 文件） | 仅保留 `sglang_semantic_probe.py`（迁 `scripts/`） | 其余留私有分支 |
| `docs/`（33 文件） | 收敛到 7 个公开文档 | 内部验收记录留私有分支 |
| `examples/` | 仅保留 `examples/native/` | — |
| `tests/` | 保留 §4 五个核心测试并补 hygiene/install/bootstrap 测试 | 删除旧环境与研究栈测试 |
| 根目录杂项 | 移除 `legacy-requirements.txt`、`sitecustomize.py`、`runs/rjob/`、旧 logo、本地 probe | 更新 `pyproject.toml` 只打包 `harborrl*` |

清理规则：在 release 分支用普通 `git rm` + 提交完成；不删除本地未跟踪实验内容；本地 `runs/` 原样保留。

## 6. v0.1 Example 要求（发布必含）

1. **`examples/native/tasks/hello_world/`**：可公开、可离线检查、无真实凭据的最小 Harbor task；通过 `inspect_native`，digest 稳定；reward 0/1 可由 fixture 复现。
2. **`cpu_contract_smoke.sh`**：无 GPU、无内部网络、无外部 backend 即可运行；覆盖 task inspect → digest → receipt → IR → `export_training_batch` → converter 标准字段校验；退出码非 0 即失败。
3. **`train_qwen_native.yaml`**：显式声明 model、gateway、harbor workers、catalog、`training.backend_contract: slime-v0.3.2-native-v1`；无个人绝对路径、无私有镜像默认值；`--dry-run` 输出完整可解释命令。
4. **`worker_setup.md`**：外部 Harbor worker 的前置条件、Python/Harbor 版本、网络与安全要求、环境变量注入、健康检查、常见 fail-closed 行为。
5. 验收：example 四件套在 clean clone + `pip install -e .[dev]` 环境下全部可用，并纳入 CI。

## 7. README 要求（发布必含）

双语同构，结构固定：

1. **What is HarborRL**：三句话说明 Native agentic RL MVP 范围。
2. **Status**：已验证能力（CPU 契约、GPU 单步/有界训练）与未承诺项（多 benchmark、多轮稳定训练、集群运维）。
3. **Architecture**：Config → Doctor → Gateway → Harbor Runner → Reward/IR → Neutral Export → Slime v0.3.2 → Checkpoint。
4. **Install**：基础安装、optional extras（`native`/`gateway`/`train`/`dev`）、backend bootstrap。
5. **Quickstart**：hello-world CPU contract smoke 一条命令跑通。
6. **GPU Training**：dry-run 前置条件、外部 worker、pinned backend、最小 YAML。
7. **Repository map**：≤20 行。
8. **Extension points**：task、gateway backend、training backend、reward/trajectory schema。
9. **Testing**：CPU CI 与 optional GPU 门禁。
10. **Acknowledgement / Citation / License**。

必须修正：`cleanup_status.md` 临时链接、“native 尚未验收”过期陈述、LightRL 旧命名/旧 logo/旧 citation、backend 前置条件缺失。README review 作为发布前验收项（§11）。

## 8. Exporter 决策门

| 状态 | 条件 | release 动作 |
| --- | --- | --- |
| 目标路径 | v0.3.2 官方镜像 spike + GPU 单步验收通过，零 patch 成立 | tree 只含 `export/` + `backends/slime_v032/`；删除 legacy exporter；无 patch 文件 |
| 降级路径 | GPU 证明必须最小 patch | 保留 v032 adapter + `patches/slime/*.patch` + hash 门禁；理由写入 `backend-integration.md` |
| 阻塞路径 | GPU 未完成或 loss 缩放未确认 | 不打 v0.1 tag；不发布未验证默认能力 |

GPU 最低验收：官方组合启动、0/1 reward contrast、rollout logprob 与 Gateway 审计一致、非零 advantage/gradient/PG loss、显式权重变化、checkpoint 保存重载、新 policy version 生效、与已验收 MVP 关键指标误差在允许范围。

## 9. Harbor Hub 数据兼容门

发布定义：外部 task root / Hub 下载 / 本地缓存可用；`harborrl hub inspect`（或等价 CLI）可检查 task；inspector 识别 profile；catalog 记录 dataset/revision/path/digest；Native runner 按 catalog 执行；reward receipt 进 IR；v0.3.2 batch 保持 group/trajectory/turn 语义；公开树无私有数据与个人路径。

发布前建立 `docs/dataset-compatibility.md` 或 `configs/harbor_hub/manifests.yaml`：dataset 名、upstream repo/revision/branch、license、下载方式、task root 约定、expected task count、digest 策略、inspector 预期、Docker/GPU 需求、已过 Native smoke、负责人。任何 dataset 无解释地从 `SUPPORTED` 变 `UNSUPPORTED`，不得合入。

## 10. 实施步骤

### Phase 0：准备（0.5 天）

1. 在独立 clone 中从 `refactor/slime-v032-exporter-dev`（`c16a8f6e`）建 `release/public-v0.1`。
2. 可选留档：为 MVP 验收点 `999513bb` 打 tag / 建归档分支；ignored 实验内容按 allowlist 归档（不删除）。
3. 确认 refactor 分支 CPU 基线（91 tests）绿色后开始清理提交。

### Phase 1：tracked tree 清理（1–2 天）

1. 按 §5 `git rm` benchmarks、vendored backends、deploy、旧研究栈与内部运维模块。
2. 处理 `probe.py` → TerminalEnv 依赖（最小兼容层或 Native Runner 改造）。
3. 调整 `pyproject.toml` 打包范围与 extras。
4. 每类删除独立提交，保持 PR 可 review。

### Phase 2：example 与文档（1 天）

1. 实现 §6 四件套并接入 CI。
2. 按 §7 重写双语 README 与 7 个核心文档。
3. 建立 Hub manifest / dataset compatibility 文档。

### Phase 3：验收与门禁（1 天 + GPU 等待）

执行 §11 全部检查；未过项修复后重跑。GPU 门与 Hub matrix 是硬门。

### Phase 4：合入与打标（0.5 天）

1. push `release/public-v0.1`，发起 → `main` 的 PR。
2. review 白名单新增 + 删除清单双重核对。
3. merge 到 `main`（不改写历史、不 force push）。
4. 在 merge 后的 `main` 打 `v0.1.0-public` 并 push tag。

## 11. 发布前验收清单

- [ ] release 分支基于 `refactor/slime-v032-exporter-dev`，且 CPU 目标套件通过（contracts/gateway/lifecycle/training/slime_v032）。
- [ ] tracked tree 符合 §4/§5：无 benchmarks、vendored backends、deploy、旧研究栈、内部运维脚本。
- [ ] hello-world example 四件套可用并进 CI。
- [ ] 双语 README 符合 §7，无过期状态、旧身份、临时链接与失效引用。
- [ ] §8 exporter 决策门有结论：零 patch 或最小 patch + hash 门禁；默认入口唯一。
- [ ] Slime/Megatron/SGLang 版本 doctor 校验通过；`--dry-run` 命令完整且无内部默认地址。
- [ ] §9 Hub matrix 通过；`SUPPORTED` 无未解释缩减。
- [ ] GPU 单步验收通过（或明确记录 owner 批准的例外，不建议）。
- [ ] clean clone + `pip install -e .[dev]` + `harborrl --help` 成功。
- [ ] `python -m build` 成功；CI（pytest/ruff/shell syntax/package/hygiene）绿色。
- [ ] 新增 diff hygiene 扫描：无新密钥、内部域名/IP/SSH alias、私有 registry、个人绝对路径、大数据与 checkpoint。
- [ ] PR review 完成后合入 main；tag 只打在 merge 后的 main。

## 12. 发布切分

- **v0.1.0-public**：Native MVP framework + v0.3.2 adapter（零 patch 优先）+ hello-world example + Qwen GPU dry-run 配置 + 核心 CI + 双语 README。
- **v0.2.0**：checkpoint/resume 文档、多任务 manifest、backend 升级测试、GPU smoke workflow。
- **v0.3.0**：benchmark adapters 独立包、更多 harness/gateway backend、任务下载与缓存管理。

## 13. 明确不做的事情

1. 不新建公开仓库，不使用 orphan root。
2. 不改写 `main` 历史，不 force push；`main` 只接收 reviewed PR。
3. 不在本次处理 `main` 历史中的存量私钥（owner 已接受该风险）。
4. 不在共享工作区切换分支或执行 `git clean -fdx`；本地实验内容不删除。
5. 不把内部集群默认值、私有 registry、SSH alias 作为公共 fallback。
6. 不把未验证的多环境/多算法包装成稳定能力。
7. 不为“功能全”保留 1 万+ 任务资产与两个第三方整仓。
8. 不在 GPU / Hub / example / README 门未过时打 v0.1 tag。
