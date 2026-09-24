# HarborRL

HarborRL 是面向适配 Harbor 格式任务的强化学习框架。它把模型 serving、隔离任务执行、verifier 回执、轨迹记录和 GRPO 训练连接起来，通过 Slime/Megatron actor 更新模型参数。

[English](README.md)

## 摘要

HarborRL 面向的是这样的训练场景：

- 模型调用、任务执行、verifier 结果、reward 和 policy version 被记录为同一段 rollout 历史，而不是事后从分散日志重建。
- Harbor 任务在外部 worker 的环境中运行，trainer 只负责调度、数据校验和训练。
- native 轨迹先经过校验并导出为后端中性数据，然后才交给 Slime/Megatron。

HarborRL 不是 benchmark 、模型、Docker 编排系统或通用 RL 库，也不替代 Slime、Megatron-LM 或 SGLang。

框架按 dataset、harness、model、train framework/algorithm 四个维度组织可插拔能力。这些选择在训练 YAML 中分开声明，实验时可以替换其中一个维度，同时保持 rollout 轨迹格式和训练数据导出口径不变。当前提供的默认组合是 Harbor task catalog、Claude Code harness、Qwen3-8B 与 SGLang serving 语义，以及 Slime/Megatron GRPO；这组组合已经通过 CPU 测试和固定版本 GPU 闭环验证。

## Framework

HarborRL 把产生数据的 rollout 和 actor 更新分开。native 侧生成并校验任务轨迹；adapter 侧只把被接受的训练字段交给 Slime 官方 hooks；Megatron 仍然是 actor 训练器。

```text
Training configuration
    -> native group rollout and Harbor execution
    -> Native IR and backend-neutral export
    -> Slime v0.3.2 official hooks
    -> Megatron actor update and checkpoint
```

| 区域 | 内容 |
| --- | --- |
| Native pipeline | 配置、Messages gateway、rollout 调度、Harbor-job runner、轨迹校验和导出 |
| Training integration | Slime v0.3.2 hook adapter、版本检查、launcher 规划和 policy/checkpoint 协调 |
| Examples | 离线 CPU fixture 和通用 Qwen GPU 启动模板 |
| Tooling | 后端 bootstrap、SGLang 语义 probe 和发布审计 |
| Tests | 覆盖配置、gateway、生命周期、导出和 adapter 行为的 CPU 套件 |

可插拔维度可以概括为：

| 维度 | 当前选择 | 主要配置 |
| --- | --- | --- |
| Dataset | Harbor 格式 task catalog | `tasks` |
| Harness | Claude Code profile | `harness` |
| Model | 配置化的 actor/reference checkpoint 和 serving 语义 | `model` 和 `gateway` |
| Train framework / algorithm | Slime/Megatron GRPO hooks | `training` |

## 安装

使用 Python 3.10 或更新版本。CPU 开发不需要 CUDA、Docker、模型权重、Slime、Megatron-LM 或 SGLang。

创建虚拟环境：

```bash
python -m venv .venv
. .venv/bin/activate
```

预期结果：创建 `.venv/`，shell 提示符出现 `(.venv)`。两个命令都不会启动服务，也不会修改示例文件。

安装可编辑包和开发工具：

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

预期结果：pip 安装 HarborRL、`pytest` 和 `ruff`，但不会安装重型训练栈，也不会编译 CUDA kernel。

检查命令行入口：

```bash
harborrl --help
```

预期结果：argparse 输出训练 CLI 用法并以状态码 0 退出。公开命令是 `train` 和 `doctor`，两者都要求 `--config`。

## CPU 启动

### 查看示例启动计划

```bash
python -m harborrl.cli train --config examples/native/train_qwen_native.yaml --dry-run
```

预期结果：命令输出一个包含 `config`、`command`、`environment` 和 `training_command` 的 JSON 对象。`config` 中的路径会被解析成绝对路径，GPU 和采样预算会被规范化，`training_command` 显示外部 native-train dispatcher。命令以状态码 0 退出，不会创建 run 目录，不会启动 Ray 或 Docker，不会加载模型，也不会连接 worker。

示例刻意使用 `worker-1`、`/models/qwen3-8b` 和占位 serving 摘要等通用值。在 dry-run 输出中看到这些值是预期行为，正式训练必须替换它们。

### 运行离线示例

```bash
bash examples/native/cpu_contract_smoke.sh
```

预期结果：脚本输出 JSON，其中 `"status": "passed"`，`task_digest` 是 64 位十六进制字符串，`reward_contrast` 为 `[0.0, 1.0]`，两个 `advantages` 数值相反且非零，两个 group 成员的 rollout ID 相同。脚本以状态码 0 退出。

该示例会锁定内置 hello-world 任务，构造两种结果的 reward 回执，组装一个完整 Native IR group，导出后端中性 batch，并校验 Slime adapter postprocess。不使用 Docker、live model、Harbor worker 或 GPU；仅证明 CPU 侧行为正确，而非训练链路可用。

## GPU 启动

### 检查节点并准备固定版本后端

确认节点暴露的 GPU 数量不少于配置数量：

```bash
nvidia-smi
```

预期结果：`nvidia-smi` 列出预期 GPU 和显存状态。示例 YAML 默认使用 8 卡，其中 4 卡给 actor，4 卡给 rollout。

在有 Git、Docker 和网络访问的节点上安装固定版本开源栈：

```bash
scripts/bootstrap_backends.sh /path/to/backends
. /path/to/backends/harborrl-backend.env
```

预期结果：脚本检出 Slime commit `3778dbf6d1a533ab478ecf5ddaa11449a47752b2`、Megatron-LM commit `1dcf0dafa884ad52ffb243625717a3471643e087`，拉取 `lmsysorg/sglang:v0.5.15.post1-cu129`，并写出 `harborrl-backend.env`。source 该文件后，当前 shell 会设置 `SLIME_DIR`、`MEGATRON_DIR` 和 `SGLANG_IMAGE`。


### 准备私有机器配置

把 `examples/native/train_qwen_native.yaml` 复制到私有路径。替换通用 worker、actor/reference checkpoint 路径、gateway origin、tokenizer/template 摘要、GPU 布局和输出位置；同时让 `tasks.catalog` 和 `harness.profile` 指向目标 catalog 和 profile，因为相对路径按私有 YAML 所在目录解析。

预期结果：私有 YAML 只包含对你的机器有效的值。

在准备 worker 前先验证规范化后的私有计划：

```bash
python -m harborrl.cli train --config /path/to/private.yaml --dry-run
```

预期结果：JSON 中出现你的私有值和解析后的路径，`training.backend_contract` 仍为 `slime-v0.3.2-native-v1`，该命令不会创建 run 目录或启动后端。

### 准备 Harbor worker

每个 worker 需要 Linux、Docker Engine、来自 trainer 的免密 SSH、Python 3.12 或更新版本，以及 Harbor `0.23.0`。worker 必须能拉取或本地构建 catalog 中的所有任务镜像，并能访问 advertised Messages gateway；它不需要模型 provider 的 API key。

在每个 worker 安装匹配的 HarborRL runner 和 Harbor runtime。直接探测的示例：

```bash
ssh WORKER 'cd /path/to/Harbor-RL && /opt/harbor/bin/python -m harborrl.rollout.harbor_job.runner --probe'
```

预期结果：worker 输出 JSON，其中 `harbor_version` 为 `0.23.0`，Python 版本不低于 3.12，Claude option 字段可用，Harbor trial 接口可调用。命令以状态码 0 退出，且不会创建 trial。

### 运行 doctor

```bash
harborrl doctor --config /path/to/private.yaml
```

预期结果：doctor 输出包含 `checks` 和 `scope` 的 JSON 对象。所有检查项都是 `"ok": true`，进程以状态码 0 退出，并且不会启动训练。

doctor 会检查私有 YAML、catalog 摘要、模型文件、worker SSH 探测、固定后端路径和 commit、必需 import、hooks、GPU 预算以及 SGLang image tag。doctor 不会启动模型服务，也不会生成 token，因此 serving 协议行为、token ID 和 logprob 会在后续 live 语义检查或真实训练中验证，而不是由 doctor 验证。

### 启动训练

```bash
harborrl train --config /path/to/private.yaml
```

预期结果：命令先再次运行 doctor，然后 HarborRL 在 `output.root/training/` 下创建以时间戳和 run ID 命名的新目录。run 中会记录 launch plan、source manifest、物化后的 task catalog、prompt rows、轨迹、policy 历史、metrics，以及按保存间隔生成的 checkpoint。成功运行的进程状态码为 0。

如果任意 preflight 检查失败，训练不会启动，失败检查会以 JSON 输出到 stderr。如果外部后端异常退出，子进程状态码会被透传，run 目录会保留失败时已有的证据供检查。

## 运行配置与任务格式

训练通过一个 YAML 文件配置，section 和字段集合是固定的。相对路径按 YAML 文件所在目录解析。可重复使用 `--set section.field=value`，值会按 YAML 解析并再次校验。

| Section | 用途 |
| --- | --- |
| `execution` | 固定选择 `harbor_job` backend |
| `tasks` | 选择非空 immutable task catalog |
| `harness` | 选择 Claude CLI、模型和 timeout profile |
| `harbor` | 配置 Harbor 版本、解释器和 SSH worker |
| `gateway` | 配置 Messages listener、origin、serving 摘要和 raw-logprob 审计 |
| `model` | 配置 actor/reference checkpoint 和 Slime model preset |
| `training` | 配置 rollout 数、学习率、保存节奏和 backend 标识 |
| `sampling` | 配置 group 大小、batch group 数、重试、上下文和生成限制 |
| `deployment` | 配置 GPU 数量和 actor/rollout 并行度 |
| `output` | 配置 run tree 根目录 |

一个 native task 是内容寻址目录：

```text
task/
├── task.toml
├── instruction.md
├── environment/
│   └── Dockerfile
└── tests/
    └── test.sh
```

catalog 身份格式为：

```json
{
  "id": "task",
  "revision": "source-revision",
  "path": "relative/or/absolute-task",
  "task_digest": "64-hex-sha256",
  "reward_profile": {
    "key": "reward",
    "scale": 1.0,
    "offset": 0.0,
    "raw_range": [0, 1]
  }
}
```

verifier 会写出共享环境 terminal result 以及 `reward.json` 或 `reward.txt`。HarborRL 通过 reward profile 分别解析两者，要求选定值一致，并记录源字节和 SHA-256 回执。布尔值和非有限 reward 会被拒绝。

机器可读 Hub inventory 位于 [configs/harbor_hub/manifests.yaml](configs/harbor_hub/manifests.yaml)。只有记录上游 revision、license、任务布局、资源要求、摘要、每类 worker 的奖励对比，以及一条完整 native 训练 batch 后，数据集才会标记为 `supported`。

## 贡献、安全与许可

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，漏洞报告流程见 [SECURITY.md](SECURITY.md)。HarborRL 使用 MIT 许可证；第三方声明见 [LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。
