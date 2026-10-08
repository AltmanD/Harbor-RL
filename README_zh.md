<p align="center">
  <img src="assets/HarborRL-title.jpg" alt="HarborRL" width=800>
</p>

<p align="center">
  <h1>HarborRL</h1>
</p>

<p align="center">
  <strong>面向所有人的轻量级 agentic reinforcement learning 框架</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg" alt="PyTorch 2.4+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文
</p>

---

## News

- **10/2026** 📣📣📣 HarborRL released

## About

HarborRL 是一个用于在可验证、沙箱化任务上训练工具使用型 agent 的轻量级框架。它将模型 serving、轨迹生成、工具执行、verifier reward 和策略更新组织成一条可复现的完整流水线。

HarborRL 支持灵活组合 **Models × Algorithms × Tasks × Harnesses (MATH)**：

- **Models:** 使用 Qwen 和 GLM 作为基础模型进行 RL 训练。
- **Algorithms:** 支持 GRPO 和 DAPO 强化学习训练算法。
- **Tasks:** 在隔离 worker 中对 **200+** 个 Harbor 格式任务进行 RL 训练，包括 Terminal-Bench、SWE-Bench、Deep-SWE 等。
- **Harnesses:** 支持 **40+** 个 harness 框架，包括 Claude Code、Codex、LangGraph、SWE-agent 等。

## 快速开始

先基于 `examples/native/train_qwen_native.yaml` 准备一个YAML，然后用以下命令启动完整 native 流水线：

```bash
bash scripts/launch_native.sh \
  --config /path/to/private.yaml \
  --backends /path/to/harborrl-backends
```

launcher 会复用已有的 `harborrl-backend.env`，或在需要时 bootstrap 固定版本的 Slime、Megatron-LM 与 SGLang 栈；加载这些设置后调用 `harborrl train`。训练会重新执行 doctor，在 `output.root/training/` 下创建带时间戳的 run 目录；只有配置、任务、worker、checkpoint、后端版本、依赖和 GPU 预算全部通过预检后才会启动。

使用 `--doctor` 可只执行预检，使用 `--dry-run` 可输出解析后的 launch plan。私有 YAML 必须指向可达的 Harbor worker、task catalog、harness profile、model/reference checkpoint、gateway URL 与 serving 摘要、GPU 布局以及输出根目录。不要将凭据或站点专有配置提交到仓库。

## 安装

使用 Python 3.10 或更新版本。CPU 开发环境不需要 CUDA、Docker、模型权重、Slime、Megatron-LM 或 SGLang。

创建并进入虚拟环境：

```bash
python -m venv .venv
. .venv/bin/activate
```

预期结果：创建 `.venv/`，shell 提示符显示 `(.venv)`。两个命令都不会启动服务，也不会修改示例。

安装可编辑包和开发工具：

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

预期结果：pip 安装 HarborRL、`pytest` 和 `ruff`，但不会安装重型训练栈。命令完成时不会编译 CUDA kernel。

检查命令行入口：

```bash
harborrl --help
```

预期结果：argparse 输出训练 CLI 用法，并以状态码 0 退出。公开命令是 `train` 和 `doctor`，两者都要求提供 `--config`。

## 运行配置与任务格式

训练由一个 YAML 文件配置，其 section 和字段集合固定。相对路径按 YAML 文件所在目录解析。可重复使用 `--set section.field=value` 覆盖配置；值会按 YAML 解析并再次校验。

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

verifier 会写出共享环境 terminal result 以及 `reward.json` 或 `reward.txt`。HarborRL 通过 reward profile 解析两者，要求选定值一致，并记录源字节和 SHA-256 回执。布尔值和非有限 reward 会被拒绝。

机器可读 Hub inventory 位于 [configs/harbor_hub/manifests.yaml](configs/harbor_hub/manifests.yaml)。只有记录上游 revision、license、任务布局、资源要求、摘要、每类 worker 的奖励对比，以及一条完整 native 训练 batch 后，数据集才会标记为 `supported`。

## License

HarborRL 使用 MIT 许可证发布。

## Citation

如果你在研究中使用 HarborRL，请引用我们的技术报告：

```text

```

## Acknowledgement

感谢 [Slime](https://github.com/THUDM/slime) 和 [Harbor](https://github.com/harbor-framework/harbor) 为训练与任务执行构建的基础设施。
