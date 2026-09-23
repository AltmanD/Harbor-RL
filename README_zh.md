# HarborRL

HarborRL 是面向 Harbor 任务的可审计智能体强化学习框架。它把任务执行、
serving 证据、verifier 回执和训练张量放在严格契约之后，再通过一个很薄的
adapter 连接外部 Slime v0.3.2 后端。

[English](README.md)

## 状态

公开 v0.1 树包含 native 框架、CPU 契约测试、hello-world 示例和官方
Slime v0.3.2 adapter。CPU 契约已通过；在记录固定版本 GPU 单步验收和
Harbor Hub 兼容矩阵之前，不会打发布 tag。见 [发布范围](docs/release-scope.md)。

## 安装

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
harborrl --help
```

训练依赖刻意外部化：Slime、Megatron-LM、SGLang、模型权重和 Harbor worker
都不是本包的 Python 依赖。可在 GPU 机器使用
[bootstrap_backends.sh](scripts/bootstrap_backends.sh)，或提供已经固定版本的
环境。

## 快速开始

运行离线 0/1 reward 对比和 adapter 契约检查：

```bash
bash examples/native/cpu_contract_smoke.sh
```

不启动 Ray、Docker、SGLang 或训练，查看完整 GPU 命令：

```bash
python -m harborrl.cli train \
  --config examples/native/train_qwen_native.yaml --dry-run
```

示例中的 worker、模型路径和 gateway 名都是通用占位符。请在私有未跟踪副本中
替换，不要提交机器特定配置。

## 仓库地图

| 路径 | 用途 |
| --- | --- |
| `harborrl/config/` | 严格 native schema-2 加载与 launch plan |
| `harborrl/gateway/` | Anthropic 兼容 Messages gateway 与 SGLang 证据 |
| `harborrl/rollout/` | native group 生成与外部 Harbor runner |
| `harborrl/trajectories/native.py` | Native IR v2、身份、奖励与 trace 校验 |
| `harborrl/export/` | 后端中性的不可变训练 batch |
| `harborrl/backends/slime_v032/` | 官方 Slime hook adapter 与版本门 |
| `examples/native/` | 离线任务 fixture 与 Qwen 启动示例 |
| `scripts/` | 后端 bootstrap、语义 probe 与发布审计 |

## 测试

```bash
python -m pytest
python -m ruff check .
bash -n harborrl/platform/slime_train.sh scripts/*.sh
bash scripts/audit_public_tree.sh
```

CI 运行 CPU 套件、示例、dry-run、打包和 hygiene 审计。GPU 语义探测与单步
训练是单独的显式门禁。

## 文档

- [快速开始](docs/quickstart.md)
- [架构](docs/architecture.md)
- [配置](docs/configuration.md)
- [任务格式](docs/task-format.md)
- [后端集成](docs/backend-integration.md)
- [数据集兼容性](docs/dataset-compatibility.md)
- [发布范围](docs/release-scope.md)

## 贡献、安全与许可

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，漏洞报告流程见
[SECURITY.md](SECURITY.md)。HarborRL 使用 MIT 许可证；第三方声明见
[LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。
