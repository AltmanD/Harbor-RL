# Harnesses

Harness 实现位于 `harborrl/harnesses/`，训练 recipe 通过 `HARNESS_OPTION` 选择：

- `camel-agent`：默认 terminal-agent harness；
- `claude-code`：Claude Code CLI 与 HarborRL MCP bridge。

例如：

```bash
HARNESS_OPTION=claude-code \
  bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
```

映射集中在 `harborrl/harnesses/factory.py`，使用惰性 import 保持可选依赖隔离。
展示名（canonical → 连字符形式）由同一文件的 `display_harness_name()` 提供。
训练与评估共用的 canonical 名称、别名和能力声明位于
`harborrl/harnesses/identity.py`；实现注册仍按能力分开，避免评估运行时
加载训练依赖。

注：PRM(process reward model）奖励 agent 不是 harness，位于
`harborrl/algorithms/prm/`，通过 recipe 的 PRM 相关参数启用。

## 训练 rollout harness vs 评估适配层

上面的 `camel-agent` / `claude-code`(`harborrl/harnesses/factory.py`）是
**训练 rollout** 用的 harness，由 recipe 的 `HARNESS_OPTION` 选择。

离线**评估**另有适配层 `harborrl/harnesses/eval/`，入口为
`create_eval_harness()` / `normalize_eval_harness_name()`，支持：

- `terminus-2`：Harbor terminus-2 agent(Terminal-Bench 风格数据集）;
- `claude-code`：Harbor 内置 claude-code agent;
- `camel-agent`:slime `eval_only` 链路。

两套注册表互相独立；评估的命令行工具、配置示例与批量/报告能力见
[tools/evaluation/README.md](../../tools/evaluation/README.md)，用户侧的
评测配方入口见 [examples/evaluation/README.md](../../examples/evaluation/README.md)。

另有 `tools/evaluation/benchmarks/swebench/report.py` 是被 rollout 管线
import 的官方格式导出库，与上述两层都不相同，三者分工勿混淆。
