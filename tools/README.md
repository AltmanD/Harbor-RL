# HarborRL 辅助工具

`tools/` 不提供训练、worker 启动或端到端验证入口。这些用户工作流统一位于
[`examples/`](../examples/README.md)。

```text
tools/
├── analysis/       # 轨迹、训练指标、case study 和 DIVE-PO 对比分析
├── evaluation/     # 安全/SWE/SETA/LWM 评测、probe 和结果汇总
├── dev/            # worker 等开发期冒烟与诊断
└── infra/          # 网络、可复现性快照与运行环境辅助
```

这里的脚本通常处理已有运行结果或协助开发，不是正式训练配方。典型用法：

```bash
python3 tools/analysis/analyze_trajectories.py --run-dir runs/training/<RUN_ID>
python3 tools/analysis/plot_training_metrics.py --run-dir runs/training/<RUN_ID>
bash tools/analysis/run_case_study.sh runs/training/<RUN_ID>
```

评测入口及参数见 [`docs/evaluation/README.md`](../docs/evaluation/README.md)；
LWM 离线验证的设计与命令见
[`docs/algorithms/lwm_offline_zh.md`](../docs/algorithms/lwm_offline_zh.md)。
