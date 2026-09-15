# HarborRL 用户工作流

`examples/` 是仓库的公开训练与评估入口；`tools/` 只保存分析、评测和
开发诊断等辅助工具的实现。

```text
examples/
├── training/                   # 正式训练配方
│   ├── train.sh
│   ├── train_qwen3_8b_seta_dapo.sh
│   ├── train_qwen3_8b_seta_dive_po.sh
│   ├── train_qwen3_8b_mixed_dapo.sh
│   ├── train_glm_5_1_seta_dapo.sh
│   └── world_model/             # tb2.1 LWM 离线训练、MPC 与 rjob 提交
├── validation/                 # 不含站点拓扑的通用辅助文件
└── evaluation/                 # 离线评估配方(机制在 tools/evaluation/)
```

## 评估入口

离线评估(对任意 ckpt 跑 benchmark、批量对比多个 ckpt)的用户配方见
[evaluation/README.md](evaluation/README.md);评估机制(CLI、配置、
harness 适配器)由 `tools/evaluation/` 与 `harborrl/harnesses/eval/`
提供。

## 训练入口

| 脚本 | Harness | Model | 数据 | 算法 |
|---|---|---|---|---|
| `training/train_qwen3_8b_seta_dapo.sh` | Camel-Agent | Qwen3-8B | SETA | DAPO |
| `training/train_qwen3_8b_seta_dive_po.sh` | Camel-Agent | Qwen3-8B | SETA | DIVE-PO |
| `training/train_qwen3_8b_mixed_dapo.sh` | Camel-Agent | Qwen3-8B | SETA + Agent-SafetyBench + AgentHarm | DAPO |
| `training/train_glm_5_1_seta_dapo.sh` | Camel-Agent | GLM-5.1 | SETA | DAPO |
| `training/train.sh` | 由配置选择 | 由配置选择 | 由配置选择 | 由配置选择 |

`training/world_model/` 是独立实验流程，不会改变稳定 GRPO/DAPO 配方。

先用 `--dry-run` 检查配置：

```bash
WORKER_URLS=http://127.0.0.1:18081 \
  bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_mixed_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_seta_dive_po.sh --dry-run
bash examples/training/train_glm_5_1_seta_dapo.sh --dry-run
```

真实 SETA 训练需要通过 `WORKER_URLS` 或 `WORKER_URLS_FILE` 提供可达的 Docker
worker。站点专用地址放在环境变量或被 Git 忽略的本地文件中，
不要写入公共配置。

若已准备好 GPU 与 Docker worker，启动完整 4-GPU SETA+DAPO 训练：

```bash
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
  bash examples/training/train_qwen3_8b_seta_dapo.sh
```

脚本启动前会检查 worker 健康状态和 4 张可见 GPU，并以前台方式运行。
必须显式设置 `WORKER_URLS`；可用 `RUN_ID` 覆盖运行名；`BACKGROUND=1` 时日志写入
`runs/training/<RUN_ID>/launcher.log`（评测脚本对应 `runs/evaluation/`，测试对应
`runs/testing/`）。

## 验证入口

公开仓库保留单元测试与配方 `--dry-run`。含集群地址、调度器参数和本地
数据路径的端到端验证 wrapper 不入库。

GLM-5.1 训练还需要 `HF_CKPT`、`REF_LOAD` 和兼容的
`MODEL_ARGS_FILE`。完整部署、输出目录和排障说明见仓库根目录
[README](../README.md)。
