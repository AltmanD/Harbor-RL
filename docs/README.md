# HarborRL 文档导航

## 架构与配置

- [Harbor Gateway 最终 MVP 设计](integration/harbor_gateway_mvp_final_zh.md)——
  Harbor 原生执行，Terminus2 / Claude Code / Codex / OpenHands 四 harness、Gateway 三协议、统一 reward 与训练验收；后续完整 MVP 的设计基线。
- [native_mvp_fullchain_readiness_zh.md](native_mvp_fullchain_readiness_zh.md)——native MVP 真实训练前的 GPU/CPU worker、task lock、CLI、镜像、doctor 与 Go/No-Go 清单。
- [native_mvp_gpu_result_zh.md](native_mvp_gpu_result_zh.md)——2026-09-22
  8×H200 首次真实全链路成功结果、验收矩阵、checkpoint 重载与清理状态。
- [native_mvp_reward_contrast_result_zh.md](native_mvp_reward_contrast_result_zh.md)——
  2026-09-22 最终 reward 0/1 对比、非零 advantage/gradient/PG loss、显式
  权重变化与 updated checkpoint 重载证据；Native MVP 验收闭环。
- [native_cpu_development_zh.md](native_cpu_development_zh.md)——原生 Claude Code
  CPU/契约开发边界、schema 2 入口、Slime 训练契约测试与 GPU 验证记录。
- [architecture.md](architecture.md)——包边界、主链路与扩展点（环境注册表、
  env 解析、训练三层入口)。
- [configuration.md](configuration.md)——recipe 即配置；`harborrl/env.py`
  的 `ENV_VARS` 声明表；`environments/registry.py` 的 `EnvSpec` 表。
- [records/refactor/refactor_review_20260731.md](records/refactor/refactor_review_20260731.md)——2026-07-31
  包结构审查记录（分层边界的历史快照）。

## 算法

- [algorithms/dive_po_dual_stream.md](algorithms/dive_po_dual_stream.md)——
  DIVE-PO dual-stream advantage 注入的公式与正确性分析；对应实现
  `harborrl/algorithms/dive_po/rewards/dual_stream.py`（生产默认）。
- [algorithms/dive_po_iclr2027_draft.md](algorithms/dive_po_iclr2027_draft.md)——
  DIVE-PO 论文草稿。
- [algorithms/lwm_offline_zh.md](algorithms/lwm_offline_zh.md)——LWM（Latent
  World Model）离线验证:自包含的设计原理、使用指南与验证结果（实现位于
  Slime 侧 `slime/slime/world_model/`）。

## 使用

- [harnesses/README.md](harnesses/README.md)——harness 选择与注册。
- [evaluation/README.md](evaluation/README.md)——评测工具、SETA fixed12 协议与
  SWE-bench 产物导出。
- [../deploy/README.md](../deploy/README.md)——worker 运行时、部署资源、运维工具与
  本地 RJob 的职责边界。
- [performance/seta_training_efficiency_zh.md](performance/seta_training_efficiency_zh.md)——
  SETA 训练耗时、同 Pod 私有 worker 的长窗口提速结果、GPU 等待瓶颈、
  slime trace 与高吞吐执行 profile。
- [../examples/README.md](../examples/README.md)——训练与验证入口清单。

## 运维

- [operations/README.md](operations/README.md)——通用运维文档导航。
- [operations/checkpoint-wandb.md](operations/checkpoint-wandb.md)——
  tracker 感知的 checkpoint 清理、磁盘满非致命策略与 W&B offline 同步。
- `records/operations/`——仅保留本地的 RJob、node53、Docker worker 和评测现场记录，
  入口见其 `README.md`。
