# 仓库整理：当前交付状态

统一入口为 `python -m harborrl.cli`（安装后为 `harborrl`）。本轮是渐进迁移，尚未完成实施方案全部 P0–P6 验收。

## 安装与入口

CPU 静态检查只需 `pip install -e .`；开发检查使用 `pip install -e '.[dev]'`。
Worker、Camel、Claude 与训练角色分别参考 pyproject 的 extras。训练 extras 不保证 CUDA、Megatron、SGLang 的 ABI 配套，Claude 还需单独安装 CLI。

本地已获取的固定版本任务可独立检查，无需 Torch、Ray 或 Docker：

```bash
harborrl hub inspect --tasks-dir /path/to/tasks --dataset dataset-name --revision source-commit --run-dir runs/hub/local
harborrl hub report --run-dir runs/hub/local
```

每次检查保留独立 JSONL；digest 标识内容，revision 由调用者提供。静态检查不会产生 runnable catalog，也不会把 NEEDS_PROBE 当作执行成功。下载与 Docker 对照探测仍使用已有工具 `harborrl.data.harbor`、`harborrl.data.harbor.probe`。

配置训练需显式设置 `HARBORRL_CATALOG`、`HARBORRL_MODEL_PATH`、`HARBORRL_REFERENCE_PATH`、`HARBOR_LOGPROB_SOURCE`、`HARBORRL_WORKER_URLS`；catalog 必须由已有探测和 materializer 验证生成。模型 preset 必须与 checkpoint 匹配；reference 是初始 Megatron 权重，不表示恢复 optimizer 状态。

```bash
harborrl train --config configs/train_interactive_smoke.yaml --dry-run
harborrl doctor --config configs/train_interactive_smoke.yaml
harborrl train --config configs/train_interactive_smoke.yaml
harborrl train --config configs/train_interactive_smoke.yaml --set harness.name=claude_code_cli --dry-run
harborrl eval --config tools/evaluation/configs/tb21_terminus2.example.yaml --dry-run
```

配置相对路径以 YAML 所在目录解析，环境仅用于显式插值，CLI `--set` 优先；未知配置拒绝执行。dry-run 不启动 shell、Ray、Docker 或下载。doctor 当前检查本地依赖和路径，尚不证明 worker 连通性、模型模板或 GPU ABI 可用。训练时记录 launch.json、commit、dirty 状态和已跟踪源码补丁；未跟踪源码不在补丁中，正式复现实验应先提交代码。

Python 入口控制已迁移的任务、模型、harness 和资源字段，调用已有 Slime shell。其余 backend tuning 仍使用 shell 默认值，完整命令生成迁移尚未完成。旧环境变量实验入口保留，`examples/training/train_harbor_terminal.sh --config ...` 委托统一 CLI。新入口跳过全局进程清理，仍需独占相应 Ray 端口与 GPU 资源。

## 模块边界与后续验收

- `tasks/metadata.py` 是不依赖 Slime 的 task helper；原 sample_builder 保留兼容导入。
- TurnClient 公开模板与截断接口；SGLang 保留私有名称兼容别名。
- harness descriptor 声明可忽略的共享构造参数，未知参数和必需参数缺失显式报错。
- `_build_samples` 仍被 Slime exporter 使用，旧 trajectory store 仍有消费者，未删除。
- Claude SSE、generation metadata 和 MCP id 修复纳入契约回归；标题辅助请求识别仍是版本相关过渡实现。

尚未验收：固定版本 GPU Camel/Claude 更新与同步、完整 policy gateway/native producer、Hub 下载/构建断点续跑和跨 dataset 扩量、统一 eval/train manifest、训练后配对评测、完整 shell 编排迁移。native 不能作为可运行训练默认值；本轮不声称完成这些阶段。

## 本次检查记录

定向回归：60 passed、10 skipped（跳过项需要当前环境缺失的可选依赖）。Python 编译、shell 语法和 git diff whitespace 检查通过。全量 `tests/harborrl` 收集被 6 项错误阻断，涉及缺少 Torch 和历史测试的已迁移文件路径；未执行真实 GPU、Docker 或 native 训练验收。

GPU 回归准备发现并补齐的入口缺口：`backend_options` 为有限白名单的 Slime 过渡配置，用于显式指定采样预算、输出、CLI 与观测 hook；不能覆盖顶层模型或 harness 字段。Megatron doctor 接受仓库随附源码。参数变化观测器为 `harborrl.platform.parameter_probe.before_step`，输出到本次 RUN_DIR，避免依赖历史实验脚本。真实三机验收结果另存实验记录。

2026-09-17 三机真实回归完成：Camel 48/48 RL_READY、5 次非零更新；Claude 24/24 RL_READY、6 step 中 4 次非零更新，67 条 MCP 工具记录无错误。两者权重版本 1→2→3、model-only checkpoint iteration 2、Ray SUCCEEDED，worker 租约和 GPU 均释放。Camel 暴露旧 Ray CLI 成功状态解析错误，修复后 Claude 完整入口退出 0。GPU 定向回归最终 59 passed。详细证据保存在 `runs/cleanup-regression-20260917/`；这补齐 interactive 真实训练回归，不表示 native 已验收。
