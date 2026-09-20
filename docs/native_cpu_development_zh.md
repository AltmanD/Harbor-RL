# Native Claude Code：CPU 开发阶段

本轮承接 2026-09-20 的 Claude Code 最小原生闭环方案，起点为
`696bfa0e`（IR v2、严格 reward、trajectory GRPO 导出），以及工作区尚未提交的
Messages Gateway、Runner、消费审计和协调器草稿。

## 已实现边界

- 原始 single-step / 单 Dockerfile task 静态检查，保留原始 verifier，不要求物化。
- 严格 Messages 文本/工具协议、缓冲 SSE、count-tokens 与 SGLang `/generate` adapter。
- attempt 独立凭据、生成记录、交付与原生 session 消费确认、不可变 seal、版本排空屏障。
- 隔离 Harbor 0.23.0 / Python ≥3.12 Runner，prepared/run 握手、事件身份和产物路径校验。
- 有界 slot 重试、新 attempt lineage、完整 group 导出；有效零分不作为重试理由。
- 子进程启动失败、崩溃、超时/取消的故障记录，异常退出产生资源检查记录。
- 独立 CPU HTTP → session audit → reward receipt → IR → GRPO export 验证入口。

Claude 控制变量同时注入原生 Agent 环境；这证明配置构造，不证明真实 CLI 已遵守。
SGLang logprob 默认 `unverified`，必须实际核对 raw-model 语义才允许启用审计标志。
SSE 当前缓冲完整生成后发送，尚不是逐 token 实时转发。

## 复现

只需 Python 与项目基础依赖；测试还需 `pytest`，旧 interactive 回归需 `pytest-asyncio`。
HTTP 测试需要允许监听本机回环端口。输出目录必须不存在，避免覆盖旧证据。

```bash
python -m harborrl.rollout.harbor_job inspect /path/to/original/task
python -m harborrl.rollout.harbor_job offline-smoke --output runs/native-cpu-check
python -m pytest -q tests/harborrl/test_native_contracts.py tests/harborrl/test_native_gateway.py tests/harborrl/test_native_lifecycle.py
```

`offline-smoke` 固定合成 1-turn / 3-turn 两条轨迹，分数 0 / 1，组均值应为 0.5。
输出包含 `traces/`、逐 attempt 的 session/result/reward/evaluation/IR、`export.json`
与 `acceptance.json`。所有这些是合成证据，标为 `CONTRACT_READY`，生产 exporter 默认拒绝。
版本推进也只是屏障协议验证，没有模型更新。

本轮组合回归结果：**122 passed / 10 skipped**。跳过项依赖未安装的 Torch 或
Terminal Bench；因此不算这些运行时的回归通过。新增 native 契约/HTTP/生命周期用例均通过，
新增模块 Ruff F 检查通过。

故障注入另覆盖进程崩溃、超时强制终止、错误 trial 事件、凭据复用、重启抢占、
辅助模型请求、缺消费证据、错误版本、错误 token/logprob 与不完整 groups。

## 剩余开发与真实验收

当前不是 M0–M5 已完成，也不能直接通过 `harborrl train` 启动 native 训练。
以下必须继续开发或联调，不能仅提供 GPU 后就视作已经具备：

1. schema 2 配置/lock catalog、单一训练入口、实际 batch 调度和恢复台账。
2. Slime/Megatron 张量传递、预计算 advantage、token 权重 reduction、DP/padding 和梯度对照。
   当前 exporter 是独立契约，尚未接入真实 optimizer。
3. 固定真实 CLI、镜像、模型/tokenizer/template、serving 版本；从实际容器验证端点、
   辅助请求关闭、工具轮转、最后响应消费与 CLI session 格式。示例测试版本不是验收 lock。
4. serving 输出 token 与文本/stop/EOS 对账，生成时版本事实及训练侧 logprob 数值核验。
5. 正常预算截断分类、结果半写恢复、资源补偿执行与核销。当前异常 trial 保守拒绝；
   `cleanup-required.json` 是人工检查线索，尚不是自动容器回收器。
6. 原生真实 task 的完整分差组、非零梯度/参数更新、全部副本同步、新版本采样、checkpoint 加载。
7. 多副本/colocate 和真实中断回归。Terminus2/Codex/OpenHands 与 DAPO 按后续方案扩展。

提供 GPU 后，先完成已合入 interactive 路径的机器基线检查，再进行真实 Claude/serving 探针；
依据探针修订 adapter 和 lock，之后接入训练。真实 Harbor、GPU 与 Hub 覆盖率本轮均未验收。
