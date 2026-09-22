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
- 严格 schema 2 配置、原生 `harborrl train` dispatch、run/source/catalog 冻结和
  doctor 预检；真实训练仍要求全部 doctor 门禁通过。
- Slime 原生 group hook、trajectory → turn 训练边界、预计算 advantage/token
  weight、显式 padding 标记、custom clipped loss 和权重版本发布联动。

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
python -m pytest -q tests/harborrl/test_native_training.py
```

`offline-smoke` 固定合成 1-turn / 3-turn 两条轨迹，分数 0 / 1，组均值应为 0.5。
输出包含 `traces/`、逐 attempt 的 session/result/reward/evaluation/IR、`export.json`
与 `acceptance.json`。所有这些是合成证据，标为 `CONTRACT_READY`，生产 exporter 默认拒绝。
版本推进也只是屏障协议验证，没有模型更新。

schema 2 采用严格固定字段：`execution`、`tasks`、`harness`、`harbor`、`gateway`、
`model`、`training`、`sampling`、`deployment`、`output`。本地路径相对配置文件解析；
override 只接受 `section.key=value`，未知字段在加载期失败。真实 lock 生成前，可参考
`tests/harborrl/test_native_training.py` 中的最小目录/catalog/profile 结构；不要把该
fixture 当作生产任务或 CLI 版本验收。

2026-09-21 CPU 回归：native 契约、Gateway、生命周期、GPU 布局与新增训练边界
**87 passed**；`test_native_training.py` 同时执行 Slime launcher 的 native dry-run。
GPU 机复核：8×H200 上上述 GPU/native 集合 **88 passed**，Python 3.13 下
`test_native_training.py` **3 passed**。独立 SGLang 0.4.5 服务（Python 3.12、
`--weight-version 0`）使用 Qwen2.5-0.5B 完成 token/text/EOS 与 raw logprob 审计，
HF 对照最大绝对误差 **0.006644**，复现入口为
`tools/verification/sglang_semantic_probe.py`。本次同时确认系统 Python 3.10 的
FlashInfer sampler 会崩溃，必须使用 `lightrft_py312` 训练环境。两台 worker 可使用共享独立环境
`.venv-harbor-0.23`（Python 3.12 + Harbor 0.23.0），probe 已通过。
真实 Docker
debug 已确认三层环境限制：默认 `/data` 上的 overlay2 返回 `EINVAL`；改用
`/tmp` + `vfs` 后 Harbor 可创建 trial、构建镜像，但默认 compose 网络因缺少
`CAP_NET_ADMIN` 创建失败；临时 `network_mode:none` 可绕过网络，最终 runc 仍因
`/sys/fs/cgroup/cpuset` 只读无法创建子 cgroup，且 remount 返回 busy/保持 RO。
因此 GPU Pod 不应承载 Docker task，也不应尝试嵌套 Docker。部署边界固定为：
GPU Pod 只运行 Ray/Megatron、SGLang、Messages Gateway 与调度；两台共享存储
CPU worker 运行 Harbor 0.23 Runner 和 Docker task。0.5 的 `luyd-dev` 与
`dev` worker 均有可用 daemon 与空闲 pool 服务，native 侧需接入同等外部 worker
拓扑，而不是在 GPU Pod 内补 Docker。
基础 Conda Python 3.13 仅用于纯 Python 回归，训练/serving 依赖位于
`lightrft_py312`；不要把基础环境的 import 缺失解释为功能回归。

故障注入另覆盖进程崩溃、超时强制终止、错误 trial 事件、凭据复用、重启抢占、
辅助模型请求、缺消费证据、错误版本、错误 token/logprob 与不完整 groups。

## 剩余开发与真实验收

2026-09-22，8×H200 真实链路已经完成自然 reward 0/1 对比、非零
advantage/gradient/PG loss、policy version 发布、checkpoint 参数变化、
updated checkpoint 重载和清理验收；Native MVP 证据闭环。以下不是 MVP 阻塞，
而是从单次可复现实验走向长期生产训练时仍需补齐的边界：

1. 真实 lock catalog/checkpoint/tokenizer/template 的填入与跨机器路径核验；恢复
   台账仍只覆盖不可变产物，不支持半条 Agent session 续跑。
2. Slime/Megatron 张量传递、预计算 advantage、token 权重 reduction、DP/padding
   与真实 optimizer 梯度/参数变化已经通过 MVP 验收；仍需扩大任务与多次更新
   的稳定性覆盖。
3. 固定真实 CLI、镜像、模型/tokenizer/template、serving 版本；在两台外部 CPU worker
   上验证 Harbor Runner、Docker task、共享路径、Gateway 可达性、辅助请求关闭、工具轮转、
   最后响应消费与 CLI session 格式。GPU Pod 不承载 Docker task。
4. serving 输出 token 与文本/stop/EOS 对账、生成时版本事实和 Qwen3-8B 的
   raw logprob 语义已经通过 GPU 探针；仍需把该探针纳入例行发布门禁。
5. 正常预算截断分类、结果半写恢复、资源补偿执行与核销。当前异常 trial 保守拒绝；
   `cleanup-required.json` 是人工检查线索，尚不是自动容器回收器。
6. 原生真实 task 的完整分差组、非零梯度/参数更新、全部副本同步、新版本采样、
   checkpoint 加载已在 MVP 单步验收中通过；多轮持续训练与故障恢复仍需扩展。
7. 多副本/colocate 和真实中断回归。Terminus2/Codex/OpenHands 与 DAPO 按后续方案扩展。

后续扩展前，先保持 doctor、语义探针、清理检查和 native 回归作为发布门禁；
不要用系统 Python、GPU 内 Docker 或未锁定的任务/模型替代上述验收。
