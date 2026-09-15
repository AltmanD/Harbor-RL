# OpenClaw Terminal Latent World Model v2

当前实现把 SETA/tb2.1 turn 轨迹映射为 policy-hidden-conditioned latent transition，并用 action-conditioned AdaLN Transformer 预测环境反馈 latent。完整设计与命令见 [`docs/algorithms/lwm_offline_zh.md`](../../../docs/algorithms/lwm_offline_zh.md)。

## 主路径

```text
traj.json / records.jsonl / replay.pt
  -> TerminalTransition(h_t, a_t, o_t+1, h_t+1)
  -> policy hidden[prompt_end, action_span]
  -> source adapters + shared latent projector
  -> AdaLN predictor(state latent, action condition)
  -> predicted feedback latent
```

`modules.py` 中 action 不进入 self-attention token 序列，也不与 state 做特征拼接；默认
AdaLN Transformer（以及 `predictor_type=mlp` 的轻量 FiLM/AdaLN 对照）只把 action
映射为 shift/scale/residual gate。

## 模块

| 文件 | 作用 |
| --- | --- |
| `seta_dataset.py` | 统一读取 tb2.1 ATIF `trajectory.json`、SETA `traj.json`、records JSONL、replay `.pt` |
| `hidden_encoder.py` | 同一 causal forward 提取 prompt-end state 与 action-span hidden |
| `modules.py` | shared latent、AdaLN predictor、SIGReg、contrast/value loss |
| `replay_buffer.py` | 可选 DAPO world-model trajectory replay |
| `train_latent.py` | 端到端训练、预测、checkpoint |
| `stream_latent.py` | 流式（online-style）replay A/B：分 chunk 到达、等算力对照 |
| `mpc.py` / `plan_mpc.py` | 同一 state 上的候选 action latent one-step MPC |
| `metadata.py` | rollout 侧轻量 transition metadata |

旧的 `build_dataset.py -> cache_text_hidden.py -> train_probe.py -> evaluate_probe.py` 路径继续保留，用于 v1 artifact 和 Stage-A ablation。

## 快速运行

```bash
WM_USE_DAPO_REPLAY_BUFFER=1 \
WM_MAX_TRAJECTORIES=2 \
WM_MAX_TRANSITIONS=8 \
WM_EPOCHS=1 \
examples/training/world_model/train_seta_latent.sh
```

默认 hash hidden 只验证工程闭环。正式 policy hidden：

```bash
WM_ENCODER=hf-policy \
WM_HF_MODEL=/path/to/Qwen3-8B \
examples/training/world_model/train_seta_latent.sh
```

两个默认关闭的 option：

- `WM_BACKPROP_TO_LLM=1` / `--backprop-to-llm`：允许 latent loss 更新 HF policy backbone；
- `WM_USE_DAPO_REPLAY_BUFFER=1` / `--use-dapo-replay-buffer`：通过 replay 接口训练。

DAPO rollout 侧收集使用：

```text
--world-model-enable
--world-model-use-dapo-replay-buffer
--world-model-replay-buffer-size 4096
```

策略级 latent WM 对照可在 DAPO 命令中额外启用：

```text
--world-model-enable
--world-model-backprop-to-llm
--world-model-loss-coef 0.01
--world-model-target-provider-path package.module:function
```

target provider 在每个 rollout 为样本附加冻结 WM 的 target latent；hook
对 target detach，并从 policy response logits（或 provider 直接给出的
`wm_pred_latents`）计算 aux loss。`wm/loss` 与
`wm/policy_gradient_path` 会进入训练指标，便于与同 seed 的 DAPO-only
对照统计 fresh transitions、optimizer steps 和 wall-clock。

正式数据应来自 SETA `tool_calls[].result` 的 stdout/stderr/exit code；
`capture-pane` 屏幕文本只保留为兼容旧 tb2.1 ATIF 的离线格式。

## 默认安全边界

- backbone 梯度默认关闭；feedback/next-state target 始终 detached，但与 current branch 共享 policy checkpoint，并非独立 EMA teacher。
- replay 默认关闭；开启后成功和失败 transition 都会入库。
- `world_model_loss_coef=0` 时不改变 DAPO/GRPO loss。
- value loss 默认关闭且不创建 value head，也不接管 GRPO advantage。
- hash smoke、低 training loss 或 SIGReg 都不能证明在线候选筛选有效。
