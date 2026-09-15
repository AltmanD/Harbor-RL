# GLM-5.1 支持说明与使用指南

本文说明 HarborRL 如何支持 GLM-5.1(`zai-org/GLM-5.1`,`model_type=glm_moe_dsa`)
进行 colocate RL 训练，以及如何启动一次 smoke。现场经验、故障史与 TODO 见
`local/records/operations/rjob/glm51-smoke.md`；逐日状态见
`local/records/report/glm51_smoke_status_2026-09-07.md`。

## 模型与挑战

GLM-5.1 是 ~744B 的稀疏 MoE(256 专家 + 1 共享专家，3 dense + 75 MoE 层),
MLA 注意力 + DSA(Dynamic Sparse Attention）索引器，BF16 权重合计约 1.45 TB。
在 140 GiB H200 上做 colocate（训练与推理共用 GPU）的挑战：

- 镜像的 Transformers(4.57）与 SGLang(0.5.5）都早于 GLM-5.1，不认识
  `glm_moe_dsa`；
- 训练侧需要 `megatron.bridge`，镜像未安装；
- 每卡权重分片大（TP16 下 ~90.8 GiB)，训练与引擎的显存交接必须严丝合缝。

## 支持方式（四层）

### 1. 引擎层：独立 venv 的 SGLang 子进程

rollout HTTP server 不用镜像里的旧 SGLang，而是以子进程方式跑在预制 venv
`glm51-stack`(SGLang 0.5.10 + Transformers 5.3，原生支持
`GlmMoeDsaForCausalLM`）里，由 `LIGHTRL_SGLANG_SERVER_PYTHON` 选择
（`slime/slime/backends/sglang_utils/sglang_engine.py` 的
`launch_server_process` 分支；`ServerArgs` 序列化后按目标版本字段过滤）。
训练侧与 Ray actor 仍留在镜像栈上。

### 2. 镜像兼容层：`sitecustomize.py`

仓库根的 [`sitecustomize.py`](../../sitecustomize.py) 随 `PYTHONPATH` 自动生效，
只做三件幂等的事（新版本库存在时自动让路）：

- 给 Transformers 4.57 注册 `glm_moe_dsa`/`glm51` 的最小 `AutoConfig`;
- 把 `TokenizersBackend` 映射到 fast-tokenizer 兼容子类（转换 list 型
  `extra_special_tokens`);
- 把 `LIGHTRL_MEGATRON_BRIDGE_OVERLAY` 追加到 `sys.path`,让 `megatron.bridge`
  从 GPFS overlay 解析（namespace 包不能用 `__path__.append`，会被重算丢弃）。

### 3. 训练层：Megatron-Bridge overlay + GLMMoEDSABridge

`megatron.bridge` 来自 GPFS overlay(`runtime/megatron-bridge-fzyzcjy-35b4ebfc/`,
fzyzcjy fork，版本钉在 `legacy-requirements.txt`)。模型适配在
`slime/slime_plugins/megatron_bridge/glm_moe_dsa.py`：把上游 NVIDIA GLM5Bridge
移植到该 fork 的 API,**全部结构参数从 HF config 推导**(3 dense + 75 MoE、
MLA nope/rope 维度拆分、DSA 索引器维度、routing scaling 2.5、MTP 关闭）,
并提供完整的 HF↔Megatron 权重映射表。默认 CPU 侧构造参数
（`GLM_CPU_INITIALIZATION=1`）并跳过无用的随机初始化
（`GLM_PERFORM_INITIALIZATION=0`)，权重随后逐 tensor 流式上卡。
模型结构 CLI 参数在 `slime/scripts/models/glm51-744B.sh`（与 HF config 对齐）。

### 4. colocate 显存编舞与 launcher 加固

训练与引擎分时复用每卡 140 GiB，依赖（均在 `slime/` 上游机制之上）:

- `--colocate` 默认开 `offload_train`/`offload_rollout`:Megatron 用
  torch-memory-saver sleep 到 CPU backup，引擎侧用
  `SGLANG_ENABLE_WEIGHTS_CPU_BACKUP=1` 保证 release 真正 unmap（否则 200 OK
  但权重不释放）——两侧永不同时满载；
- 权重 update 从 CPU backup 流式读（~2 GiB 桶）,LoRA 逐模块原地 merge，不需要
  双份权重驻留；`recompute=full` + `mbs=1` 把激活压到极小；
- `RolloutManager.offload()` 带重试与显式 `SLIME_RELEASE_MEMORY_TAGS`;
  launcher 对 `ray job status` 做重试，只有明确终态才 cleanup，dashboard
  瞬断不再误杀集群；`SGLANG_*`/`TORCH_NCCL_*` 经 runtime
  env(`EXTRA_ENV_PASSTHROUGH_JSON`)透传到所有 actor。

16 GPU(TP16）是 colocate 训练的下限（12 卡无论如何放不下）,32 GPU(TP32）
是舒适区；对应入口见下。

## 使用

### 前置条件

- 4 个（16 GPU）或 8 个（32 GPU)RJob replica 全部 Running 且拿到 IP;
  每个 replica 4×H200、内存请求 ≥900 GiB(backup 约占 730 GiB);
- rank0 上 SETA pool 已就绪（`curl -sf http://127.0.0.1:18081/readyz`);
- Ray 集群包含全部节点（`ray status` 看到 4/8 nodes、16/32 GPUs);
  worker 加入时必须 `export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"`。

详细 RJob/Ray 操作序列（含排队与故障处理）见
`local/records/report/glm51_smoke_status_2026-09-07.md` §3。

### 启动

```bash
# 16 GPU(4 replica,TP16;默认 GLM_GPU_PROFILE=16)
NUM_ROLLOUT=4 RUN_ID=glm51-nonasync-smoke-<date>-rN \
  nohup bash slime/scripts/run-glm51-seta-dapo-16gpu.sh \
  > runs/<RUN_ID>.launch.log 2>&1 &

# 32 GPU(8 replica,TP32)
NUM_ROLLOUT=4 RUN_ID=glm51-nonasync-smoke-<date>-rN \
  nohup bash slime/scripts/run-glm51-seta-dapo-32gpu.sh \
  > runs/<RUN_ID>.launch.log 2>&1 &
```

关键可调环境变量（默认值已按 smoke 调优，均在
`local/rjob/run_glm51_nonasync_smoke.sh` 里有注释）:

| 变量 | 默认 | 说明 |
|---|---|---|
| `NUM_ROLLOUT` | 10 | train step 数；首轮验收用 4 |
| `SGLANG_MEM_FRACTION_STATIC` | 0.68 | 控制引擎 KV 池，给 Megatron 留 ~8 GiB 余量 |
| `SGLANG_ENABLE_WEIGHTS_CPU_BACKUP` | 1 | 必须；否则 release 不真正释放权重 |
| `SGLANG_DISABLE_CUDA_GRAPH` | 1 | 省 capture 的瞬时显存 |
| `SLIME_RELEASE_MEMORY_TAGS` | weights | 16 卡不要加 kv_cache（会踩 resume 崩溃） |
| `SGLANG_ATTENTION_BACKEND` | 空 | 设为 `torch_native` 可兜底全量 release 的 resume |
| `GLM_CPU_INITIALIZATION` | 1 | Megatron 参数 CPU 构造 |
| `GLM_LORA_RANK` / `GLM_LORA_ALPHA` | 16/32 | LoRA 配置 |

### 验收

盯 `runs/testing/debug/<RUN_ID>/logs/train.log`，里程碑顺序：
`Load weight end` → server healthy → `/release_memory_occupation 200 OK` →
Megatron 初始化完成 → rollout 生成 → 连续 `global_step`。首轮通过标准：
4 个连续 `global_step` + Ray job `SUCCEEDED`；随后 `NUM_ROLLOUT=10` 做完整
smoke。失败时先看 run 目录 `logs/mirror/` 下的浓缩错误文件，再对照
`local/records/operations/rjob/glm51-smoke.md` 的故障史。
