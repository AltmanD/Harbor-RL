# 评估配方入口(Evaluation Recipes)

通用 benchmark 命令统一推荐 `python3 -m tools.evaluation ...`；需要本地
Ray、Slime 和环境 worker 的重量级配方，使用本目录提供的启动脚本。

本目录是**离线评估的用户入口**:给出典型评测场景的配方(配置 +
命令)。评测机制本身(适配层、runner、CLI)在
[`tools/evaluation/`](../../tools/evaluation/README.md),
harness 适配器在 `harborrl/harnesses/eval/`;本目录不放代码,
只放"怎么跑"的配方,与 `examples/training/`(训练配方)对称。

三个目录的分工:

| 位置 | 角色 |
|---|---|
| `examples/evaluation/`(本目录) | 用户配方:选哪个 harness、配什么参数、跑哪条命令 |
| `tools/evaluation/` | 通用评估工具集:`eval_cli.py`、core/、configs/、site/ |
| `harborrl/harnesses/eval/` | harness 适配层:terminus-2 / claude-code-cli / camel-agent |

## 配方 1:Terminal-Bench 风格 benchmark(terminus-2)

Harbor + terminus-2,适用于 terminal-bench-2.x 这类 docker 任务集:

```bash
cp tools/evaluation/configs/tb21_terminus2.example.yaml my-tb-eval.yaml
# 编辑 dataset.path / serving.model_path / serving.model_name / output_dir
python3 -m tools.evaluation run --config my-tb-eval.yaml --dry-run
python3 -m tools.evaluation run --config my-tb-eval.yaml
```

先单题冒烟再全量:

```bash
python3 -m tools.evaluation smoke --config my-tb-eval.yaml --task <task-name>
```

常用覆盖(不改配置文件):

```bash
python3 -m tools.evaluation run --config my-tb-eval.yaml \
  --set serving.model_path=/path/to/ckpt --set serving.model_name=my-ckpt \
  --set run.concurrency=16 --set run.max_input_tokens=32768
```

## 配方 2:批量评估多个 ckpt 并对比

```bash
cp tools/evaluation/configs/batch.example.yaml my-batch.yaml
# 在 models: 列表里逐个填 model_path / model_name
python3 -m tools.evaluation batch --config my-batch.yaml
python3 -m tools.evaluation report \
  --results "my-batch-output/*/eval_result.json" --output my-batch-output/compare
```

产物:`compare.md`(模型 × pass@1 / mean_reward / 异常分布 对比表)+
`compare.csv`。

## 配方 3:HarborRL 自研链路(camel-agent / SETA)

走 slime `eval_only` 重量级运行时(自行拉起推理引擎,`serving` 段不生效):

```bash
cp tools/evaluation/configs/seta_camel.example.yaml my-seta-eval.yaml
# 编辑 extra.slime_root / extra.slime_args(--hf-checkpoint、--load、--prompt-data 等)
python3 -m tools.evaluation run --config my-seta-eval.yaml
```

### 配方 3a:4-GPU 一键 SETA fixed12 + Qwen3-8B + camel-agent

以下脚本复用 `examples/training/train_qwen3_8b_seta_dapo.sh` 的 Ray/Slime
启动链路，但把入口切换为 `slime/eval_only.py`，固定评测
`seta_fixed12_score_v1` 的 12 个 held-out 任务。默认使用 2 张 actor GPU、2
张 rollout GPU、TP=2，并保存完整轨迹。

运行环境需已安装项目依赖（尤其是 `PyYAML`、Ray、CUDA/sglang 和 Docker）。

先确认当前机器上的 SETA worker 已在 `WORKER_URLS` 指定的地址提供
`/healthz`（默认 `http://127.0.0.1:18081`），再执行：

```bash
# 检查路径、参数和最终 Slime 命令，不启动 Ray/worker
bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh --dry-run

# 正式运行（启动器会清理本机 Ray/SGLang，因此需要显式确认）
CONFIRM_LOCAL_CLEANUP=1 \
BACKGROUND=0 \
  bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh
```

在私有 DinD RJob 中手动运行时，先从开发机提交并保持 worker 容器：

```bash
export RUN_ID=lightrl-seta-fixed12-dind-$(date +%Y%m%d-%H%M%S)
RJOB_NAME="${RUN_ID}" RJOB_GPU=4 RJOB_CPU=50 RJOB_MEMORY=800000 \
bash local/rjob/start_fixed12_dind.sh
```

该命令会启动本 Pod 独立的 dockerd 和 SETA pool (`127.0.0.1:18081`)，然后
保持容器运行。另开终端找到 Replica 并进入：

```bash
brainctl -n ailab-narmodel get replica \
  -l rjob.brainpp.cn/rjob-name="$RUN_ID" -o wide
brainctl -n ailab-narmodel exec replica/<REPLICA> -- bash
```

在容器内执行：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL
source /run/lightrl-dind/${RUN_ID:0:32}/worker.env
CONFIRM_LOCAL_CLEANUP=1 \
BACKGROUND=0 \
  bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh
```

`worker.env` 同时提供私有 Docker 的 `DOCKER_HOST` 和 `WORKER_URLS`；evaluation
脚本不会再次启动或重启 pool，只做 `/healthz` 检查并使用该本地 worker。退出评估后，
停止 RJob 即可释放 dockerd 和 GPU。

模型默认从 `/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B` 和
`/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B_torch_dist` 读取；可通过
`HF_CKPT`、`REF_LOAD`、`WORKER_URLS`、`RUN_ID` 覆盖。默认后台运行，日志在
`runs/evaluation/<RUN_ID>/launcher.log`，评测汇总在
`runs/evaluation/<RUN_ID>/evaluations/seta/step_0000/summary.json`，逐题轨迹在
`runs/evaluation/<RUN_ID>/trajectories/`。脚本不会自动启动或重启共享 node53
worker；worker 生命周期请按
[`rjob/seta-worker.md`](../../docs/records/operations/rjob/seta-worker.md)
管理。若 worker 已由其他方式启动，可用 `SKIP_WORKER_HEALTHCHECK=1` 跳过本地
预检。

## 配方 4:Claude Code CLI 作为 agent

```bash
cp tools/evaluation/configs/tb21_claude_code.example.yaml my-cc-eval.yaml
# agent_kwargs / agent_env 需按所用 Harbor 版本核对(见 tools/evaluation/README 备注)
python3 -m tools.evaluation run --config my-cc-eval.yaml
```

## 站点差异怎么处理

节点/集群强相关的配置(代理、docker 网络覆盖、relay、远端执行)一律
不进通用配置,见 `tools/evaluation/site/` 的说明与 profile 示例。
