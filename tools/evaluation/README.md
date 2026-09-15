# tools/evaluation — 通用评估工具集

推荐从仓库根目录使用包入口：

```bash
python3 -m tools.evaluation --help
```

安装项目后也可使用 `lightrl-eval --help`。历史的
`python3 tools/evaluation/eval_cli.py ...` 路径继续支持，便于直接运行已有
作业脚本。

把"起服务 → 生成 harness 配置 → 跑评测 → 回收归一化结果 → 多模型对比"
这条链路从一次性脚本重构成通用工具。分两层:

- **适配层** `harborrl/harnesses/eval/`:每种评测 harness 一个适配器
  (Harbor `terminus-2`、Harbor `claude-code`、slime `eval_only`/camel-agent),
  负责生成原生配置、给出启动命令、轮询进度、归一化结果。
- **工具层** `tools/evaluation/`(本目录):YAML 配置、managed SGLang 生命周期、
  单模型 runner、多 ckpt 批量、对比报告,入口是 `eval_cli.py`。

评测默认产物写入仓库 `runs/evaluation/<job_name>/`；设置 `RUNS_ROOT` 或
配置中的 `output_dir` 可覆盖。一次性路径、节点名、代理地址等一律进配置或
`site/` profile,不进代码。

## 快速开始(单 ckpt 单 harness)

```bash
cp tools/evaluation/configs/tb21_terminus2.example.yaml /path/to/my-eval.yaml
# 编辑 dataset.path / serving.model_path / output_dir 等
python3 -m tools.evaluation run --config /path/to/my-eval.yaml --dry-run   # 先看将执行什么
python3 -m tools.evaluation run --config /path/to/my-eval.yaml
```

`--dry-run` 只打印生成的 harness 配置、启动命令、进程环境 overlay 和
(managed 模式下)SGLang 启动命令,不执行。

## 目录布局

通用编排代码与 benchmark 实现分开，benchmark 脚本按数据集/评测协议归类：

```text
tools/evaluation/
├── core/                 # 配置、serving 生命周期、runner、batch、report
├── benchmarks/
│   ├── seta/             # SETA fixed-suite 构建、审计、paired gate
│   ├── safety/           # AgentSafetyBench、AgentHarm、ShieldAgent
│   ├── swebench/         # SWE-bench Verified 官方 harness
│   └── world_model/      # world-model probes 与 candidate-set eval
├── utilities/            # 与单一 benchmark 无关的工具
├── configs/              # 通用/ harness 配置模板
├── site/                 # 节点与站点相关 profile
└── runtime_overlays/     # 运行时注入 overlay
```

新脚本请放入对应的 `benchmarks/<name>/`，不在根目录新增
benchmark-specific 文件。

常用 benchmark 入口：

| Benchmark | 入口 |
| --- | --- |
| SETA | `build_seta_fixed_eval.py`, `audit_seta_fixed48_run.py`, `compare_seta_fixed_eval.py`（均位于 `benchmarks/seta/`） |
| Safety | `bash tools/evaluation/benchmarks/safety/run_safety_official_eval.sh` |
| SWE-bench | `bash tools/evaluation/benchmarks/swebench/run_swebench_verified_official_harness.sh` |
| World Model | `bash tools/evaluation/benchmarks/world_model/run_world_model_<probe>.sh` |

## 评估一个 ckpt

1. 选一个示例配置(`configs/` 下按 harness 分),复制后改数据集和模型字段。
   不填写 `output_dir` 时自动使用 `runs/evaluation/<job_name>/`。
2. `serving.mode: managed` 时工具层自动起本地 SGLang 并轮询
   `/v1/models` 就绪;`external` 时使用已有端点(填 `api_base`)。
3. 跑 `eval_cli.py run --config <cfg>`。产物:
   - `<output_dir>/<job_name>.config.json`(harness 原生配置)
   - `<output_dir>/<job_name>.log`(评测进程日志)
   - `<output_dir>/eval_result.json`(归一化结果,跨 harness 同构)
4. 单题冒烟用 `smoke` 子命令(自动 concurrency=1、max_retries=0):

```bash
python3 -m tools.evaluation smoke --config <cfg> --task <task-name>
```

命令行覆盖配置(可重复):`--set run.concurrency=8 --set serving.port=30001`。

## 批量评估多个 ckpt

参考 `configs/batch.example.yaml`:`defaults` 放公共配置,`serving` 放共享
managed serving 参数,`models` 列表逐个给 `name` / `model_path` /
`model_name`(可选 `overrides` 做深合并覆盖)。每个模型落在
`<defaults.output_dir>/<name>/` 子目录,互不覆盖。

```bash
python3 -m tools.evaluation batch --config /path/to/batch.yaml --dry-run
python3 -m tools.evaluation batch --config /path/to/batch.yaml
```

managed 模式下批量会自动逐模型切换 SGLang(stop → start → 等待就绪);
单个模型失败会被记录,批量继续;结尾按 `report.output` 自动产出对比报告。

## 结果对比(report)

```bash
python3 -m tools.evaluation report \
  --results '/path/to/eval-runs/*/eval_result.json' \
  --output /path/to/compare
```

输出 markdown 表格(model / harness / pass@1 / mean_reward / completed /
errored / top exceptions)到终端,并写 `compare.md` 与 `compare.csv`。

## 新增一个 harness

1. 在 `harborrl/harnesses/eval/` 加一个文件,实现
   `BaseEvalHarness`(`build_config` / `launch_command` / `progress` /
   `collect`,同步方法即可)。
2. 在 `harborrl/harnesses/eval/__init__.py` 的注册表加一行
   (alias → canonical → 模块/类名)。注册表保持惰性 import,
   不会给 `harborrl` 引入重依赖。
3. 在 `tools/evaluation/configs/` 加一份示例 YAML,在
   `tests/harborrl/test_eval_harnesses.py` 加 build_config/collect 单测。

## 配置字段参考

| 字段 | 说明 | 默认 |
| --- | --- | --- |
| `harness` | `terminus-2` / `claude-code` / `camel-agent` | `terminus-2` |
| `job_name` | 评测 job 名(Harbor job 目录名) | 必填 |
| `output_dir` | Harbor `jobs_dir`;归一化结果也写在这里 | `runs/evaluation/<job_name>` |
| `dataset.path` | 数据集路径(Harbor tasks 目录或 prompt jsonl) | 必填 |
| `dataset.task_names` | 只跑指定题目;`null` 跑全量 | `null` |
| `run.n_attempts` | Harbor `n_attempts` | 1 |
| `run.concurrency` | Harbor `n_concurrent_trials` | 4 |
| `run.max_retries` | Harbor `retry.max_retries` | 1 |
| `run.timeout_multiplier` | Harbor `timeout_multiplier` | 1.0 |
| `run.max_input_tokens` / `run.max_output_tokens` | agent `model_info` | 8192 / 8192 |
| `environment` | 透传进任务容器/agent 的环境变量 | `{}` |
| `serving.mode` | `external`(已有端点)/ `managed`(工具层起 SGLang) | `external` |
| `serving.api_base` | OpenAI 兼容端点;external 必填 | `""` |
| `serving.model_path` / `serving.model_name` | HF ckpt 路径 / served 名 | `""` |
| `serving.port` / `gpu_ids` / `tp_size` / `mem_fraction` | managed SGLang 参数 | 30000 / `[]` / 1 / 0.70 |
| `serving.health_timeout_s` | 等待 `/v1/models` 就绪秒数 | 900 |
| `serving.command_template` / `launcher` / `tmux_session` | 启动命令模板(占位符 `{model_path}` `{served_name}` `{port}` `{tp_size}` `{gpu_ids}` `{mem_fraction}`)/ `nohup`\|`tmux` / tmux 会话名 | 见 `core/serving.py` |
| `extra.harbor_bin` | harbor CLI 路径 | `harbor` |
| `extra.mounts` / `extra.extra_docker_compose` | 原样进 Harbor config | 空 |
| `extra.process_env` | 追加给评测进程自身的环境 | `{}` |
| `extra.slime_root` / `run_dir` / `hf_checkpoint` / `load` / `rollout_config` / `slime_args` | camel-agent(slime)专有 | 见示例 |

YAML 支持 `${VAR}` 与 `${VAR:-default}` 环境变量展开;`--set a.b=c`
按点路径覆盖,值按 YAML 标量解析。

## site/ 隔离说明

brainctl RJob、model relay、节点级 docker 外部网络、正向代理、离线镜像
导入链等都是站点/节点强相关的,**不进通用层**。站点差异用
`site/` 下的 profile(示例:`site/remote_relay.example.yaml`)描述;
需要远端执行时用 ssh / systemd-run 把 runner 包装到目标节点。
详见 `site/README.md`。

## 备注

- SWE-bench 官方格式导出库在 `tools/evaluation/benchmarks/swebench/report.py`
  (被 rollout 管线 `misc/rollout_log.py` 在设置了 `SWEBENCH_RESULTS_DIR` 时
  import,不是离线评估 CLI,勿混淆)。用户侧的评测配方入口见
  `examples/evaluation/`。
- camel-agent 适配的是 slime `eval_only` 重量级运行时(自行拉起训练/推理
  引擎),通常需配合集群启动脚本;`serving` 段对它不生效。
- Harbor `claude-code` agent 的 kwargs/env 参数面随 Harbor 版本变化,
  `agent_kwargs` / `agent_env` 请按实际 Harbor 版本核对。
- Harbor 运行要求 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `NO_PROXY` 同时
  出现在 harbor 进程自身环境(`agents[].env` 只进容器)——适配器的
  `launch_command` 已处理。
