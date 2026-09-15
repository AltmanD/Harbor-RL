# HarborRL 配置与运行配方

训练配置直接写在 `examples/training/` 的 recipe 脚本中。每个脚本包含模型、数据集、
算法、GPU 拓扑和 rollout 配置，环境变量可覆盖默认值；不再经过 Python CLI、配置组合
或插件 registry。

```bash
# 查看最终 Slime 参数，不启动训练
bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run

# 在 4-GPU 计算任务内启动完整训练（前台）
bash examples/training/train_qwen3_8b_seta_dapo.sh

# 显式后台启动
BACKGROUND=1 bash examples/training/train_qwen3_8b_seta_dapo.sh
```

调用链固定为：

```text
examples/training/<recipe>.sh
  -> harborrl/platform/slime_train.sh
  -> slime/train_async.py
```

`configs/rollout/` 只保留传给 rollout 的模型模板配置。站点地址、凭据和调度参数应通过
环境变量或被 Git 忽略的 `local/cluster/` 提供。

离线评测采用同样的 Slime 编排层，但入口不同：通用 benchmark 使用
`python3 -m tools.evaluation`，需要本地 Ray/worker 的完整配方位于
`examples/evaluation/`。SETA fixed12 的 4-GPU 一键配方为
`examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh`，它通过
`SLIME_ENTRYPOINT=slime/eval_only.py` 运行评测，不执行 actor checkpoint 更新。

部署相关变量按执行环境分层：公共 recipe 只读取 `WORKER_URLS`/
`WORKER_URLS_FILE` 等运行时变量；站点 RJob/DinD 提交和生命周期脚本保存在
被 Git 忽略的 `local/rjob/`（本地目录总览见 `local/README.md`），worker
运行时、资源和运维操作分别见
`deploy/workers/`、`deploy/runtime/` 与 `deploy/ops/`。

Python 侧的环境变量解析统一在 `harborrl/env.py`（`env_bool` /
`env_int` / `env_float` / `env_flag` 等）；该模块的 `ENV_VARS` 表是
rollout 域变量的集中声明（名称 → 含义），新增变量请在此登记。环境
（数据源）相关的判定集中在 `harborrl/environments/registry.py` 的
`EnvSpec` 表，无需再散落修改 if-else 分支。
