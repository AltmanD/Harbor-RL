# 多数据集接入与并行 Rollout 完整训练方案（2026-09-23）

## 0. 目标

在已验收的 native/gateway 训练链路（`feat/harborrl-mvp`，commit `999513bb`，
run `20260923-114103-ab5abb9c`）基础上：

1. 把实际接入的 Hub 任务从 2 个数据集扩展到 **12–16 个任务、8–12 个数据集**；
2. 将开发机 1、2 的 rollout 并行度从 4+4 提升到 **8+8 → 12+12（上限 16+16）**，
   把训练等待占比（当前 53–68%）压到 **35% 以下**；
3. 完成一次 6 步、192 条真实轨迹的多数据集 GRPO 训练，保持既有验收标准
   （组内奖励方差 → 非零梯度 → 真实权重变化 → 版本链推进）。

## 1. 现状与瓶颈

### 1.1 已验证基线

| 项 | 当前值 |
|---|---|
| 链路 | gateway(SGLang TP4, GPU4-7) → 2 台 CPU worker → GRPO(Megatron TP4, GPU0-3) |
| 并行 | group_size=8，slot 轮询 4+4 分布在两台开发机 |
| 实测 | rollout 68.8/71.4s；等待占比 52.7%/68.3%；actor 82/151 TFLOPS |
| 资源余量 | worker load≈1.1（远未饱和）；rollout GPU 利用率均值 2.3%；gateway 信号量 512 |

### 1.2 worker 实际容量（关键约束）

开发机 1（luyd-dev）实测：**7 CPU、15GB 内存**（当前 available ≈3GB，buff/cache
≈7GB可回收）；docker 可见 7C/16G。开发机 2 需按 §4.1 同样预检。

每个 trial = 1 个 SSH runner + 1 个 Docker 容器（规格 cpus=1 / memory_mb=2048，
内含 Claude Code node 进程）。当前 4 并发/机时 load 仅 ~1.1，说明 trial 主要是
等待远端 GPU 生成、CPU 不是第一瓶颈；**内存是并发上限的主要约束**：
8 容器/机 ≈ 名义 16GB 已达上限，需按实测驻留内存分阶段爬坡并设置回退阈值。

### 1.3 结论

- 提高单机并发到 8 大概率安全；12–16 需实测内存驻留后决定；
- 若要突破 16+16，应增加第 3 台 worker 而非继续压单机（本方案不含）。

## 2. 数据集选择（三层漏斗）

### 2.1 漏斗

```text
静态预期可接入（09-23 重扫：386,625 缓存任务中 97.97% NEEDS_PROBE）
    ↓  按 §3 完成镜像与双机动态验证
动态验证通过任务
    ↓  8 样本难度探测（pass rate 0.2–0.8 才入选）
正式训练任务池（12–16 个）
```

静态数据源：`runs/native-mvp-20260923-reward-contrast/hub-native-profile-rescan.json`
（91.4% 任务 test.sh 直接引用 reward artifact；选题优先从此子集取）。

### 2.2 候选池（按难度分层）

| 层 | 数据集 | 重扫静态通过 | 已有难度证据 | 备注 |
|---|---|---:|---|---|
| 已证有方差 | `bigcode/humanevalfix` | 94/94 | 09-22 对比 run：同组 0/1 混合 | 首批必选（python-1 已有镜像与执行锁） |
| 已证有方差 | `nvats/codeskills-bench` | 16/16 | 六候选 RT：8 样本 7×0+1×1 | 首批必选 |
| 偏易/校准 | `quixbugs/quixbugs` | 55/55 | 小规模 bug 修复，预期中高通过率 | 混合批中的高分位 |
| 中等 | `deveval/deveval`、`evoeval/evoeval` | 38/38、72/72 | 动态验证 5 任务通过 | 代码生成中等难度 |
| 中等 | `adyen/dabstep` | 299/299 | 动态验证 5 任务通过 | 数据分析型，多样性好 |
| 中等 | `aider/aider-polyglot` | 138/138 | 动态验证 5 任务通过 | 多语言 diff 修复 |
| 偏难 | `abundant/swe-gen-cpp` | 646/646 | 六候选：8×0 | 提供低分样本，防止全对饱和 |
| 偏难 | `aarr/aarri-bench` | 56/56 | 六候选：8×0 | 同上 |
| 备选 | `camel-ai/seta-env`(871/875)、`swe-bench/swe-bench-verified`(329/329)、`vmax/vmax-tasks`(689/689)、`userbench/UserBench`(415/415) | 全通过 | 无难度证据 | 探测后替补 |

明确排除：`aime`/`usaco`（纯数学，8B 预期全 0，工具无收益）、
`theagentcompany`（长程多轮，超时风险）、`abundant/swe-marathon`（11/16 通过且
六候选全 0，收益低）。

### 2.3 难度探测 run（必做，防止再次全 0/全 1）

用完整链路做探测（顺带回归），配置：

```yaml
sampling: {group_size: 4, groups_per_batch: 8, max_attempts: 2, ...}
training: {num_rollout: 1, save_interval: 2, ...}
```

- 每次探测 8 任务 × 4 样本 = 32 trial，16 并发约 2 波 ≈ 5–8 分钟 + 训练一步；
- 每任务记录 reward 分布，入选标准：`0 < mean < 1` 且组内出现不同 reward
  （理想 mean 0.2–0.8）；
  全 0/全 1 的任务保留证据后移出正式 catalog；
- 两轮探测即可覆盖 16 个候选任务。

## 3. 任务接入流水线（每任务）

复用 09-22 `humanevalfix-python1` 的成熟流程（证据模板：
`runs/native-mvp-contrast-20260922/`）：

1. **取源**：从 hub 缓存取任务目录（缺失则从 Hub 重新下载对应 digest 包）；
2. **镜像**：以任务原镜像为 base 构建 `harbornative:<task>-claude-2.1.141`
   派生镜像（预装精确 Claude ELF；禁止改动 instruction/solution/tests 语义）；
3. **双机验证**：在开发机 1、2 各跑一次 baseline（期望 reward=0）与
   solution（期望 reward=1），并核对 `verifier_result` 与 reward artifact 一致；
4. **执行锁**：生成 `task-<id>-claude-preinstalled/`（task.toml 指向派生镜像 +
   代理 env），计算 `tree_digest`，写入 catalog；
5. **doctor + dry-run** 全绿后才进入探测。

成本估算：每任务镜像构建+双机验证 3–8 分钟，4 路并行构建，16 任务约
0.5–1.5 小时（base 镜像多为 python:3.x-slim / ubuntu:24.04，需先在两台 worker
预拉取，避免构建风暴）。

## 4. Rollout 并行扩展（分阶段爬坡）

### 4.1 预检（每台 worker）

```bash
nproc; free -g; df -h /var/lib/docker
docker info --format '{{.NCPU}} {{.MemTotal}}'
# 并发压力探针：8 个并发容器跑真实任务 3 分钟，记录 load/RAM/失败率
```

### 4.2 阶段与门槛

| 阶段 | 并行/机 | 总并发 | 触发配置 | 进入下一阶段条件 | 回退条件 |
| --- | ---: | ---: | --- | --- | --- |
| P0 基线 | 4 | 8 | group_size=8（今日已验） | — | — |
| P1 | 8 | 16 | group_size=16（或 2 组×8） | load<5(7C 机)、available>2GB、trial 失败率<5% | available<1.5GB 或出现 cleanup-required |
| P2 | 12 | 24 | group_size=24 / groups_per_batch=3×8 | 同上，且单 trial 时长增幅<50% | 同上 |
| P3（可选） | 16 | 32 | 4 组×8 | P2 稳定且内存驻留<12GB | 回 P2 |

说明：native runner 直接经 SSH 起 Harbor Trial，不受旧 worker_pool 的
`WORKER_MAX_*` 限制；slot 分布已按 `(slot+retry) % 2` 轮询，扩并发只需扩大
group_size/groups_per_batch，无需改代码。

### 4.3 服务端余量核对

- SGLang 引擎（4×H200 TP4）：8 并发时利用率 2.3%，32 并发预计仍 <20%，无需改；
- gateway 信号量 = sglang_server_concurrency(512) × 引擎数，不是瓶颈；
- 可选项（P3 之后）：`rollout_gpus_per_engine=2` 起 2 个 TP2 引擎提升服务隔离，
  但会改变已验收布局，本轮默认不做。

## 5. 正式训练配置

```yaml
sampling:
  group_size: 8          # GRPO 组内样本
  groups_per_batch: 4    # 每步 4 个不同任务 → 32 样本/步
  max_attempts: 2
  max_tokens: 1024
  max_context: 32768
training:
  num_rollout: 6         # 6 步，共 192 条轨迹
  learning_rate: 0.000001
  save_interval: 6       # 只保存最终 checkpoint，避免多份 107G
deployment:              # 布局不变
  layout: split
  actor_gpus: 4 / rollout_gpus: 4 / TP 4+4
```

预计耗时：rollout 每步 2 波×~2.5min ≈ 5–6min；train 32 样本/步 ≈ 2.5–4min；
6 步 + 启动 + checkpoint ≈ **60–75 分钟**。

### 磁盘预算（必须先做）

- 共享 ipfs 卷当前 ≈90G 可用，低于 checkpoint 预检线 128G；
- 按既有流程 retire 今日 107G Megatron checkpoint（保留 16G updated-hf、权重
  diff、日志与验收 JSON，写 retirement 记录）→ 恢复 ≈197G；
- 新 run 只写一份最终 checkpoint（save_interval=6）。

### 运行纪律

1. 每次 launch 前 `ray stop --force`（Ray 残留会 6379 冲突）；
2. 确认 GPU Pod 到两台 worker 的 root SSH key 有效（Pod 重建会丢）；
3. 复用 monitor 脚本：`monitor-gpu.sh`、`monitor-worker.sh`（5s 采样）。

## 6. 监控与 KPI

| 指标 | 采集 | 目标 |
|---|---|---|
| rollout 时长 / 等待占比 | LIGHTRL_METRIC_JSON `perf/*` | wait_time_ratio < 35% |
| rollout GPU 利用率 | gpu.csv | 均值 > 10%（当前 2.3%） |
| worker 并发/负载/内存 | worker csv + free | 峰值并发=设定值，load<5，available>1.5GB |
| 每 trial 时长与失败率 | runner-events/evaluation 时间戳 | 中位数 <180s，失败率 <5% |
| 每任务奖励分布 | trajectory.v2.json | 入选任务保持 0<mean<1 |
| 训练信号 | grad_norm/pg_loss | 每步非零 |
| 权重变化 | checkpoint→HF→diff | 变化元素 >1e8 量级，非零 |

## 7. 执行顺序

| 步骤 | 内容 | 预计 |
|---|---|---|
| 1 | retire 今日 checkpoint；worker 容量预检；ray/SSH/端口体检 | 0.5h |
| 2 | 16 候选任务接入（镜像+双机验证+锁+catalog），4 路并行 | 1–1.5h |
| 3 | 两轮难度探测 run（32 trial/轮），筛出 12–16 任务 | 0.5h |
| 4 | P1 正式训练（16 并发起步，视门槛升 P2） | 1–1.25h |
| 5 | 权重 diff、验收 JSON、文档归档 | 0.5h |

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| worker 内存不足（15GB/机） | 分阶段爬坡 + available<1.5GB 自动回退；必要时降低容器 memory_mb 需重锁任务 |
| 任务全 0/全 1（零梯度） | 探测 run 前置筛选；混合难度分层组批 |
| 上下文超限（>32768） | 接入时检查 instruction+环境渲染 token，超限任务淘汰 |
| 镜像拉取/构建风暴 | 提前双机预拉 base 镜像；4 路并行构建 |
| Ray/端口/SSH 残留 | §5 运行纪律三条 |
| 磁盘低于 128G 守卫 | 先 retire，save_interval=num_rollout |
| 高并发下 trial 失败率上升 | max_attempts=2 + 失败率>5% 降并发重跑 |

## 9. 验收标准

1. catalog 含 ≥12 个跨 ≥8 个数据集的真实 Hub 任务，全部有双机 baseline/solution 证据；
2. 每步 32 样本、6 步训练完成，所有组内奖励非全同，每步 grad_norm>0；
3. 最终 checkpoint 转换 HF 后与基线 diff 非零（变化元素 ≥1e8 量级）；
4. wait_time_ratio <35% 且无 cleanup-required / trace-unsealed 残留；
5. 产出验收 JSON + 更新 leadership report 的“实际接入过”清单。
