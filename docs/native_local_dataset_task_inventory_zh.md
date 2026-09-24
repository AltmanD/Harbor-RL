# 本地多数据集任务环境清单（2026-09-24）

本文档汇总当前本机已准备好的数据集 / task 执行环境，供后续训练流程验证直接复用。所有路径均位于 GPU 训练机与两台 CPU worker 共享的存储上，可直接跨流程引用。

## 0. image-tars 可用性总账（2026-09-24）

| 统计项 | 数量 | 说明 |
|---|---:|---|
| tar 总数 | 30 | 位于 `runs/native-mvp-20260923-multids/image-tars/`，合计 7.4G |
| 可用 task 镜像 | 29 | 均可定位执行锁并可 `docker load` |
| 备用镜像（无执行锁） | 1 | `mmau-claude-2.1.141.tar`；同 dataset 的 `dyn-mmau` 已有可用锁 |
| 覆盖外部 dataset | 25 | `bigcode/humanevalfix` 与 `aider/aider-polyglot` 各含 2 个可用 task |
| 外部 dataset task | 27 | 原 catalog 15 + 动态扩充 9 + 历史 hello-world 1 + humanevalfix-python0/aider-grade-school 复用 dataset 各 1 |
| 内部契约测试 task | 2 | `log-summary-native`、`regex-log-native`，用于 native 契约 / GRPO 导出验证 |

可用 task 构成：15 个原 catalog task（§3.1）+ 9 个动态验证扩充 task（§3.2）+ 5 个历史验证 task（hello-world、humanevalfix-python0、aider-grade-school、log-summary、regex-log）。历史 task 的执行锁分别位于 `runs/native-mvp-20260922/` 与 `runs/native-mvp-contrast-20260922/`；`humanevalfix-python1` tar 同时属于原 catalog 与历史训练证据，但只按 1 个 task 计数。

## 1. 环境根目录与结论

| 项 | 值 |
|---|---|
| 环境根目录 | `/mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260923-multids` |
| 任务清单（原 catalog） | `tasks.json`（15 个数据集 × 各 1 个 task） |
| 动态验证扩充清单 | `dynamic-tasks.json`（21 个数据集 × 各 1 个 task；9 个镜像就绪，12 个源就绪待重建） |
| 训练 catalog | `native-catalog.json`（15 个执行锁，schema v2） |
| 训练配置 | `native-config.yaml`（backend=`harbor_job`，harness=`claude_code`） |
| 一键启动 | `launch.sh`（支持 `--preflight-only`） |
| 镜像导入 | `load-images.sh`（批量 `docker load` `image-tars/*.tar`） |
| 执行锁目录 | `task-<short-id>-claude-preinstalled/`（改写镜像 + 代理 env，语义不变） |
| 原始任务副本 | `task-<short-id>-original/`（Hub 源任务完整拷贝） |
| 双机验证证据 | `image-validation/<worker>/summary.json` 与逐任务 `result.json` |
| 镜像 tar | `image-tars/`（30 个 tar / 7.4G；29 个 task 镜像可用，详见 §0） |

结论：`image-tars/` 中共有 **30 个 tar / 7.4G**，其中 **29 个是可直接使用的 task 镜像**（均能定位到执行锁），**1 个是备用镜像**（`mmau-claude-2.1.141.tar`，当前无执行锁）。29 个可用 task 覆盖 **25 个外部 dataset**（其中 `bigcode/humanevalfix`、`aider/aider-polyglot` 各含 2 个 task）另加 2 个内部契约测试 task（`log-summary`、`regex-log`）；外部 dataset 的构成为原 catalog 15 个 + 动态验证扩充 9 个 + `harbor/hello-world` 1 个。若只做训练链路连通性验证，建议优先使用双机全绿任务；动态扩充 task 的验证等级见 §5.3。

## 2. 训练拓扑与关键配置

| 项 | 值 |
|---|---|
| Harbor 版本 | `0.23.0`（`.venv-harbor-0.23/bin/python`） |
| CPU worker 1 | `root@100.98.161.238`（hostname `luyd-dev`） |
| CPU worker 2 | `root@100.103.145.28`（hostname `dev`） |
| Gateway 端口 | `32123`（`advertised_url` 由 `launch.sh` 按 GPU 机 routable IP 自动改写） |
| 模型 | Qwen3-8B，`checkpoint=reference=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B` |
| 布局 | split，8 GPU：actor 4（TP4）+ rollout 4（TP4） |
| 采样 | `group_size=2 × groups_per_batch=8`，`max_attempts=2`，`max_tokens=1024`，`max_context=32768` |
| 训练 | `num_rollout=2`，`lr=1e-6`，`save_interval=2` |
| 输出根 | 环境根目录本身（`output.root`） |
| Claude 二进制 | 派生镜像内预装 `claude` 2.1.141（`/usr/local/bin/claude`） |
| 代理 | `httpproxy-headless.kubebrain.svc.pjlab.local:3128`，已写入各执行锁与验证脚本 |

关键绝对路径：

```text
RUN=/mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260923-multids
CATALOG=$RUN/native-catalog.json
CONFIG=$RUN/native-config.yaml
TASKS=$RUN/tasks.json
LAUNCH=$RUN/launch.sh
LOAD=$RUN/load-images.sh
```

## 3. 数据集 / task 清单

资源列为 `CPU / 内存 / 存储`，超时列为 `agent / verifier`。验证列统计两台 CPU worker上的 baseline(0) + solution(1) 连通验证结果；`humanevalfix1` 复用既有已验收锁。

| 数据集 | short id | Hub task | 镜像 tag（`harbornative:` 前缀） | 资源 | 超时 | 双机验证 |
|---|---|---|---|---|---|---|
| nvats/codeskills-bench | `codeskills` | `nvats/bench-circular-import-cold-load` | `codeskills-claude-2.1.141` | 2 / 2048MB / 2048MB | 600s / 120s | 2/2 通过 |
| quixbugs/quixbugs | `quixbugs` | `quixbugs/python-bitcount` | `quixbugs-claude-2.1.141` | 1 / 2048MB / 4096MB | 600s / 600s | 1/2 通过 |
| adyen/dabstep | `dabstep` | `adyen/3` | `dabstep-claude-2.1.141` | 1 / 4096MB / 8192MB | 1800s / 600s | 2/2 通过 |
| aider/aider-polyglot | `aiderpoly` | `aider/polyglot_go_octal` | `aiderpoly-claude-2.1.141` | 1 / 4096MB / 10240MB | 1800s / 1800s | 2/2 通过 |
| deveval/deveval | `deveval` | `deveval/cpp-area-calculation-unit-testing` | `deveval-claude-2.1.141` | 1 / 2048MB / 10240MB | 1000s / 600s | 2/2 通过 |
| evoeval/evoeval | `evoeval` | `evoeval/45` | `evoeval-claude-2.1.141` | 1 / 2048MB / 4096MB | 600s / 120s | 2/2 通过 |
| abundant/swe-gen-cpp | `swegencpp` | `abundant/envoyproxy__protoc-gen-validate-13` | `swegencpp-claude-2.1.141` | 1 / 2048MB / 10240MB | 600s / 600s | 0/2 通过 |
| aarr/aarri-bench | `aarri` | `aarr/broken-dataset-download` | `aarri-claude-2.1.141` | 1 / 2048MB / 10240MB | 600s / 600s | 2/2 通过 |
| camel-ai/seta-env | `setaenv` | `camel-ai/475` | `setaenv-claude-2.1.141` | 1 / 2048MB / 10240MB | 360s / 360s | 0/2 通过 |
| arcprize/arc-agi-2 | `arcagi` | `arcprize/20270e3b_1` | `arcagi-claude-2.1.141` | 1 / 1024MB / 2048MB | 600s / 60s | 2/2 通过 |
| gaia/gaia | `gaia` | `gaia/9f41b083-683e-4dcf-9185-ccfeaa88fa45` | `gaia-claude-2.1.141` | 1 / 2048MB / 10240MB | 600s / 300s | 2/2 通过 |
| gorilla/bfcl | `bfcl` | `gorilla/live-irrelevance-123-9-3` | `bfcl-claude-2.1.141` | 1 / 2048MB / 10240MB | 300s / 300s | 2/2 通过 |
| crustbench/crustbench | `crustbench` | `crustbench/math-library-in-c` | `crustbench-claude-2.1.141` | 1 / 2048MB / 10240MB | 1800s / 120s | 2/2 通过 |
| bauerjustin/terminal-bench-3-test | `tb3test` | `bauerjustin/hello-world` | `tb3test-claude-2.1.141` | 1 / 2048MB / 10240MB | 120s / 120s | 1/2 通过 |
| bigcode/humanevalfix | `humanevalfix1` | HumanEvalFix Python/1 | `humanevalfix-python1-claude-2.1.141` | 1 / 2G / 5G | 600s / 600s | 复用 09-22 已验收锁 |

说明：

- `humanevalfix1` 的执行锁位于
  `runs/native-mvp-contrast-20260922/task-humanevalfix-python1-claude-preinstalled/`，  不在 multids 根目录下，但已加入 multids catalog。
- 每个任务均有 `instruction.md`、`solution/`、`tests/` 与改写后的 `task.toml`；
  原始语义未改动，仅替换 `docker_image` 并补充代理环境变量。
- `apple/mmau` 曾在候选清单中并留有 `mmau-claude-2.1.141.tar`，但最终未进入
  `tasks.json` / catalog；`tb3preview` 目录为额外实验任务，同样未进入最终 catalog。

### 3.2 动态验证扩充清单（21 个 dataset × 1 task）

来源为 `20260918-full-coverage` 动态验证（每个 dataset ≥5 个 task 通过）；每个dataset 选 1 个缓存存在、历史耗时最短的通过 task。资源列为 `CPU / 内存 / 存储`，超时列为 `agent / verifier`；`状态=就绪` 表示 tar 已在双 worker 加载验证。

| 数据集 | short id | Hub task | Hub digest（短） | task_digest（短） | 资源 | 超时 | 状态 |
|---|---|---|---|---|---|---|---|
| abundant/swe-gen-js | `dyn-swe-gen-js` | `abundant/mtth__avsc-261` | `ce48ff3b2673` | `2dad73c73699` | 1 / 2048MB / 10240MB | 600s / 600s | 就绪 |
| aime/aime | `dyn-aime` | `aime/83` | `532165190ddd` | `f0d3bf5a7209` | 1 / 2048MB / 10240MB | 3000s / 3000s | 就绪 |
| apple/mmau | `dyn-mmau` | `apple/44d41585-a609-400c-8e40-dafef61c91f7` | `3e68d81331af` | `2d4dcc5032d6` | 1 / 2048MB / 10240MB | 600s / 7200s | 就绪 |
| benchflow/skillsbench | `dyn-skillsbench` | `benchflow/jpg-ocr-stat` | `fc1c03683981` | `eb44e3f4756f` | 4 / 4096MB / 10240MB | 1800s / 600s | 就绪 |
| cais/swebenchpro | `dyn-swebenchpro` | `cais/instance_nodebb__nodebb-445b70...` | `00031f7d9e13` | `500db36dbd66` | 1 / 4096MB / 10240MB | 3000s / 3000s | 待重建 |
| factory-ai/legacy-bench | `dyn-legacybench` | `factory-ai/6fe1ab-java7-mtom...` | `e766954e5297` | `65b4e39e05d0` | 1 / 2048MB / 10240MB | 300s / 300s | 待重建 |
| featurebench/featurebench | `dyn-featurebench` | `featurebench/pandas-dev__pandas.82fa2715...` | `0cba9991f159` | `afd6ccc6f954` | 2 / 8192MB / 15360MB | 3600s / 3600s | 待重建 |
| featurebench/featurebench-lite | `dyn-featurebench-lite` | `featurebench/pandas-dev__pandas.82fa2715...` | `1d7dcfbbc012` | `5606d14cbb6b` | 2 / 8192MB / 15360MB | 3600s / 3600s | 待重建 |
| featurebench/featurebench-lite-modal | `dyn-featurebench-lite-modal` | `featurebench/pandas-dev__pandas.82fa2715...` | `05e000db19df` | `dcfe7bf426c0` | 2 / 8192MB / 15360MB | 3600s / 3600s | 待重建 |
| gorilla/bfcl_parity | `dyn-bfclparity` | `gorilla/bfcl-parallel-5` | `59b39220746f` | `6098de3b1dde` | 1 / 2048MB / 10240MB | 300s / 300s | 就绪 |
| openthoughts/openthoughts-tblite | `dyn-tblite` | `openthoughts/mech-system` | `fc5759f82ccd` | `4bd7c524748a` | 2 / 4096MB / 10240MB | 900s / 900s | 就绪 |
| openthoughts/tasktrove-nemotron-code-oracle-filtered | `dyn-nemotron` | `openthoughts/...-nemotron-006431` | `06606db4ed17` | `299f5e7c0907` | 1 / 2048MB / 10240MB | 1500s / 600s | 就绪 |
| openthoughts/tasktrove-swe-rebench-patched-oracle | `dyn-swe-rebench-patched` | `openthoughts/...swe_rebench-00324` | `2577053536b5` | `9707bd5cbabf` | 1 / 2048MB / 10240MB | 1500s / 600s | 待重建 |
| openthoughts/tasktrove-swesmith-oracle-filtered | `dyn-swesmith` | `openthoughts/...swesmith-24669` | `0eb39ecff89e` | `3325ab026346` | 1 / 2048MB / 10240MB | 1500s / 600s | 待重建 |
| qcircuitbench/qcircuitbench | `dyn-qcircuitbench` | `qcircuitbench/qaoa-n8` | `17d747dda1e5` | `530230adf0b9` | 2 / 4096MB / 10240MB | 600s / 600s | 待重建 |
| quesma/compilebench | `dyn-compilebench` | `quesma/coreutils` | `f9ab9de8f38b` | `6d6d53efd7d4` | 2 / 4096MB / 10240MB | 900s / 900s | 待重建 |
| replicationbench/replicationbench | `dyn-replicationbench` | `replicationbench/ver_waves__gaia_breathing_typical` | `00c1fea3150f` | `edeb16bc2306` | 1 / 2048MB / 10240MB | 3600s / 1800s | 待重建 |
| scale-ai/swe-bench-pro | `dyn-swe-bench-pro` | `scale-ai/instance_ansible__ansible-1ee70...` | `009f37c067e1` | `124e03c834e9` | 1 / 4096MB / 10240MB | 3000s / 3000s | 待重建 |
| swe-rebench/swe-rebench-leaderboard | `dyn-swe-rebench-lb` | `swe-rebench/imperialcollegelondon__pyrealm-593` | `0033291247d9` | `9e0dbddfdf86` | 未声明 / 未声明 / 未声明 | 3000s / 3000s | 待重建 |
| terminal-bench-pro/terminal-bench-pro | `dyn-tbp` | `terminal-bench-pro/bash-ddos-traffic-analyzer` | `323cdf1c656e` | `ea9dbd3c8ad6` | 1 / 2048MB / 10240MB | 3600s / 360s | 就绪 |
| usaco/usaco | `dyn-usaco` | `usaco/1038` | `936b77e3ea72` | `7c319527c7b9` | 1 / 2048MB / 4096MB | 600s / 7200s | 就绪 |

完整 digest、源路径、历史动态验证证据与 tar 状态见`dynamic-verified-image-tars.json`；执行锁目录为`task-dyn-<short-id>-claude-preinstalled/`。

## 4. Digest 对照表

`Hub digest` 为 Hub 缓存源任务 digest；`task_digest` 为执行锁目录 tree digest（`native-catalog.json` 引用该值）。源任务缓存根目录为`/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/luyudong/harbor-hub-validation/datasets/<前2位>/<digest>/task`。

| short id | Hub digest（短） | Hub digest | task_digest |
|---|---|---|---|
| `codeskills` | `1358b19fd4a2` | `1358b19fd4a2a1fa72de340f63deb300d387e9cd27017b1ac43cdcf98ec0aa11` | `db032e46f32fe6672bc4cfb1fe57428655e253e65c74c86f72e46c78fdfbcbc3` |
| `quixbugs` | `641d0ea3a9f3` | `641d0ea3a9f312ca58c91d4ca70fe18ace29a78c5756199323d41428408bd481` | `601e7c6d5a29dfc9e35d7be00433674892e17517114f1d7b6340af28cf72d20b` |
| `dabstep` | `02e2d4306863` | `02e2d4306863889b5dccdab9a9b25bd45c23ac149e8ef511da45f62a3ec5ed4e` | `248105a7996d5e8e95fdce051fcc4076808fd563bff382f0706320347bbd42bf` |
| `aiderpoly` | `612708401c41` | `612708401c419277261956ead5752f0e54f4d948c655e1204d2b5dfcce5d250b` | `cd9e3b648228704faf93e122c40b2624fd93c13d7a9ed54f29ab3601f3e2c724` |
| `deveval` | `1923a7ec79da` | `1923a7ec79da0140ca2c7769baedf35d7979b9a111cd509940be917808092d51` | `0e48522a8ac4e1f9d5b336908a98dab44b6b5fc534c8a9f69c24935adabdc97b` |
| `evoeval` | `5a3db2b3d103` | `5a3db2b3d103f2de7c5b091beec0699c964f998127285d44acbfc897ee40c387` | `d142a47d532abcba279db1e78135f4fd2de7d1423f05b13a4f75eb0a4594434a` |
| `swegencpp` | `c38236c28e9e` | `c38236c28e9ec91cbdc0c320946076da9877a6f7634e5212265e87a54a649067` | `8f3621c12277ddcc44e28a525d138a84c5939b2c08a06c0b50e7a6646798ab5f` |
| `aarri` | `4244042fdd79` | `4244042fdd7984d697e66109affb190ec0199fe0516ee22cf1dd95e64d2a2346` | `28cd1ef0ea6332fb4ea6d0fa6c6f623d8eba023735019d79c318b058ae982692` |
| `setaenv` | `604c20eee5e5` | `604c20eee5e59474a21f633c6be471b6af99242bc54da2a20e7a94d6eb39da44` | `8cc1d6a876f2e0789b0cc9f4eb7874ccb6614ee70cd173434fd548e68ed63c4d` |
| `arcagi` | `57ba455570a8` | `57ba455570a85eb5e235e15d03e7e0d27257f6d78cc7f531f22ffa1e3a53ddd5` | `4c3f43955a6333d1864922cfbbe125eff010da085a5087db91ddba1158a4bf51` |
| `gaia` | `0bea003ddcf5` | `0bea003ddcf5c60d922280b32018cecc1aa1834566a7fb7fc653640fe4be1055` | `d277e25422adf9fadf65c0315bccffd8a1a5dd86589bf96b9142930721bdd4cb` |
| `bfcl` | `4b8d8fc79170` | `4b8d8fc791709709d11de8378e2ef5df925b005e221ea2fd527a2b10b1490f08` | `bf46fe672a1946440b663582fb46ab8ff47dba591272647fde4dd743b862cc18` |
| `crustbench` | `e1e8bc113e9d` | `e1e8bc113e9d102caa8411703e4b6cda4a7b868f3391d35e18c516a82fbcee7a` | `a4c8fb928955da01acb83738b522434462d25157d9111ae630cff72b25353f28` |
| `tb3test` | `2b4aa0d5e03a` | `2b4aa0d5e03aacedf6dfb1ef8b4e58471a380d862970511620d6bebaaa2ff9f9` | `ee85c61a12cc2a14af077005544ceaafd0de483f492702b7a48867302a2f51b3` |
| `humanevalfix1` | `01fd388b37e6` | `01fd388b37e627c92bdb87041f7cc89f23de8f6f1645f444db7a870695902eaa` | `01fd388b37e627c92bdb87041f7cc89f23de8f6f1645f444db7a870695902eaa` |

## 5. 镜像 tar 清单

位置：`runs/native-mvp-20260923-multids/image-tars/`，合计 **30 个 tar / 7.4G**。在任一 CPU worker 上执行 `bash <RUN>/load-images.sh` 即可批量导入本机 Docker。

### 5.1 多数据集任务镜像

| tar | 大小 |
|---|---:|
| `aarri-claude-2.1.141.tar` | 114M |
| `aiderpoly-claude-2.1.141.tar` | 443M |
| `arcagi-claude-2.1.141.tar` | 135M |
| `bfcl-claude-2.1.141.tar` | 118M |
| `codeskills-claude-2.1.141.tar` | 121M |
| `crustbench-claude-2.1.141.tar` | 458M |
| `dabstep-claude-2.1.141.tar` | 336M |
| `deveval-claude-2.1.141.tar` | 1.5G |
| `evoeval-claude-2.1.141.tar` | 119M |
| `gaia-claude-2.1.141.tar` | 117M |
| `mmau-claude-2.1.141.tar`（未入最终 catalog） | 112M |
| `quixbugs-claude-2.1.141.tar` | 166M |
| `setaenv-claude-2.1.141.tar` | 98M |
| `swegencpp-claude-2.1.141.tar` | 576M |
| `tb3test-claude-2.1.141.tar` | 98M |
| `humanevalfix-python1-claude-2.1.141.tar` | 464M |

上表覆盖全部 15 个 catalog 任务；§6 中的失败项指 baseline/solution 奖励连通验证未达预期，而非 tar 损坏或容器无法启动（`swegencpp` 为验证脚本 600s 超时，rc=125）。

### 5.2 历史验证 / 实际训练镜像（09-24 补充导出）

以下镜像来自此前真实训练或动态验证流程，已从 `luyd-dev` 导出为 tar，并在`dev` 上完成 `docker load` 加载验证：

| tar | 大小 | 验证 / 使用证据 |
|---|---:|---|
| `hello-world-a411e432-claude-2.1.141.tar` | 157M | 09-22 native MVP 全链路与 09-23 reward-contrast 训练实际使用 |
| `humanevalfix-python1-claude-2.1.141.tar` | 464M | 09-22 contrast run 双机 baseline=0 / solution=1，最终训练验收通过 |
| `humanevalfix-python0-claude-2.1.141.tar` | 464M | 本地 smoke 已验证任务镜像 + GPU dry-run 通过 |
| `aider-grade-school-claude-2.1.141.tar` | 375M | 已验证 no-oracle prepared task + GPU dry-run 通过 |
| `log-summary-native-claude-2.1.141.tar` | 153M | native 契约测试 / trajectory GRPO 导出链路使用 |
| `regex-log-native-claude-2.1.141.tar` | 152M | native 契约测试链路使用 |

另外 `mmau-claude-2.1.141.tar`（§5.1 表内）未进入最终 catalog，仅作备用。

### 5.3 动态验证 dataset 扩充（09-24）

来源：`runs/harbor-hub-validation/20260918-full-coverage/dynamic.sqlite`。动态验证口径为 baseline=0 / positive=1（native + materialized 双侧）。共 35 个 dataset 达到“≥5 个 task 通过”；剔除原 catalog 已覆盖的 14 个后，为剩余 21 个 dataset 各选 1 个缓存存在、历史耗时最短的通过 task 重建派生镜像。任务与环境明细见 §3.2，证据见`dynamic-verified-selection.json`、`dynamic-verified-image-tars.json`。

**已就绪（9 个 tar，均已在两台 worker `docker load` 验证）：**

| tar | dataset | 派生镜像复验状态 |
|---|---|---|
| `dyn-swe-gen-js-claude-2.1.141.tar` | abundant/swe-gen-js | baseline=0 / solution=1 通过 |
| `dyn-aime-claude-2.1.141.tar` | aime/aime | baseline=0 / solution=1 通过 |
| `dyn-mmau-claude-2.1.141.tar` | apple/mmau | 动态源验证 + Claude smoke |
| `dyn-skillsbench-claude-2.1.141.tar` | benchflow/skillsbench | 动态源验证 + Claude smoke |
| `dyn-bfclparity-claude-2.1.141.tar` | gorilla/bfcl_parity | 动态源验证 + Claude smoke |
| `dyn-tblite-claude-2.1.141.tar` | openthoughts/openthoughts-tblite | 动态源验证 + Claude smoke |
| `dyn-nemotron-claude-2.1.141.tar` | openthoughts/tasktrove-nemotron-code-oracle-filtered | 动态源验证 + Claude smoke |
| `dyn-tbp-claude-2.1.141.tar` | terminal-bench-pro/terminal-bench-pro | 动态源验证 + Claude smoke |
| `dyn-usaco-claude-2.1.141.tar` | usaco/usaco | 动态源验证 + Claude smoke |

**暂未恢复（12 个 dataset）**：历史动态验证镜像当时构建成功但之后已被清理；本轮按源任务重建时，外部 registry（Docker Hub / ghcr）auth token 拉取在 BuildKit 内不走代理而超时，或依赖超大第三方 base / Azul 镜像源。任务源与执行锁仍在`task-dyn-*-claude-preinstalled/`，后续可先预拉 base 再运行`build-dynamic.sh` 续建。详细名单见 `dynamic-verified-image-tars.json` 中`status=blocked` 项。

## 6. 双 worker 验证状态

验证口径：每台 worker 分别运行 baseline（期望 reward=0）与官方 solution（期望 reward=1），并要求 `/logs/verifier/reward.txt` 与之一致。汇总文件：`image-validation/dev/summary.json`、`image-validation/luyd-dev/summary.json`。

| short id | dev（100.103.145.28） | luyd-dev（100.98.161.238） | 结论 |
|---|---|---|---|
| `aarri` | b rc=0/r=0；s rc=0/r=1 | b rc=0/r=0；s rc=0/r=1 | 通过 |
| `aiderpoly` | b rc=0/r=0；s rc=0/r=1 | b rc=0/r=0；s rc=0/r=1 | 通过 |
| `arcagi` | b rc=1/r=0；s rc=0/r=1 | b rc=1/r=0；s rc=0/r=1 | 通过 |
| `bfcl` | b rc=0/r=0；s rc=0/r=1 | b rc=0/r=0；s rc=0/r=1 | 通过 |
| `codeskills` | b rc=0/r=0；s rc=0/r=1 | b rc=0/r=0；s rc=0/r=1 | 通过 |
| `crustbench` | b rc=1/r=0；s rc=0/r=1 | b rc=1/r=0；s rc=0/r=1 | 通过 |
| `dabstep` | b rc=0/r=0；s rc=0/r=1 | b rc=0/r=0；s rc=0/r=1 | 通过 |
| `deveval` | b rc=0/r=0；s rc=0/r=1 | b rc=0/r=0；s rc=0/r=1 | 通过 |
| `evoeval` | b rc=0/r=0；s rc=0/r=1 | b rc=0/r=0；s rc=0/r=1 | 通过 |
| `gaia` | b rc=1/r=0；s rc=0/r=1 | b rc=1/r=0；s rc=0/r=1 | 通过 |
| `quixbugs` | b rc=0/r=0；s rc=2/r=missing | b rc=0/r=0；s rc=0/r=1 | dev 失败 |
| `setaenv` | b rc=0/r=0；s rc=1/r=missing | b rc=0/r=0；s rc=1/r=missing | 双机失败 |
| `swegencpp` | b rc=125/r=missing；s rc=125/r=missing | b rc=125/r=missing；s rc=125/r=missing | 双机失败 |
| `tb3test` | b rc=0/r=0；s rc=0/r=0 | b rc=0/r=0；s rc=0/r=1 | dev 失败 |
| `humanevalfix1` | 复用 09-22 contrast run 证据 | 复用 09-22 contrast run 证据 | 通过 |

失败项处理建议：

- `quixbugs`：仅 dev 的 solution 执行 rc=2，luyd-dev 通过；怀疑单机执行环境差异，
  复跑 `validate-task.sh quixbugs <digest>` 可定位。
- `setaenv`：solution rc=1 且 reward missing；验证脚本依赖
  `task-setaenv-prestart.cmd` 启动 `myserver`，需先确认该 prestart 与 Harbor  实际容器启动路径一致。
- `swegencpp`：双机 baseline/solution 均 rc=125（timeout），当前不适合作为连通性
  验证任务，除非单独放大验证超时并确认镜像内构建耗时。
- `tb3test`：dev 上 solution reward=0，luyd-dev 通过；属轻量 hello-world 任务，
  建议优先排查 dev 写入/挂载差异。

## 7. 其他流程复用方法

1. **在 CPU worker 导入镜像**（两台各执行一次）：

   ```bash
   bash /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260923-multids/load-images.sh
   ```

2. **GPU 机预检**（SSH 可达、镜像存在、端口/GPU、doctor/dry-run）：

   ```bash
   bash /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260923-multids/launch.sh --preflight-only
   ```

3. **启动训练**（脚本内部会 `ray stop --force`）：

   ```bash
   bash /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260923-multids/launch.sh
   tail -f /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260923-multids/launch.log
   ```

4. **复用 catalog/config 到新流程**：直接引用 §2 中的 `CATALOG` / `CONFIG`
   绝对路径；若复制到新 run 目录，需同步修正 `native-config.yaml` 中
   `tasks.catalog`、`output.root` 与 `advertised_url`。

5. **单任务连通复验**（在 worker 上执行）：

   ```bash
   cd /mnt/shared-storage-user/luyudong/Harbor-RL/runs/native-mvp-20260923-multids
   ./validate-task.sh <short-id> <hub-digest>
   ```

## 8. 使用边界

- 两台 worker 实测约 7 CPU / 15–16G 内存；当前设计并发为 8+8，进一步扩并发前需
  重新评估内存驻留。
- 训练启动前需确认 GPU Pod 到两台 worker 的 root SSH key 有效。
- `launch.sh` 会清理 Ray 并检查 32123/8265/15000 端口；不要与其他训练任务共用
  同一 GPU 组同时启动。
- 共享存储剩余空间需满足 checkpoint 预检线（历史流程按 ≥128G 管理）。
