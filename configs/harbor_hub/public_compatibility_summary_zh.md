# Harbor Hub 兼容性汇总

HarborRL 直接消费 Harbor 认可的 native task。本页不只给统计值：机器可读的精确 dataset/task 索引见 [compatibility_index.json](compatibility_index.json)，其中包含 task version ID、revision、内容 digest 和 archive path。

筛选状态只表示本地兼容性证据，不发布内部训练就绪标记，也不承诺 benchmark 性能：

- `static_only`：通过 HarborRL 静态 native contract 检查；未完成动态构建或 reward 验证。
- `dynamic_verified`：先通过同一静态检查，又完成了环境构建/物化和 reward 输出验证。动态验证不等于已完成 RL 训练，也不保证所有同类 task 都可用。

## 验证汇总

| 口径 | 数量 |
| --- | ---: |
| 扫描过的 Hub dataset | 303 |
| 至少一个静态兼容 task 的 dataset | 183 |
| 静态兼容 dataset/task membership | 524,391 |
| 静态兼容唯一 task 内容 digest | 524,416 |
| 至少一个动态验证 task 的 dataset | 50 |
| 动态验证 dataset/task pair | 210 |
| 动态验证唯一 task 内容 digest | 203 |

计数来自 2026-09-18 快照中可访问的最新 dataset tag。同一 task 内容可能属于多个 dataset；因此 dataset/task membership 计数可能大于唯一内容 digest 计数。取回任务后仍必须运行 `harborrl hub inspect`，生成 catalog 后运行 `harborrl doctor`。

## 候选任务集合

下面给出所有 183 个静态兼容 dataset 的入口任务。动态验证 dataset 会列出全部 210 个已验证 dataset/task pair；静态-only dataset 各给出一个按 `(task, digest)` 字典序选择的代表任务。Digest 仅展示前 16 位，完整值见 JSON 索引。

<details>
<summary>动态验证任务（50 datasets / 210 dataset-task pairs）</summary>

| Dataset | Task | Revision | Digest | 筛选状态 |
| --- | --- | ---: | --- | --- |
| `aarr/aarri-bench` | `aarr/cite-misuse` | 1 | `336874b510d08a4a` | 静态 + 动态 |
| `aarr/aarri-bench` | `aarr/idea-curse` | 1 | `e221e249df13f345` | 静态 + 动态 |
| `aarr/aarri-bench` | `aarr/leakage-hunt` | 1 | `105246fb8fc8454f` | 静态 + 动态 |
| `aarr/aarri-bench` | `aarr/rebuttal-reply` | 1 | `53519f1e1a224c83` | 静态 + 动态 |
| `aarr/aarri-bench` | `aarr/sharp-ac` | 1 | `a0910823a36d7a9d` | 静态 + 动态 |
| `abundant/swe-gen-cpp` | `abundant/fmtlib__fmt-264` | 1 | `4df1db8bf88317b7` | 静态 + 动态 |
| `abundant/swe-gen-cpp` | `abundant/fmtlib__fmt-312` | 1 | `b3f315cf2a5129d7` | 静态 + 动态 |
| `abundant/swe-gen-cpp` | `abundant/opencv__opencv-12319` | 1 | `017fb232e720d155` | 静态 + 动态 |
| `abundant/swe-gen-cpp` | `abundant/opencv__opencv-13359` | 1 | `d1e5552d8d7a98df` | 静态 + 动态 |
| `abundant/swe-gen-cpp` | `abundant/simdjson__simdjson-1122` | 1 | `008714083ef6eb3b` | 静态 + 动态 |
| `abundant/swe-gen-go` | `abundant/go-chi__chi-30` | 1 | `905c40cbf005baf4` | 静态 + 动态 |
| `abundant/swe-gen-go` | `abundant/go-kit__kit-57` | 1 | `b05bf5eab0f0cff2` | 静态 + 动态 |
| `abundant/swe-gen-go` | `abundant/go-kit__kit-67` | 1 | `3044dab6fec372ad` | 静态 + 动态 |
| `abundant/swe-gen-js` | `abundant/eslint__eslint-16272` | 1 | `0125cc265b5061a1` | 静态 + 动态 |
| `abundant/swe-gen-js` | `abundant/mtth__avsc-261` | 1 | `ce48ff3b2673b2b1` | 静态 + 动态 |
| `abundant/swe-gen-js` | `abundant/ohmjs__ohm-340` | 1 | `28b5484f5dca19ab` | 静态 + 动态 |
| `abundant/swe-gen-js` | `abundant/prettier__prettier-17708` | 1 | `4a6e1828c228d947` | 静态 + 动态 |
| `abundant/swe-gen-js` | `abundant/sindresorhus__execa-775` | 1 | `00d6903c7f5bc123` | 静态 + 动态 |
| `abundant/swe-gen-rust` | `abundant/toml-rs__toml-28` | 1 | `42974ee7979a150b` | 静态 + 动态 |
| `abundant/swe-gen-rust` | `abundant/toml-rs__toml-70` | 1 | `313c44521d171a4a` | 静态 + 动态 |
| `abundant/swe-gen-rust` | `abundant/tower-rs__tower-49` | 1 | `01011624a08ae4c9` | 静态 + 动态 |
| `adyen/dabstep` | `adyen/1645` | 1 | `01578c521cb9e6c2` | 静态 + 动态 |
| `adyen/dabstep` | `adyen/5` | 1 | `cc35b8bb406f6e17` | 静态 + 动态 |
| `adyen/dabstep` | `adyen/6` | 1 | `703ad200b3806102` | 静态 + 动态 |
| `adyen/dabstep` | `adyen/7` | 1 | `ab3791ddd23a7b21` | 静态 + 动态 |
| `adyen/dabstep` | `adyen/8` | 1 | `89cc367d70f0bc0a` | 静态 + 动态 |
| `aider/aider-polyglot` | `aider/polyglot_go_octal` | 1 | `612708401c419277` | 静态 + 动态 |
| `aider/aider-polyglot` | `aider/polyglot_go_poker` | 1 | `a5e5308d772d2dff` | 静态 + 动态 |
| `aider/aider-polyglot` | `aider/polyglot_javascript_wordy` | 1 | `0011ca971b9cf52f` | 静态 + 动态 |
| `aider/aider-polyglot` | `aider/polyglot_python_book-store` | 1 | `02053bf84c77b5c5` | 静态 + 动态 |
| `aider/aider-polyglot` | `aider/polyglot_python_phone-number` | 1 | `6339e7e8b18d91ed` | 静态 + 动态 |
| `aime/aime` | `aime/65` | 1 | `f08c1760504196cd` | 静态 + 动态 |
| `aime/aime` | `aime/83` | 1 | `532165190ddd6110` | 静态 + 动态 |
| `aime/aime` | `aime/87` | 1 | `ac67f84546a0a4f2` | 静态 + 动态 |
| `aime/aime` | `aime/i-3` | 1 | `08cb03b31a36ea02` | 静态 + 动态 |
| `aime/aime` | `aime/ii-12` | 1 | `ca3cb714af2e351b` | 静态 + 动态 |
| `apple/mmau` | `apple/1fe74624-ee85-4a25-b2ae-de1a894c2aaf` | 1 | `73cb2ce346217e38` | 静态 + 动态 |
| `apple/mmau` | `apple/36aad8e9-b351-4865-ab51-02f7400da0ff` | 1 | `4878ef031288bf9e` | 静态 + 动态 |
| `apple/mmau` | `apple/44d41585-a609-400c-8e40-dafef61c91f7` | 1 | `3e68d81331af75d2` | 静态 + 动态 |
| `apple/mmau` | `apple/5bd7a143-240e-4c72-ba7e-e3fba5821cef` | 1 | `278a8ba8730241c0` | 静态 + 动态 |
| `apple/mmau` | `apple/8cbc0f29-4cf5-41e6-83ff-abcd70fbd96d` | 1 | `d786d01fe1249307` | 静态 + 动态 |
| `arcprize/arc-agi-2` | `arcprize/195c6913_0` | 1 | `6a295fac07f4f6ab` | 静态 + 动态 |
| `arcprize/arc-agi-2` | `arcprize/65b59efc_1` | 1 | `0011d5bd697a39c3` | 静态 + 动态 |
| `arcprize/arc-agi-2` | `arcprize/8698868d_0` | 1 | `c701c6b7d2515c45` | 静态 + 动态 |
| `arcprize/arc-agi-2` | `arcprize/8f215267_0` | 1 | `137ef1aa2c8578fa` | 静态 + 动态 |
| `arcprize/arc-agi-2` | `arcprize/d59b0160_0` | 1 | `431ffaf2caa75f8c` | 静态 + 动态 |
| `bauerjustin/tb3-preview-v2` | `bauerjustin/finance-varswap-volconv-replication-bug` | 2 | `17e23840d477d51e` | 静态 + 动态 |
| `bauerjustin/tb3-preview-v2` | `bauerjustin/ot-merge-engine` | 1 | `f4cab9824005a72c` | 静态 + 动态 |
| `bauerjustin/tb3-preview-v2` | `bauerjustin/phantom-wallet-forensics` | 1 | `2b86611e1df9342b` | 静态 + 动态 |
| `bauerjustin/tb3-preview-v2` | `bauerjustin/service-mesh-certification` | 1 | `31c6a0497ad2a312` | 静态 + 动态 |
| `bauerjustin/terminal-bench-3-test` | `bauerjustin/finance-varswap-volconv-replication-bug` | 1 | `24ff66e840d306c8` | 静态 + 动态 |
| `bauerjustin/terminal-bench-3-test` | `bauerjustin/hello-world` | 1 | `2b4aa0d5e03aaced` | 静态 + 动态 |
| `bauerjustin/terminal-bench-3-test` | `bauerjustin/phantom-wallet-forensics` | 1 | `2b86611e1df9342b` | 静态 + 动态 |
| `bauerjustin/terminal-bench-3-test` | `bauerjustin/production-planning` | 1 | `1a5a9456ba6f051d` | 静态 + 动态 |
| `bauerjustin/terminal-bench-3-test` | `bauerjustin/service-mesh-certification` | 1 | `31c6a0497ad2a312` | 静态 + 动态 |
| `bauerjustin/terminal-bench-3-test` | `bauerjustin/sound-change-cascade` | 1 | `0fe11127fe38682b` | 静态 + 动态 |
| `benchflow/skillsbench` | `benchflow/adaptive-cruise-control` | 2 | `17d5e711075a7cab` | 静态 + 动态 |
| `benchflow/skillsbench` | `benchflow/jpg-ocr-stat` | 2 | `fc1c03683981d3e7` | 静态 + 动态 |
| `benchflow/skillsbench` | `benchflow/manufacturing-equipment-maintenance` | 2 | `01bcf0439a63831f` | 静态 + 动态 |
| `benchflow/skillsbench` | `benchflow/manufacturing-fjsp-optimization` | 2 | `0569f105cb996dcc` | 静态 + 动态 |
| `benchflow/skillsbench` | `benchflow/tictoc-unnecessary-abort-detection` | 1 | `10f1de394102f59d` | 静态 + 动态 |
| `bigcode/humanevalfix` | `bigcode/python-0` | 1 | `57b966cbc1de6f84` | 静态 + 动态 |
| `bigcode/humanevalfix` | `bigcode/python-5` | 1 | `90964a1ba5fe278c` | 静态 + 动态 |
| `bigcode/humanevalfix` | `bigcode/python-68` | 1 | `024abaac6ac93f6b` | 静态 + 动态 |
| `bigcode/humanevalfix` | `bigcode/python-7` | 1 | `468d971a17f64fa5` | 静态 + 动态 |
| `bigcode/humanevalfix` | `bigcode/python-9` | 1 | `92097a92487dbc8f` | 静态 + 动态 |
| `cais/swebenchpro` | `cais/instance_nodebb__nodebb-445b70deda20201b7d9a68f7224da751b3db728c-v4fbcfae8b15e4ce5d132c408bca69ebb9cf146ed` | 2 | `00031f7d9e13bd3e` | 静态 + 动态 |
| `cais/swebenchpro` | `cais/instance_nodebb__nodebb-b1f9ad5534bb3a44dab5364f659876a4b7fe34c1-vnan` | 2 | `029030c83153dd73` | 静态 + 动态 |
| `cais/swebenchpro` | `cais/instance_qutebrowser__qutebrowser-0833b5f6f140d04200ec91605f88704dd18e2970-v059c6fdc75567943479b23ebca7c07b5e9a7f34c` | 2 | `0401104de2fe66c2` | 静态 + 动态 |
| `cais/swebenchpro` | `cais/instance_qutebrowser__qutebrowser-2e961080a85d660148937ee8f0f6b3445a8f2c01-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d` | 2 | `009f8bd43317c428` | 静态 + 动态 |
| `cais/swebenchpro` | `cais/instance_qutebrowser__qutebrowser-e15d26630934d0b6415ed2295ac42fd570a57620-va0fd88aac89cde702ec1ba84877234da33adce8a` | 2 | `024cd6b319fe7d07` | 静态 + 动态 |
| `camel-ai/seta-env` | `camel-ai/1283` | 1 | `0169621e3ef61538` | 静态 + 动态 |
| `camel-ai/seta-env` | `camel-ai/163` | 1 | `016e57e141745e9e` | 静态 + 动态 |
| `camel-ai/seta-env` | `camel-ai/560` | 1 | `00501cee919d1ca5` | 静态 + 动态 |
| `camel-ai/seta-env` | `camel-ai/7` | 1 | `ee9e9de9e9444faa` | 静态 + 动态 |
| `camel-ai/seta-env` | `camel-ai/9` | 1 | `3ab85f1a07b0a6dd` | 静态 + 动态 |
| `crustbench/crustbench` | `crustbench/42-kocaeli-printf` | 1 | `0d641126ea26c9ce` | 静态 + 动态 |
| `crustbench/crustbench` | `crustbench/cjson` | 1 | `02b8c935c5d664ae` | 静态 + 动态 |
| `crustbench/crustbench` | `crustbench/rbtree-lab` | 1 | `10e7f7f9d94a214c` | 静态 + 动态 |
| `crustbench/crustbench` | `crustbench/vaultsync` | 1 | `076112b8550e06ec` | 静态 + 动态 |
| `crustbench/crustbench` | `crustbench/vec` | 1 | `0bf19d1e549ac091` | 静态 + 动态 |
| `dbt-labs/ade-bench` | `dbt-labs/ade-bench-f1002` | 1 | `6df544e7f06dfb4e` | 静态 + 动态 |
| `dbt-labs/ade-bench` | `dbt-labs/ade-bench-f1003` | 1 | `985ad0b9c10655cd` | 静态 + 动态 |
| `evoeval/evoeval` | `evoeval/34` | 1 | `1d8e48b3118daad4` | 静态 + 动态 |
| `evoeval/evoeval` | `evoeval/4` | 1 | `46d18fb2860f061e` | 静态 + 动态 |
| `evoeval/evoeval` | `evoeval/46` | 1 | `5f91db65e793f03b` | 静态 + 动态 |
| `evoeval/evoeval` | `evoeval/54` | 1 | `8780d8f4b55540ff` | 静态 + 动态 |
| `evoeval/evoeval` | `evoeval/7` | 1 | `90dd28eec6ff15b2` | 静态 + 动态 |
| `factory-ai/legacy-bench` | `factory-ai/1907c2-c-debug-legacy-buddy-fix` | 1 | `8f9959362cc4a444` | 静态 + 动态 |
| `factory-ai/legacy-bench` | `factory-ai/2831b5-java7-rating-engine-repair` | 1 | `47f5a185b0db394c` | 静态 + 动态 |
| `factory-ai/legacy-bench` | `factory-ai/3af1fe-cobol-bond-settlement-reconcile` | 1 | `dd61c3b30718ff55` | 静态 + 动态 |
| `factory-ai/legacy-bench` | `factory-ai/6fe1ab-java7-mtom-attachment-corruption-fix` | 1 | `e766954e52972de8` | 静态 + 动态 |
| `factory-ai/legacy-bench` | `factory-ai/8e8098-cobol-railcar-settlement-fix` | 1 | `7aad8a3791c9e3d4` | 静态 + 动态 |
| `featurebench/featurebench` | `featurebench/mlflow__mlflow.93dab383.test_file_store_logged_model.a9596c54.lv2` | 1 | `138a84272bdef51f` | 静态 + 动态 |
| `featurebench/featurebench` | `featurebench/mwaskom__seaborn.7001ebe7.test_distributions.f700676d.lv1` | 1 | `165aacacb3cedcb1` | 静态 + 动态 |
| `featurebench/featurebench` | `featurebench/pandas-dev__pandas.82fa2715.test_formats.05080a4e.lv1` | 1 | `137aa7723cf2fa0a` | 静态 + 动态 |
| `featurebench/featurebench` | `featurebench/pandas-dev__pandas.82fa2715.test_http_headers.aafb551e.lv1` | 1 | `078f02c1ff083036` | 静态 + 动态 |
| `featurebench/featurebench` | `featurebench/pandas-dev__pandas.82fa2715.test_string.dcfa24ea.lv1` | 1 | `0cba9991f159e7e1` | 静态 + 动态 |
| `featurebench/featurebench-lite` | `featurebench/mwaskom__seaborn.7001ebe7.test_regression.ce8c62e2.lv1` | 2 | `657c381905d97be8` | 静态 + 动态 |
| `featurebench/featurebench-lite` | `featurebench/mwaskom__seaborn.7001ebe7.test_statistics.0f2ae277.lv1` | 2 | `4895c40832a4e4bb` | 静态 + 动态 |
| `featurebench/featurebench-lite` | `featurebench/pandas-dev__pandas.82fa2715.test_concat.ebe5de39.lv1` | 2 | `1d7dcfbbc012164e` | 静态 + 动态 |
| `featurebench/featurebench-lite` | `featurebench/pandas-dev__pandas.82fa2715.test_list_accessor.7ab0b2ea.lv1` | 2 | `77a2f5d3b64d8c01` | 静态 + 动态 |
| `featurebench/featurebench-lite` | `featurebench/pydantic__pydantic.e1dcaf9e.test_deprecated_fields.40a2ec54.lv1` | 2 | `43bf33602c8182cf` | 静态 + 动态 |
| `featurebench/featurebench-lite-modal` | `featurebench/mwaskom__seaborn.7001ebe7.test_regression.ce8c62e2.lv1` | 3 | `564e780f68f7ee02` | 静态 + 动态 |
| `featurebench/featurebench-lite-modal` | `featurebench/pandas-dev__pandas.82fa2715.test_iceberg.85771c70.lv2` | 3 | `1e6a66bdb1ff43b9` | 静态 + 动态 |
| `featurebench/featurebench-lite-modal` | `featurebench/pandas-dev__pandas.82fa2715.test_list_accessor.7ab0b2ea.lv1` | 3 | `05e000db19df1747` | 静态 + 动态 |
| `featurebench/featurebench-lite-modal` | `featurebench/pydantic__pydantic.e1dcaf9e.test_deprecated_fields.40a2ec54.lv1` | 3 | `95b4087c73f85907` | 静态 + 动态 |
| `featurebench/featurebench-lite-modal` | `featurebench/sphinx-doc__sphinx.e347e59c.test_domain_c.4068b9e8.lv1` | 3 | `1b39e5b7c527eb19` | 静态 + 动态 |
| `featurebench/featurebench-modal` | `featurebench/astropy__astropy.b0db0daa.test_tree.261bac3c.lv2` | 2 | `00df06909341dda9` | 静态 + 动态 |
| `featurebench/featurebench-modal` | `featurebench/mwaskom__seaborn.7001ebe7.test_regression.ce8c62e2.lv1` | 3 | `564e780f68f7ee02` | 静态 + 动态 |
| `featurebench/featurebench-modal` | `featurebench/pandas-dev__pandas.82fa2715.test_iceberg.85771c70.lv2` | 3 | `1e6a66bdb1ff43b9` | 静态 + 动态 |
| `featurebench/featurebench-modal` | `featurebench/pandas-dev__pandas.82fa2715.test_list_accessor.7ab0b2ea.lv1` | 3 | `05e000db19df1747` | 静态 + 动态 |
| `featurebench/featurebench-modal` | `featurebench/pydantic__pydantic.e1dcaf9e.test_deprecated_fields.40a2ec54.lv1` | 3 | `95b4087c73f85907` | 静态 + 动态 |
| `featurebench/featurebench-modal` | `featurebench/pytest-dev__pytest.68016f0e.raises_group.c28bf36a.lv1` | 2 | `005298b09b1e06e6` | 静态 + 动态 |
| `featurebench/featurebench-modal` | `featurebench/sphinx-doc__sphinx.e347e59c.test_domain_c.4068b9e8.lv1` | 3 | `1b39e5b7c527eb19` | 静态 + 动态 |
| `gaia/gaia` | `gaia/3f57289b-8c60-48be-bd80-01f8099ca449` | 1 | `e628ff0b7f71ace5` | 静态 + 动态 |
| `gaia/gaia` | `gaia/544b7f0c-173a-4377-8d56-57b36eb26ddf` | 1 | `6e9a80efcfc2c7ed` | 静态 + 动态 |
| `gaia/gaia` | `gaia/708b99c5-e4a7-49cb-a5cf-933c8d46470d` | 1 | `0e2a0a82400c7024` | 静态 + 动态 |
| `gaia/gaia` | `gaia/72e110e7-464c-453c-a309-90a95aed6538` | 1 | `4199afa58d890dd9` | 静态 + 动态 |
| `gaia/gaia` | `gaia/f46b4380-207e-4434-820b-f32ce04ae2a4` | 1 | `0135ab290ec1feb9` | 静态 + 动态 |
| `gorilla/bfcl` | `gorilla/live-multiple-800-175-6` | 1 | `000b2253e9a5fc4f` | 静态 + 动态 |
| `gorilla/bfcl` | `gorilla/multiple-2` | 1 | `19d400449f5a3bfb` | 静态 + 动态 |
| `gorilla/bfcl` | `gorilla/multiple-5` | 1 | `f4e70ea6b2d59124` | 静态 + 动态 |
| `gorilla/bfcl` | `gorilla/multiple-9` | 1 | `219f29f1693b975c` | 静态 + 动态 |
| `gorilla/bfcl` | `gorilla/parallel-1` | 1 | `2aefb5c97ca62b56` | 静态 + 动态 |
| `gorilla/bfcl_parity` | `gorilla/bfcl-live-multiple-898-185-8` | 1 | `03f1f371cd0a9efd` | 静态 + 动态 |
| `gorilla/bfcl_parity` | `gorilla/bfcl-multiple-16` | 1 | `78956ed8efb43dca` | 静态 + 动态 |
| `gorilla/bfcl_parity` | `gorilla/bfcl-parallel-5` | 1 | `59b39220746f7ff8` | 静态 + 动态 |
| `gorilla/bfcl_parity` | `gorilla/bfcl-parallel-7` | 1 | `0d590b18f47d1764` | 静态 + 动态 |
| `gorilla/bfcl_parity` | `gorilla/bfcl-parallel-84` | 1 | `85c2e871157c24c2` | 静态 + 动态 |
| `harbor/hello-world` | `hello-world/hello-world` | 2 | `38d7a077f07fbee8` | 静态 + 动态 |
| `ibragim-badertdinov/swe-rebench-07-2026` | `ibragim-badertdinov/apache__dubbo-go-3357` | 3 | `d25a8c026f058552` | 静态 + 动态 |
| `infra-bench/infra-bench-v1` | `infra-bench/cute-layout-composition` | 1 | `03cab8e43425ef7e` | 静态 + 动态 |
| `nvats/codeskills-bench` | `nvats/bench-cache-stale-after-write` | 1 | `ac2722cf9e0681ee` | 静态 + 动态 |
| `nvats/codeskills-bench` | `nvats/bench-circular-import-cold-load` | 1 | `1358b19fd4a2a1fa` | 静态 + 动态 |
| `nvats/codeskills-bench` | `nvats/bench-hidden-global-now-drift` | 1 | `2c12fa654033bdd7` | 静态 + 动态 |
| `nvats/codeskills-bench` | `nvats/bench-mutable-default-argument` | 1 | `5bfc2bc39f64cc51` | 静态 + 动态 |
| `nvats/codeskills-bench` | `nvats/bench-regex-greedy-cross-line` | 1 | `4f91ca69ca42036d` | 静态 + 动态 |
| `openthoughts/openthoughts-tblite` | `openthoughts/build-system-task-ordering` | 1 | `0c40824c8ee74dc2` | 静态 + 动态 |
| `openthoughts/openthoughts-tblite` | `openthoughts/cosign-keyless-signing` | 1 | `a745dfbe3f4f19c8` | 静态 + 动态 |
| `openthoughts/openthoughts-tblite` | `openthoughts/jsonl-aggregator` | 1 | `06cf0e4507ccd18d` | 静态 + 动态 |
| `openthoughts/openthoughts-tblite` | `openthoughts/mech-system` | 1 | `fc5759f82ccd964f` | 静态 + 动态 |
| `openthoughts/openthoughts-tblite` | `openthoughts/neutron-submission` | 1 | `18328024864717b6` | 静态 + 动态 |
| `openthoughts/tasktrove-freelancer-projects-sandboxes-ta-rl-gpt-5-mini-v2` | `openthoughts/tasktrove-freelancer-projects-sandboxes-ta-rl-gpt-5-mini-v2-freelancer_ta_rl_mini-08015` | 1 | `007a8c31afec138e` | 静态 + 动态 |
| `openthoughts/tasktrove-freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2` | `openthoughts/tasktrove-freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2-freelancer_ta_rl_nano-00561` | 1 | `6ce8d3a1a6054322` | 静态 + 动态 |
| `openthoughts/tasktrove-freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2` | `openthoughts/tasktrove-freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2-freelancer_ta_rl_nano-00882` | 1 | `006744b31bd2dd5c` | 静态 + 动态 |
| `openthoughts/tasktrove-nemotron-code-oracle-filtered` | `openthoughts/tasktrove-nemotron-code-oracle-filtered-nemotron-002411` | 1 | `062ec40ccee04c74` | 静态 + 动态 |
| `openthoughts/tasktrove-nemotron-code-oracle-filtered` | `openthoughts/tasktrove-nemotron-code-oracle-filtered-nemotron-006431` | 1 | `06606db4ed17fd64` | 静态 + 动态 |
| `openthoughts/tasktrove-nemotron-code-oracle-filtered` | `openthoughts/tasktrove-nemotron-code-oracle-filtered-nemotron-017554` | 1 | `05373d56b0ade006` | 静态 + 动态 |
| `openthoughts/tasktrove-nemotron-code-oracle-filtered` | `openthoughts/tasktrove-nemotron-code-oracle-filtered-nemotron-022843` | 1 | `01e1a3da2daca231` | 静态 + 动态 |
| `openthoughts/tasktrove-nemotron-code-oracle-filtered` | `openthoughts/tasktrove-nemotron-code-oracle-filtered-nemotron-025733` | 1 | `0006596237a8b01d` | 静态 + 动态 |
| `openthoughts/tasktrove-nl2bash-tasks-cleaned-oracle` | `openthoughts/tasktrove-nl2bash-tasks-cleaned-oracle-task_1178` | 1 | `02fc1bb81fbfcbf3` | 静态 + 动态 |
| `openthoughts/tasktrove-swe-rebench-patched-oracle` | `openthoughts/tasktrove-swe-rebench-patched-oracle-swe_rebench-00324` | 1 | `2577053536b51bb3` | 静态 + 动态 |
| `openthoughts/tasktrove-swe-rebench-patched-oracle` | `openthoughts/tasktrove-swe-rebench-patched-oracle-swe_rebench-01608` | 1 | `3875e745443e0eab` | 静态 + 动态 |
| `openthoughts/tasktrove-swe-rebench-patched-oracle` | `openthoughts/tasktrove-swe-rebench-patched-oracle-swe_rebench-03859` | 1 | `00ba099313f5cf1d` | 静态 + 动态 |
| `openthoughts/tasktrove-swe-rebench-patched-oracle` | `openthoughts/tasktrove-swe-rebench-patched-oracle-swe_rebench-04191` | 1 | `36386c4d3302b9f4` | 静态 + 动态 |
| `openthoughts/tasktrove-swe-rebench-patched-oracle` | `openthoughts/tasktrove-swe-rebench-patched-oracle-swe_rebench-04772` | 1 | `008a0914451c1c16` | 静态 + 动态 |
| `openthoughts/tasktrove-swe-rebench-v2-patched-oracle` | `openthoughts/tasktrove-swe-rebench-v2-patched-oracle-bem__bh-148` | 1 | `a3787693841afb39` | 静态 + 动态 |
| `openthoughts/tasktrove-swe-rebench-v2-patched-oracle` | `openthoughts/tasktrove-swe-rebench-v2-patched-oracle-pygfx__pygfx-561` | 1 | `00030251d599fd53` | 静态 + 动态 |
| `openthoughts/tasktrove-swegym-tasks-patched-validated-v2` | `openthoughts/tasktrove-swegym-tasks-patched-validated-v2-swegym-0525` | 1 | `5f225f9577b3bc63` | 静态 + 动态 |
| `openthoughts/tasktrove-swegym-tasks-patched-validated-v2` | `openthoughts/tasktrove-swegym-tasks-patched-validated-v2-swegym-0632` | 1 | `01a58aaeb07fc485` | 静态 + 动态 |
| `openthoughts/tasktrove-swesmith-oracle-filtered` | `openthoughts/tasktrove-swesmith-oracle-filtered-swesmith-03038` | 1 | `004c0444d572fea4` | 静态 + 动态 |
| `openthoughts/tasktrove-swesmith-oracle-filtered` | `openthoughts/tasktrove-swesmith-oracle-filtered-swesmith-24008` | 1 | `0a0626c5d21ae0ba` | 静态 + 动态 |
| `openthoughts/tasktrove-swesmith-oracle-filtered` | `openthoughts/tasktrove-swesmith-oracle-filtered-swesmith-24669` | 1 | `0eb39ecff89e47f0` | 静态 + 动态 |
| `openthoughts/tasktrove-swesmith-oracle-filtered` | `openthoughts/tasktrove-swesmith-oracle-filtered-swesmith-26302` | 1 | `0034a6b6950c36aa` | 静态 + 动态 |
| `openthoughts/tasktrove-swesmith-oracle-filtered` | `openthoughts/tasktrove-swesmith-oracle-filtered-swesmith-27384` | 1 | `00128d3014b9ef80` | 静态 + 动态 |
| `orinlabs/horizon-1-public` | `orinlabs/01-example-catering-vendor` | 1 | `b692c3d505d463f7` | 静态 + 动态 |
| `orinlabs/horizon-1-public` | `orinlabs/02-example-expense-code` | 1 | `bdad4a57eed80c2e` | 静态 + 动态 |
| `orinlabs/horizon-1-public` | `orinlabs/03-example-reading-preference` | 1 | `bca4fb9783c66702` | 静态 + 动态 |
| `qcircuitbench/qcircuitbench` | `qcircuitbench/diffusion_operator-n7` | 1 | `459ff0f19daf9db1` | 静态 + 动态 |
| `qcircuitbench/qcircuitbench` | `qcircuitbench/generalized_simon_ternary-n3` | 1 | `3a1e47d397e4aa9b` | 静态 + 动态 |
| `qcircuitbench/qcircuitbench` | `qcircuitbench/generalized_simon_ternary_oracle-n2` | 1 | `69dba5f26f7ab609` | 静态 + 动态 |
| `qcircuitbench/qcircuitbench` | `qcircuitbench/qae-n8` | 1 | `0469d6aae37b9a24` | 静态 + 动态 |
| `qcircuitbench/qcircuitbench` | `qcircuitbench/qaoa-n8` | 1 | `17d747dda1e54d14` | 静态 + 动态 |
| `quesma/compilebench` | `quesma/coreutils` | 1 | `f9ab9de8f38b4512` | 静态 + 动态 |
| `quesma/compilebench` | `quesma/cowsay` | 1 | `0fc9231545e50093` | 静态 + 动态 |
| `quesma/compilebench` | `quesma/curl-ssl-arm64-static` | 1 | `497b157478e66d9d` | 静态 + 动态 |
| `quesma/compilebench` | `quesma/jq` | 1 | `e74cd29d650d997b` | 静态 + 动态 |
| `quesma/compilebench` | `quesma/jq-static-musl` | 1 | `fa0d82a29144947d` | 静态 + 动态 |
| `quesma/otel-bench` | `quesma/cpp-simple` | 1 | `2bc7f479d29c4b58` | 静态 + 动态 |
| `quixbugs/quixbugs` | `quixbugs/python-gcd` | 1 | `18e11de3a29060c9` | 静态 + 动态 |
| `quixbugs/quixbugs` | `quixbugs/python-kth` | 1 | `4b70d4bec79f5cc9` | 静态 + 动态 |
| `quixbugs/quixbugs` | `quixbugs/python-lis` | 1 | `e0dd1f6befb2fb55` | 静态 + 动态 |
| `quixbugs/quixbugs` | `quixbugs/python-minimum_spanning_tree` | 1 | `fe13608cc32fcfad` | 静态 + 动态 |
| `quixbugs/quixbugs` | `quixbugs/python-next_palindrome` | 1 | `464849c5f4be1172` | 静态 + 动态 |
| `replicationbench/replicationbench` | `replicationbench/bayes_cal__cold_hot_tandem` | 1 | `0af1f2303d70d0b7` | 静态 + 动态 |
| `replicationbench/replicationbench` | `replicationbench/hubble_trails__satellite_chance_post2020_acis` | 1 | `0b2acf8f2223cc33` | 静态 + 动态 |
| `replicationbench/replicationbench` | `replicationbench/trgb_std_candle__med_color_amp` | 1 | `0e9e9d099cb15a78` | 静态 + 动态 |
| `replicationbench/replicationbench` | `replicationbench/ver_waves__gaia_breathing_typical` | 1 | `00c1fea3150f8f80` | 静态 + 动态 |
| `replicationbench/replicationbench` | `replicationbench/ver_waves__solar_height_from_gaia_dr2` | 1 | `01ab76e62a15ac19` | 静态 + 动态 |
| `scale-ai/swe-bench-pro` | `scale-ai/instance_ansible__ansible-1ee70fc272aff6bf3415357c6e13c5de5b928d9b-v1055803c3a812189a1133297f7f5468579283f86` | 2 | `009f37c067e17014` | 静态 + 动态 |
| `scale-ai/swe-bench-pro` | `scale-ai/instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5` | 2 | `0b7c2966021e5581` | 静态 + 动态 |
| `scale-ai/swe-bench-pro` | `scale-ai/instance_protonmail__webclients-7e54526774e577c0ebb58ced7ba8bef349a69fec` | 2 | `0aff2fb4f4ed69fc` | 静态 + 动态 |
| `scale-ai/swe-bench-pro` | `scale-ai/instance_qutebrowser__qutebrowser-c09e1439f145c66ee3af574386e277dd2388d094-v2ef375ac784985212b1805e1d0431dc8f1b3c171` | 2 | `04566e64df9fdfd2` | 静态 + 动态 |
| `scale-ai/swe-bench-pro` | `scale-ai/instance_qutebrowser__qutebrowser-e64622cd2df5b521342cf4a62e0d4cb8f8c9ae5a-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d` | 2 | `00e821a42061b948` | 静态 + 动态 |
| `swe-rebench/swe-rebench-leaderboard` | `swe-rebench/imperialcollegelondon__pyrealm-593` | 1 | `0033291247d93714` | 静态 + 动态 |
| `swe-rebench/swe-rebench-leaderboard` | `swe-rebench/marimo-team__marimo-6629` | 1 | `02a4fc21b31fe355` | 静态 + 动态 |
| `swe-rebench/swe-rebench-leaderboard` | `swe-rebench/sissbruecker__linkding-1291` | 1 | `001fdd2a4a2d555f` | 静态 + 动态 |
| `swe-rebench/swe-rebench-leaderboard` | `swe-rebench/stuartmaxwell__djpress-117` | 1 | `0046389d797f89ef` | 静态 + 动态 |
| `swe-rebench/swe-rebench-leaderboard` | `swe-rebench/sympy__sympy-28454` | 1 | `01f8fff3df45050f` | 静态 + 动态 |
| `terminal-bench-pro/terminal-bench-pro` | `terminal-bench-pro/bash-ddos-traffic-analyzer` | 1 | `323cdf1c656e2d41` | 静态 + 动态 |
| `terminal-bench-pro/terminal-bench-pro` | `terminal-bench-pro/fix-neural-net-weight-init` | 1 | `00c336105fcf7a62` | 静态 + 动态 |
| `terminal-bench-pro/terminal-bench-pro` | `terminal-bench-pro/implement-lz77-file-compressor` | 1 | `03e5efeda5e59826` | 静态 + 动态 |
| `terminal-bench-pro/terminal-bench-pro` | `terminal-bench-pro/implement-nonogram-puzzle-solver` | 1 | `046f239863e2ba0d` | 静态 + 动态 |
| `terminal-bench-pro/terminal-bench-pro` | `terminal-bench-pro/python-prime-http-server` | 1 | `729a98e4d34ec76f` | 静态 + 动态 |
| `usaco/usaco` | `usaco/1038` | 1 | `936b77e3ea72e8e5` | 静态 + 动态 |
| `usaco/usaco` | `usaco/84` | 1 | `d2d64d7ebd00125c` | 静态 + 动态 |
| `usaco/usaco` | `usaco/892` | 1 | `27b42556c44e235e` | 静态 + 动态 |
| `usaco/usaco` | `usaco/90` | 1 | `4882fde0648fca1c` | 静态 + 动态 |
| `usaco/usaco` | `usaco/93` | 1 | `318b00c48e362651` | 静态 + 动态 |

</details>

<details>
<summary>静态-only 代表任务（133 datasets）</summary>

| Dataset | Task | Revision | Digest | 筛选状态 |
| --- | --- | ---: | --- | --- |
| `abundant/swe-gen-java` | `abundant/apache__kafka-21001` | 1 | `8596827b93e47a42` | 仅静态 |
| `abundant/swe-marathon` | `abundant/biofabric-rust-rewrite` | 2 | `3af85c788d60b59e` | 仅静态 |
| `agentic-labs/erp-bench` | `agentic-labs/2000_easy_01_buy_only_baseline` | 1 | `1b58d1309d3b59da` | 仅静态 |
| `agentscope-ai/pawbench` | `agentscope-ai/t001-claweval-m005-score-canon` | 3 | `b209761e5a71ecf4` | 仅静态 |
| `algotune/algotune` | `algotune/aes-gcm-encryption` | 1 | `c5d96136f811818d` | 仅静态 |
| `bespokelabs/autoresearch-exam` | `bespokelabs/act-tensor-sparse-panel-imputation-r2` | 2 | `409976c1a5e58ac3` | 仅静态 |
| `bigcode/bigcodebench-hard-complete` | `bigcode/bigcodebench_100` | 1 | `130b023f5f8fc976` | 仅静态 |
| `binary-audit/binary-audit` | `binary-audit/caddy-backdoor-detect` | 1 | `a84ef1d351747004` | 仅静态 |
| `cmu/refav` | `cmu/refav__val_02678d04_1090` | 2 | `72b286a819fa7e6f` | 仅静态 |
| `codepde/codepde` | `codepde/advection` | 1 | `b7233fa3d16146f7` | 仅静态 |
| `cookbook/test` | `cookbook/hello-world` | 1 | `531479835b200538` | 仅静态 |
| `deveval/deveval` | `deveval/cpp-area-calculation-acceptance-testing` | 1 | `59a19d34dc11fab7` | 仅静态 |
| `gnucleus-ai/cad-bench` | `gnucleus-ai/freecad-0280cda4fc` | 1 | `b6b4d029d6e6d4fe` | 仅静态 |
| `gpqa-diamond/gpqa-diamond` | `gpqa-diamond/0` | 1 | `af796772e25f8e29` | 仅静态 |
| `islo-labs/reward-hack-bench` | `reward-hack-bench/matplotlib-25775__upstream-fix` | 1 | `35fbed9cf1e02bc1` | 仅静态 |
| `islo-labs/reward-hack-bench-control` | `reward-hack-bench/matplotlib-25775__control` | 1 | `0d8755d7d862b6bf` | 仅静态 |
| `ivanleo/agent-search` | `ivanleo/fts-q-1-managed-agents-security` | 1 | `68a865c934ea0b71` | 仅静态 |
| `lica-world/gdb` | `lica-world/gdb__category-1-s0` | 1 | `cf1c8cd1cc541920` | 仅静态 |
| `livecodebench/livecodebench` | `livecodebench/1873_d` | 1 | `c9afca70881c9357` | 仅静态 |
| `luosuu/SWE-kokkos-bench` | `luosuu/kokkos__kokkos-6289` | 4 | `8fb5bde319b0059c` | 仅静态 |
| `meta/mlgym-bench` | `meta/mlgym-battle-of-sexes` | 1 | `f843d35afc324d23` | 仅静态 |
| `nanoswe/swe-bench-verified` | `nanoswe/about` | 1 | `4eb97b99835817da` | 仅静态 |
| `nlile/registry-history-20260908` | `hello-world/hello-world` | 1 | `7f30a68bb54e6cf3` | 仅静态 |
| `open-design/open-design` | `open-design/dashboard` | 8 | `50910deeb93721c0` | 仅静态 |
| `openai/mmmlu` | `openai/ar-xy-00000` | 1 | `b18f61fc65db9a8e` | 仅静态 |
| `openthoughts/tasktrove-code-contests-noblock` | `openthoughts/tasktrove-code-contests-noblock-code_contests-0000` | 1 | `8199ca81ac4cecd6` | 仅静态 |
| `openthoughts/tasktrove-exp-flat25-pseudocode-v2` | `openthoughts/tasktrove-exp-flat25-pseudocode-v2-5k-pseudocode-0000` | 1 | `6582b45af3b348bd` | 仅静态 |
| `openthoughts/tasktrove-exp-flat25-speed-bonus-v2` | `openthoughts/tasktrove-exp-flat25-speed-bonus-v2-5k-speed-0000` | 1 | `074d5ed4fdb4f70f` | 仅静态 |
| `openthoughts/tasktrove-exp-flat25-stackoverflow-v2` | `openthoughts/tasktrove-exp-flat25-stackoverflow-v2-5k-stackoverflow-0000` | 1 | `f6cf16ca7755817a` | 仅静态 |
| `openthoughts/tasktrove-exp-flat25-subtle-debug-v3` | `openthoughts/tasktrove-exp-flat25-subtle-debug-v3-5k-subtle-0001` | 1 | `b4bd6ddcf24a31f3` | 仅静态 |
| `openthoughts/tasktrove-exp-rle-adversarial` | `openthoughts/tasktrove-exp-rle-adversarial-5k-adversarial-0000` | 1 | `11967784a92a7cce` | 仅静态 |
| `openthoughts/tasktrove-exp-rle-detailed-v3` | `openthoughts/tasktrove-exp-rle-detailed-v3-5k-detailed-0001` | 1 | `e503eea707b024ab` | 仅静态 |
| `openthoughts/tasktrove-exp-rle-error-report-v3` | `openthoughts/tasktrove-exp-rle-error-report-v3-5k-error-report-0000` | 1 | `31aa9dc2b6606089` | 仅静态 |
| `openthoughts/tasktrove-exp-rle-github-issue-v3` | `openthoughts/tasktrove-exp-rle-github-issue-v3-5k-github-issue-0000` | 1 | `a2623f3e71762551` | 仅静态 |
| `openthoughts/tasktrove-exp-rle-heavy-padding-v2` | `openthoughts/tasktrove-exp-rle-heavy-padding-v2-5k-heavy-0000` | 1 | `a6a000bf7c73d280` | 仅静态 |
| `openthoughts/tasktrove-exp-rle-minimal-instructions-v3` | `openthoughts/tasktrove-exp-rle-minimal-instructions-v3-5k-minimal-0000` | 1 | `50c2163dee7d3a0a` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-codenet-python-v2` | `openthoughts/tasktrove-exp-rpt-codenet-python-v2-codenet-python-0000` | 1 | `622a89eb84e86990` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-crosscodeeval-csharp-v4` | `openthoughts/tasktrove-exp-rpt-crosscodeeval-csharp-v4-crosscodeeval-csharp-0000` | 1 | `7b92ec5e00467bf6` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-crosscodeeval-java` | `openthoughts/tasktrove-exp-rpt-crosscodeeval-java-crosscodeeval-java-0000` | 1 | `a2f5cec532b4a8b4` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-curriculum-easy` | `openthoughts/tasktrove-exp-rpt-curriculum-easy-curriculum-easy-0000` | 1 | `25ddca806eae0e4c` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-curriculum-hard` | `openthoughts/tasktrove-exp-rpt-curriculum-hard-curriculum-hard-0000` | 1 | `ce2e1d03e17aa6b7` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-curriculum-medium` | `openthoughts/tasktrove-exp-rpt-curriculum-medium-curriculum-medium-0000` | 1 | `3989f9807b1a312f` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-defects4j-v3-v4` | `openthoughts/tasktrove-exp-rpt-defects4j-v3-v4-defects4j-0001` | 1 | `1a62f2691dcf01ca` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-e2egit-large` | `openthoughts/tasktrove-exp-rpt-e2egit-large-e2egit-0000` | 1 | `a27db63ea6749b01` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-e2egit-v2` | `openthoughts/tasktrove-exp-rpt-e2egit-v2-e2egit-0000` | 1 | `35910bda20dcf1ff` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-exercism-python-v2` | `openthoughts/tasktrove-exp-rpt-exercism-python-v2-exercism-python-0000` | 1 | `181a184eb19c6f02` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-ghactions-v3` | `openthoughts/tasktrove-exp-rpt-ghactions-v3-ghactions-0000` | 1 | `07c183c862d50bdc` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-issue` | `openthoughts/tasktrove-exp-rpt-issue-issue-0000` | 1 | `97698d5331b5cd31` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-methods2test-large-v2` | `openthoughts/tasktrove-exp-rpt-methods2test-large-v2-methods2test-0000` | 1 | `c8a68ad4cbeef138` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-methods2test-large-v3` | `openthoughts/tasktrove-exp-rpt-methods2test-large-v3-methods2test-0000` | 1 | `93466c211087c0a1` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-multifile` | `openthoughts/tasktrove-exp-rpt-multifile-multifile-0000` | 1 | `5f20f9d3129ce13d` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-nemotron-cpp` | `openthoughts/tasktrove-exp-rpt-nemotron-cpp-nemotron-cpp-0000` | 1 | `61f625713b8f5d85` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-nemotron-junit` | `openthoughts/tasktrove-exp-rpt-nemotron-junit-nemotron-junit-0000` | 1 | `f081e926557e22dc` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-pr` | `openthoughts/tasktrove-exp-rpt-pr-pr-0000` | 1 | `f98eb1162d5ee3be` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-pr-v2` | `openthoughts/tasktrove-exp-rpt-pr-v2-pr-0000` | 1 | `45c28f2790bbcde0` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-pymethods2test-large` | `openthoughts/tasktrove-exp-rpt-pymethods2test-large-pymethods2test-0000` | 1 | `afda979e7607fd19` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-pymethods2test-v3` | `openthoughts/tasktrove-exp-rpt-pymethods2test-v3-pymethods2test-0000` | 1 | `972c465d6b1933c5` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-scaffold-v2` | `openthoughts/tasktrove-exp-rpt-scaffold-v2-scaffold-0000` | 1 | `66aec877181d05ea` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-bash-v3` | `openthoughts/tasktrove-exp-rpt-stack-bash-v3-stack-bash-0000` | 1 | `0e9f8beadf533eed` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-bash-withtests-gpt5mini-v2` | `openthoughts/tasktrove-exp-rpt-stack-bash-withtests-gpt5mini-v2-stack-bash-withtests-gpt4o-0001` | 1 | `29b1f99a44731e58` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-bash-withtests-v2` | `openthoughts/tasktrove-exp-rpt-stack-bash-withtests-v2-stack-bash-withtests-0001` | 1 | `1377e9907a49a1f1` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-csharp-v5` | `openthoughts/tasktrove-exp-rpt-stack-csharp-v5-stack-csharp-0000` | 1 | `66b7d7a4afe0b48e` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-dockerfile-gpt5mini-v3` | `openthoughts/tasktrove-exp-rpt-stack-dockerfile-gpt5mini-v3-stack-dockerfile-gpt4o-0000` | 1 | `67da7df6959102af` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-dockerfile-v2` | `openthoughts/tasktrove-exp-rpt-stack-dockerfile-v2-stack-dockerfile-0001` | 1 | `d72df038642c3443` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-go-v4` | `openthoughts/tasktrove-exp-rpt-stack-go-v4-stack-go-0001` | 1 | `0084456987381463` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-jest-large` | `openthoughts/tasktrove-exp-rpt-stack-jest-large-stack-jest-0000` | 1 | `42657220cc1b3e93` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-jest-v2` | `openthoughts/tasktrove-exp-rpt-stack-jest-v2-stack-jest-0000` | 1 | `434fffdc318f6b3c` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-junit-v6` | `openthoughts/tasktrove-exp-rpt-stack-junit-v6-stack-junit-0000` | 1 | `6ba9614402385092` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-php-large-v6` | `openthoughts/tasktrove-exp-rpt-stack-php-large-v6-stack-php-0000` | 1 | `7bb0e665ebf4d0ed` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-php-v2-v6` | `openthoughts/tasktrove-exp-rpt-stack-php-v2-v6-stack-php-0000` | 1 | `1b943f35b2b37368` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-pytest-large` | `openthoughts/tasktrove-exp-rpt-stack-pytest-large-stack-pytest-0000` | 1 | `29c73689e963a6b7` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-pytest-v2` | `openthoughts/tasktrove-exp-rpt-stack-pytest-v2-stack-pytest-0000` | 1 | `ac7bae0bd7398dbf` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-ruby-v2` | `openthoughts/tasktrove-exp-rpt-stack-ruby-v2-stack-ruby-0000` | 1 | `70528b20d33d41a7` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-stack-rust-v2` | `openthoughts/tasktrove-exp-rpt-stack-rust-v2-stack-rust-0000` | 1 | `b47d10cd0d98d397` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-taco-v2` | `openthoughts/tasktrove-exp-rpt-taco-v2-taco-0000` | 1 | `469193ec2431e73f` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-unitsyn-python-large` | `openthoughts/tasktrove-exp-rpt-unitsyn-python-large-unitsyn-python-0000` | 1 | `609689aad275f145` | 仅静态 |
| `openthoughts/tasktrove-exp-rpt-unitsyn-python-v3` | `openthoughts/tasktrove-exp-rpt-unitsyn-python-v3-unitsyn-python-0000` | 1 | `b445045af3c91e15` | 仅静态 |
| `openthoughts/tasktrove-inferredbugs-sandboxes-verifier` | `openthoughts/tasktrove-inferredbugs-sandboxes-verifier-inferredbugs-0729` | 1 | `0e81c13e6571e407` | 仅静态 |
| `openthoughts/tasktrove-mix-baseline-uniform-v2` | `openthoughts/tasktrove-mix-baseline-uniform-v2-mix-baseline-uniform-0000` | 1 | `2a5ccca80b9f7fc3` | 仅静态 |
| `openthoughts/tasktrove-mix-h1-struggle-zone-v2` | `openthoughts/tasktrove-mix-h1-struggle-zone-v2-mix-h1-struggle-zone-0000` | 1 | `c16fa05942d44141` | 仅静态 |
| `openthoughts/tasktrove-mix-h10-reward-binary-v2` | `openthoughts/tasktrove-mix-h10-reward-binary-v2-mix-h10-binary-0000` | 1 | `f1dde760e97d07d2` | 仅静态 |
| `openthoughts/tasktrove-mix-h10-reward-proportional-v2` | `openthoughts/tasktrove-mix-h10-reward-proportional-v2-mix-h10-proportional-0000` | 1 | `daee78dabe7f5cff` | 仅静态 |
| `openthoughts/tasktrove-mix-h10-reward-staged-v2` | `openthoughts/tasktrove-mix-h10-reward-staged-v2-mix-h10-staged-0000` | 1 | `a1ce76a6fce62522` | 仅静态 |
| `openthoughts/tasktrove-mix-h11-compositional-gradient-v2` | `openthoughts/tasktrove-mix-h11-compositional-gradient-v2-mix-h11-compositional-0000` | 1 | `07e0c7c2a4324cca` | 仅静态 |
| `openthoughts/tasktrove-mix-h11-single-skill-only-v2` | `openthoughts/tasktrove-mix-h11-single-skill-only-v2-mix-h11-single-skill-0002` | 1 | `e28c4ff073f53d16` | 仅静态 |
| `openthoughts/tasktrove-mix-h2-language-balanced-v2` | `openthoughts/tasktrove-mix-h2-language-balanced-v2-mix-h2-lang-balanced-0000` | 1 | `4d9cf80c43d413a7` | 仅静态 |
| `openthoughts/tasktrove-mix-h2-language-proportional` | `openthoughts/tasktrove-mix-h2-language-proportional-mix-h2-lang-proportional-0000` | 1 | `7cd494489a0e280a` | 仅静态 |
| `openthoughts/tasktrove-mix-h4-binary-easy` | `openthoughts/tasktrove-mix-h4-binary-easy-mix-h4-binary-easy-0000` | 1 | `aae1ec58cb12031f` | 仅静态 |
| `openthoughts/tasktrove-mix-h5-skill-diverse-v2` | `openthoughts/tasktrove-mix-h5-skill-diverse-v2-mix-h5-skill-diverse-0000` | 1 | `a5274071f03d4219` | 仅静态 |
| `openthoughts/tasktrove-mix-h6-test-quality-top25` | `openthoughts/tasktrove-mix-h6-test-quality-top25-mix-h6-test-quality-0000` | 1 | `add999b2c8ef28d0` | 仅静态 |
| `openthoughts/tasktrove-mix-h7-raw-volume-5k-v2` | `openthoughts/tasktrove-mix-h7-raw-volume-5k-v2-mix-h7-raw-volume-0000` | 1 | `863d310dae74e65c` | 仅静态 |
| `openthoughts/tasktrove-mix-h8-adversarial-tests-v2` | `openthoughts/tasktrove-mix-h8-adversarial-tests-v2-mix-h8-adversarial-0000` | 1 | `6900a59866a39977` | 仅静态 |
| `openthoughts/tasktrove-mix-h8-original-tests-v2` | `openthoughts/tasktrove-mix-h8-original-tests-v2-mix-h8-original-0000` | 1 | `da49d1ae0faec1f8` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-agent-calendar` | `openthoughts/tasktrove-nemotron-gym-agent-calendar-agent-calendar-0014e340a9bc.tar.gz` | 1 | `86ba716aa2a439b1` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-agent-workplace-v2` | `openthoughts/tasktrove-nemotron-gym-agent-workplace-v2-agent-workplace-v2-0043f75a71ee.tar.gz` | 1 | `508f133ef22b89ee` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-competitive-coding` | `openthoughts/tasktrove-nemotron-gym-competitive-coding-comp-coding-00035cfe25d9.tar.gz` | 1 | `41636d74f9141317` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-identity-following-v2` | `openthoughts/tasktrove-nemotron-gym-identity-following-v2-identity-000047d145c9.tar.gz` | 1 | `1124d4d2f614e13e` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-instruction-following-adversarial-v3` | `openthoughts/tasktrove-nemotron-gym-instruction-following-adversarial-v3-adversarial-00016b6f2462.tar.gz` | 1 | `8fec4c52f7f38881` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-instruction-following-calendar` | `openthoughts/tasktrove-nemotron-gym-instruction-following-calendar-agent-calendar-0008b3a7a4a9.tar.gz` | 1 | `1e60ce66988f2170` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-instruction-following-structured` | `openthoughts/tasktrove-nemotron-gym-instruction-following-structured-if-structured-00030a76910e.tar.gz` | 1 | `0cd56d77511903de` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-instruction-following-v2` | `openthoughts/tasktrove-nemotron-gym-instruction-following-v2-Nemotron-RL-instruction_following-00004bfc2a10.tar.gz` | 1 | `8814e0dddd2894c3` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-knowledge-web-search-mcqa` | `openthoughts/tasktrove-nemotron-gym-knowledge-web-search-mcqa-Nemotron-RL-knowledge-web_search-mcqa-001745793222.tar.gz` | 1 | `6d5ef40b0591a300` | 仅静态 |
| `openthoughts/tasktrove-nemotron-gym-math-advanced-calculations-v3` | `openthoughts/tasktrove-nemotron-gym-math-advanced-calculations-v3-math-advcalc-0007702460b8.tar.gz` | 1 | `de2cf3d76afcc884` | 仅静态 |
| `openthoughts/tasktrove-openswe-tasks-patched-v5-oracle-success` | `openthoughts/tasktrove-openswe-tasks-patched-v5-oracle-success-openswe_oss-00000` | 1 | `4185392ca2a08d62` | 仅静态 |
| `openthoughts/tasktrove-r2egym-patched-full-oracle` | `openthoughts/tasktrove-r2egym-patched-full-oracle-r2egym-v1-00005` | 1 | `4f509b8ab39c75cf` | 仅静态 |
| `openthoughts/tasktrove-selfinstruct-naive-sandboxes-2-verified` | `openthoughts/tasktrove-selfinstruct-naive-sandboxes-2-verified-selfinstruct-naive-00043` | 1 | `f8d6482f22499eb3` | 仅静态 |
| `orinlabs/horizon-public` | `orinlabs/01-example-catering-vendor` | 3 | `d2cfce37b07045ba` | 仅静态 |
| `red-hat-ai/SWE-benchify-hard` | `red-hat-ai/containers--image-90028` | 2 | `96dee7272c4a6953` | 仅静态 |
| `red-hat-ai/haiku-hard` | `red-hat-ai/containers--storage-90043` | 3 | `65b552c657cda2e1` | 仅静态 |
| `rounakbende10/rh-swe-bench` | `rounakbende10/task-0000` | 3 | `3d52577d2114b5ff` | 仅静态 |
| `ryanmarten/tb4-preview` | `ryanmarten/atrx-vep-crispr` | 1 | `e50578a5bd67543a` | 仅静态 |
| `ryanmarten/terminal-bench-4-no-prebuild` | `ryanmarten/terminal-bench-4-no-prebuild-atrx-vep-crispr` | 1 | `0e44eaf15aecb9c8` | 仅静态 |
| `scienceagentbench/scienceagentbench` | `scienceagentbench/sab_1` | 1 | `86a923579d6d2b5b` | 仅静态 |
| `snorkel-ai/senior-swe-bench-v2026.06` | `snorkel-ai/better-auth-fix-api-key-run` | 3 | `39bcec71e8b4e9c0` | 仅静态 |
| `svgap/svgap-reset-release` | `svgap/reset-config` | 1 | `54206c28db8e4ebd` | 仅静态 |
| `swe-bench/swe-bench-verified` | `swe-bench/astropy__astropy-12907` | 2 | `ffa0f9b1e9f60875` | 仅静态 |
| `swe-bench/swe-smith` | `swe-bench/oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc` | 1 | `20edeb98e497d524` | 仅静态 |
| `swt-bench/swt-bench-verified` | `swt-bench/astropy__astropy-12907` | 1 | `4e5a470c5e97c652` | 仅静态 |
| `tencent/autocodebench` | `tencent/cpp_001` | 1 | `206591ffe12bd425` | 仅静态 |
| `termigen/termigen-environments` | `termigen/R_package_dependency_missing_medium` | 1 | `7e67a3111740517d` | 仅静态 |
| `terminal-bench-science/terminal-bench-science` | `terminal-bench-science/3x2pt-inference` | 3 | `ac4d76da52625a15` | 仅静态 |
| `theagentcompany/theagentcompany` | `theagentcompany/admin-arrange-meeting-rooms` | 1 | `bd74849620f536fe` | 仅静态 |
| `tinycomputerai/bun-server-bench` | `tinycomputerai/bun-server-bench-authentication-bearer-profile-v1` | 3 | `d61dbe019a2dbc5a` | 仅静态 |
| `tmax/TMax-15K-Harbor` | `tmax/task_000000_c19dda5b` | 2 | `e201b20a1e2ba13f` | 仅静态 |
| `tmax/TMax-15K-Harbor-trial` | `tmax/task_000000_c19dda5b` | 1 | `c8d2a7db2a132797` | 仅静态 |
| `userbench/UserBench` | `userbench/135yshr__196168af` | 9 | `230d9d893789f509` | 仅静态 |
| `userbench/UserBench-train400` | `userbench/135yshr__196168af` | 10 | `3459348c11ced0e0` | 仅静态 |
| `vals/financeagent` | `vals/0` | 1 | `244d642bf0e96e4c` | 仅静态 |
| `vmax/vmax-tasks` | `vmax/chalk__chalk.51557784.stringEncaseCRLFWithFirstIndex.134f7356` | 1 | `caea4047e528c5e0` | 仅静态 |
| `webgen-bench/webgen-bench` | `webgen-bench/000001` | 2 | `1cf4a40c4a44c0f5` | 仅静态 |
| `withmartian/clojure-swerebench-v2-validated-51` | `withmartian/amperity___lein-monolith-61` | 1 | `cce91ac613206f71` | 仅静态 |
| `xiaoboai/pawbench` | `xiaoboai/003-output-preference` | 1 | `17e740da80136f4b` | 仅静态 |
| `zenml/zenml-bench` | `zenml/b3-stale-cache` | 1 | `876b7a2f9c7bbc24` | 仅静态 |

</details>
