# Evaluation layout

The evaluation stack has three deliberately separate layers:

1. `examples/evaluation/` contains user recipes only. It is safe to copy and
   customize a recipe without importing implementation details.
2. `tools/evaluation/` is the reusable orchestration package. `core/` owns
   config parsing, serving lifecycle, runner, batch execution, and reports;
   benchmark-specific validators and launch scripts live under
   `benchmarks/<name>/`, keeping each benchmark's adapters, validators, and
   launchers together.
3. `harborrl/harnesses/eval/` adapts each external evaluator to the common
   `BaseEvalHarness` data model. It does not share mutable rollout state with
   training harnesses.

Math RLVR is a deliberately dependency-light benchmark package under
`tools/evaluation/math_rlvr/`. Its data/extractor/verifier/scorer/stats modules
are independent of the generic harness lifecycle; rescoring is implemented by
the scorer module so online and offline policies cannot drift. The compatibility
entrypoints `eval_math.py`, `prepare_math_data.py`, `rescore_math_eval.py` and
`math_paired_stats.py` are safe to call from an RJob. Scheduler-facing Math RLVR
submitters are operator-provided local overlays (commonly under
`local/rjob/`) and read site-specific settings from the environment; they are
not part of the public source tree.

The canonical command is:

```bash
python -m tools.evaluation <run|smoke|batch|report> ...
```

`eval_cli.py` remains a compatibility shim for existing jobs. New benchmark
integrations should add an adapter under `harborrl/harnesses/eval/`, a config
template under `tools/evaluation/configs/`, and a recipe under
`examples/evaluation/`; serving and report logic should not be duplicated.
