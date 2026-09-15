# HarborRL slime upstream sync

HarborRL vendors and modifies slime; this directory is not a clean upstream checkout. Do not replace it wholesale.

## Current selective sync

- Upstream: `THUDM/slime` main at `d38dc29c9d72c88d221e12c320fc6aa41982e5a2` (2026-08-12).
- Imported unchanged: `slime/utils/trace_utils.py`, `tools/trace_timeline_viewer.py`, the Chinese trace/observability docs, and the upstream trace unit test.
- Adapted locally: AgenticRL SGLang response metadata is attached to `Interaction`; environment open, model generation, tool calls, and evaluation emit trace spans; rollout metrics aggregate request timing once per trajectory.
- Launcher integration: optional `SGLANG_SERVER_CONCURRENCY`, `SLIME_USE_FAULT_TOLERANCE`, and `SLIME_SAVE_DEBUG_ROLLOUT_DATA` environment variables.

## Deliberately not merged

- `examples/fully_async`: not used for a formal DAPO/DIVE-PO comparison. It changes which completed groups enter each update and therefore changes policy staleness/on-policy semantics; upstream also documents evaluation/resume limitations.
- Whole-tree SGLang/router/Megatron updates: the deployment image is pinned to an older HarborRL runtime stack. These require a separate container upgrade and GPU smoke test.
- Async checkpoint changes: current checkpoint writes account for about 0.14% of the first 100 rollout wall time, so they do not address the active bottleneck.

## Update procedure

1. Fetch official main and record the exact commit.
2. Compare against the local vendored tree and preserve HarborRL-specific rollout, DAPO, DIVE-PO, eval scheduling, and diagnostics changes.
3. Backport a bounded feature with its upstream unit test.
4. Run CPU tests, then a 1–3 rollout GPU smoke before changing a formal run.
5. Use identical source and execution profiles for DAPO and DIVE-PO.
