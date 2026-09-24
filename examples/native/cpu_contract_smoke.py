#!/usr/bin/env python3
"""Offline 0/1 reward contract check for the public native example."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from harborrl.backends.slime_v032.converter import (
    STANDARD_FIELDS,
    convert_samples_to_train_data,
)
from harborrl.backends.slime_v032.postprocess import rollout_data_postprocess
from harborrl.backends.slime_v032.rollout import to_slime_samples
from harborrl.data.harbor.native_inspector import inspect_native
from harborrl.export.native import export_training_batch
from harborrl.export.validate import batch_weight_sum
from harborrl.rollout.harbor_job.collector import collect
from harborrl.trajectories.native import (
    Identity,
    RewardProfile,
    assemble,
    digest,
    publish,
)

SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_new_tokens": 8,
    "stop": [],
}


def make_identity(slot: str, *, task_digest: str) -> Identity:
    return Identity(
        run_id="cpu-contract-smoke",
        batch_id="example-batch",
        group_id="hello-world-group",
        slot_id=slot,
        attempt_id=f"fixture-attempt-{slot}",
        trajectory_id=f"fixture-trajectory-{slot}",
        task_digest=task_digest,
        policy_version="fixture-v1",
        harness_digest="claude-code-profile",
        reward_digest=RewardProfile().digest,
        sampling_digest=digest(SAMPLING),
    )


def make_ir(root: Path, slot: str, reward: int, *, task_digest: str) -> dict:
    identity = make_identity(slot, task_digest=task_digest)
    trial_dir = root / identity.attempt_id
    trial_dir.mkdir()
    trial_id = f"fixture-trial-{slot}"
    terminal = {
        "id": trial_id,
        "finished_at": "2026-01-01T00:00:00Z",
        "verifier_environment_mode": "shared",
        "verifier_result": {"rewards": {"reward": reward}},
    }
    result_path = trial_dir / "result.json"
    result_path.write_text(json.dumps(terminal, sort_keys=True) + "\n")
    (trial_dir / "reward.json").write_text(json.dumps({"reward": reward}) + "\n")
    evaluation = collect(
        identity,
        trial_id,
        result_path,
        trial_dir,
        RewardProfile(),
        evidence_kind="harbor",
    )
    turn = {
        "identity": identity.to_dict(),
        "harbor_trial_id": trial_id,
        "request_id": f"request-{slot}",
        "response_id": f"response-{slot}",
        "engine_id": "offline-fixture",
        "input_ids": [1, 2, 3],
        "output_ids": [4, 5],
        "logprobs": [-0.20, -0.30],
        "logprob_semantics": "raw_model",
        "policy_version": identity.policy_version,
        "tokenizer_digest": "0" * 64,
        "template_digest": "1" * 64,
        "delivery": "consumed_confirmed",
        "consumption_ref": "offline-fixture#consumed",
        "role": "policy",
        "sampling": SAMPLING,
        "client_request": {"model": "policy"},
        "serving_input": {"input_ids": [1, 2, 3]},
        "response": {"content": [{"type": "text", "text": "ok"}]},
        "finish_reason": "end_turn",
        "evidence_kind": "serving",
    }
    for key in ("client_request", "serving_input", "response"):
        turn[key + "_digest"] = digest(turn[key])
    seal = {
        "identity": identity.to_dict(),
        "harbor_trial_id": trial_id,
        "pending_requests": 0,
        "status": "sealed",
        "request_ids": [turn["request_id"]],
        "turns_digest": digest([turn]),
        "errors": [],
    }
    return assemble(identity, trial_id, [turn], seal, evaluation)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    task = root / "examples/native/tasks/hello_world"
    report = inspect_native(task)
    if report["status"] != "NEEDS_PROBE":
        raise SystemExit(f"hello_world inspection failed: {report}")

    with tempfile.TemporaryDirectory(prefix="harborrl-smoke-") as temporary:
        evidence = Path(temporary)
        groups = [[
            make_ir(evidence, "0", 0, task_digest=report["task_digest"]),
            make_ir(evidence, "1", 1, task_digest=report["task_digest"]),
        ]]
        publish(evidence / "reward-receipt.json", groups[0][0]["evaluation"])
        publish(evidence / "native-ir.json", groups[0][0])
        batch = export_training_batch(groups, rollout_ids=[7])
        samples = [
            sample
            for group in to_slime_samples(
                batch, groups, sample_factory=SimpleNamespace
            )
            for sample in group
        ]
        args = SimpleNamespace(n_samples_per_prompt=2)
        training_data = convert_samples_to_train_data(args, samples)
        rollout_data_postprocess(args, 7, training_data)

        assert tuple(training_data) == STANDARD_FIELDS
        assert training_data["raw_reward"] == [0.0, 1.0]
        assert training_data["rewards"][0] < 0.0 < training_data["rewards"][1]
        assert math.isclose(sum(training_data["rewards"]), 0.0, abs_tol=1e-9)
        assert batch_weight_sum(batch) == 1.0
        print(json.dumps({
            "task_digest": report["task_digest"],
            "reward_contrast": training_data["raw_reward"],
            "advantages": training_data["rewards"],
            "rollout_ids": training_data["rollout_ids"],
            "status": "passed",
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
