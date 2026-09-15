import hashlib
import json

from tools.evaluation.benchmarks.seta.audit_seta_fixed48_run import _audit_eval_artifact


def _write_artifact(run_dir, *, stream="seta_fixed48_explore", k=8):
    artifact_dir = run_dir / "evaluations" / stream / "step_0000"
    artifact_dir.mkdir(parents=True)
    rows = []
    for prompt_index in range(48):
        rewards = [float(prompt_index % 2 == 0 and sample_index == 0) for sample_index in range(k)]
        responses = [f"response-{prompt_index}-{sample_index}" for sample_index in range(k)]
        rows.append(
            {
                "group": f"prompt:{prompt_index}",
                "prompt_index": prompt_index,
                "task_id": f"task-{prompt_index}",
                "k": k,
                "rewards": rewards,
                "successes": [reward > 0 for reward in rewards],
                "pass_at_k": float(any(reward > 0 for reward in rewards)),
                "best_reward_at_k": max(rewards),
                "response_unique_ratio": 1.0,
                "response_pairwise_jaccard_distance": 0.5,
                "statuses": ["completed"] * k,
                "sampling_seeds": list(range(prompt_index * k, (prompt_index + 1) * k)),
                "response_sha256": [
                    hashlib.sha256(response.encode()).hexdigest() for response in responses
                ],
                "responses": responses,
            }
        )
    summary = {
        "dataset": stream,
        "global_step": 0,
        "eval/task_count": 48,
        "eval/k": k,
        "eval/pass_at_k": 0.5,
        "eval/reward_best_at_k": 0.5,
        "eval/response_unique_ratio": 1.0,
        "eval/response_pairwise_jaccard_distance": 0.5,
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary))
    (artifact_dir / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return artifact_dir


def test_audit_eval_artifact_accepts_complete_task_level_records(tmp_path):
    _write_artifact(tmp_path)
    errors = []
    rows = _audit_eval_artifact(
        tmp_path, "seta_fixed48_explore", 0, 8, errors
    )
    assert errors == []
    assert set(rows) == set(range(48))


def test_audit_eval_artifact_rejects_response_hash_mismatch(tmp_path):
    artifact_dir = _write_artifact(tmp_path, stream="seta_fixed48_exploit", k=1)
    rows = [json.loads(line) for line in (artifact_dir / "tasks.jsonl").read_text().splitlines()]
    rows[7]["response_sha256"][0] = "0" * 64
    (artifact_dir / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    errors = []
    _audit_eval_artifact(tmp_path, "seta_fixed48_exploit", 0, 1, errors)
    assert any("response hashes disagree" in error for error in errors)
