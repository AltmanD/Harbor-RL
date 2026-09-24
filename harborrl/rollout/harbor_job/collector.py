"""Per-trial collection. Reward artifacts audit, but never replace, terminal results."""
import hashlib
from pathlib import Path

from harborrl.trajectories.native import publish, read_json


def collect(identity, trial_id, result_path, verifier_dir, profile, *, evidence_kind="harbor"):
    if profile.digest != identity.reward_digest:
        raise ValueError("reward profile differs from attempt manifest")
    result_path, verifier_dir = Path(result_path), Path(verifier_dir)
    # Caller retries incomplete files; no receipt is published before this succeeds.
    result = read_json(result_path)
    if result.get("id") != trial_id or not result.get("finished_at"):
        raise ValueError("missing terminal result or mismatched trial")
    if result.get("step_results") or result.get("verifier_environment_mode") != "shared":
        raise ValueError("only single-step shared verifier is enabled")
    if result.get("exception_info"):
        raise ValueError("exceptional trial requires a separately verified termination profile")
    rewards = (result.get("verifier_result") or {}).get("rewards")
    artifact = verifier_dir / "reward.json"
    if artifact.exists():
        artifact_rewards = read_json(artifact)
    else:
        artifact = verifier_dir / "reward.txt"
        artifact_rewards = {"reward": float(artifact.read_text().strip())}
    resolved = profile.resolve(rewards)
    profile.resolve(artifact_rewards)  # reject bool / nonfinite before numerical comparison
    if rewards != artifact_rewards:
        raise ValueError("reward artifact and terminal result differ")
    return {**resolved, "identity": identity.to_dict(), "harbor_trial_id": trial_id,
            "generation_attempt_id": identity.attempt_id, "evaluation_attempt_id": identity.attempt_id,
            "evaluation_trial_id": trial_id, "source_trial_id": trial_id,
            "verifier_environment_mode": "shared", "evaluation_status": "valid",
            "evidence_kind": evidence_kind, "source_result_ref": str(result_path),
            "source_result_digest": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "source_artifact_ref": str(artifact),
            "source_artifact_digest": hashlib.sha256(artifact.read_bytes()).hexdigest()}


def commit_receipt(path, *args, **kwargs):
    receipt = collect(*args, **kwargs)
    publish(path, receipt)
    return receipt
