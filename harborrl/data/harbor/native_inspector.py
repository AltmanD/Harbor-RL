"""Native preflight: task metadata, not TerminalEnv materialization rules."""
from pathlib import Path
from .inspector import tree_digest, tomllib


def inspect_native(path, *, expected_digest=None):
    root = Path(path)
    reasons = []
    try:
        if not root.is_dir():
            raise ValueError("task path is not a directory")
        checksum = tree_digest(root)
        if expected_digest is not None and checksum != expected_digest:
            raise ValueError("task content changed since catalog lock")
        spec = tomllib.loads((root / "task.toml").read_text())
        for name in ("instruction.md", "environment/Dockerfile", "tests/test.sh"):
            if not (root / name).is_file():
                raise ValueError("missing " + name)
        if not (root / "instruction.md").read_text().strip():
            raise ValueError("empty instruction")
        env, verifier = spec.get("environment", {}), spec.get("verifier", {})
        if not isinstance(env, dict) or not isinstance(verifier, dict):
            raise ValueError("invalid environment/verifier configuration")
        if not isinstance(spec.get("agent", {}), dict):
            raise ValueError("invalid agent configuration")
        if spec.get("user_agent") or spec.get("bridge"):
            reasons.append("simulated_user_or_bridge_not_enabled")
        if spec.get("steps"):
            reasons.append("multi_step_not_enabled")
        if verifier.get("environment", {}).get("type") if isinstance(verifier.get("environment"), dict) else verifier.get("environment"):
            reasons.append("separate_verifier_needs_probe")
        if verifier.get("environment_mode", "shared") != "shared":
            reasons.append("separate_verifier_not_enabled")
        if env.get("gpus", 0) or env.get("tpu"):
            reasons.append("accelerator_task_requires_resources")
        if env.get("os", "linux") != "linux":
            reasons.append("non_linux_not_enabled")
        if any((root / "environment" / name).exists() for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")):
            reasons.append("compose_profile_not_enabled")
        return {"path": str(root.resolve()), "task_digest": checksum,
                "status": "UNSUPPORTED" if reasons else "NEEDS_PROBE", "reasons": reasons,
                "resources": env, "profile": "native_single_step_text_v1"}
    except (ValueError, OSError, TypeError) as exc:
        return {"path": str(root), "status": "INVALID", "reasons": [str(exc)]}
