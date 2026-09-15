"""Immutable terminal records; an exclusive link publishes a complete JSON file."""

import json
import os
import math
from pathlib import Path
import tempfile
from .schema import Trajectory


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def save(ir: Trajectory, root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # IDs are hashed for filenames; identities remain intact in the payload.
    import hashlib

    key = hashlib.sha256(f"{ir.trajectory_id}/{ir.attempt_id}".encode()).hexdigest()
    target = root / f"{key}.json"
    payload = (
        json.dumps(
            _json_value(ir.to_dict()),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )
    fd, temp = tempfile.mkstemp(dir=root, prefix=".ir-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, target)
    finally:
        os.unlink(temp)
    return target


def load(path: Path) -> Trajectory:
    return Trajectory.from_dict(json.loads(Path(path).read_text()))
