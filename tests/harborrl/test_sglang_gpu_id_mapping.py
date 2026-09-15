import ast
import os
from pathlib import Path

import pytest


def _load_gpu_mapper():
    """Load only the pure helper without importing SGLang's process runtime."""
    source_path = (
        Path(__file__).parents[2]
        / "backends"
        / "slime"
        / "slime"
        / "backends"
        / "sglang_utils"
        / "sglang_engine.py"
    )
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_to_local_gpu_id"
    )
    namespace = {"os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_to_local_gpu_id"]


_to_local_gpu_id = _load_gpu_mapper()


def test_uuid_cuda_visible_devices_uses_local_ordinals(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d,GPU-e")

    assert _to_local_gpu_id(0) == 0
    assert _to_local_gpu_id(4) == 4


def test_uuid_cuda_visible_devices_rejects_out_of_range_ordinal(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    with pytest.raises(RuntimeError, match="local id in 0..1"):
        _to_local_gpu_id(2)


def test_numeric_cuda_visible_devices_remaps_physical_id(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert _to_local_gpu_id(6) == 2
    assert _to_local_gpu_id(1) == 1
