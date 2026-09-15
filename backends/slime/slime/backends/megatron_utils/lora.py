from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

logger = logging.getLogger(__name__)

ADAPTER_TRACKER_FILE = "latest_adapter_checkpointed_iteration.txt"
STANDARD_TRACKER_FILE = "latest_checkpointed_iteration.txt"


def _split_target_modules(target_modules: str | None) -> list[str]:
    if not target_modules:
        return []
    return [x.strip() for x in target_modules.split(",") if x.strip()]


def _target_matches(module_name: str, targets: Iterable[str]) -> bool:
    leaf_name = module_name.rsplit(".", 1)[-1]
    return any(target == leaf_name or target in module_name for target in targets)


def _is_supported_linear(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.nn.Parameter) and weight.ndim == 2


def _set_lora_param_attrs(param: torch.nn.Parameter, *, average_across_tp: bool) -> None:
    # DDP should reduce LoRA grads across data-parallel ranks. For duplicated
    # non-TP adapters, also keep TP ranks aligned by averaging across TP.
    setattr(param, "allreduce", True)
    setattr(param, "tensor_model_parallel", False)
    if average_across_tp:
        setattr(param, "average_gradients_across_tp_domain", True)


def _parallel_mode(module: torch.nn.Module) -> str | None:
    mode = getattr(module, "parallel_mode", None)
    if mode in {"column", "row", "duplicated"}:
        return mode
    name = type(module).__name__
    if name.endswith("ColumnParallelLinear"):
        return "column"
    if name.endswith("RowParallelLinear"):
        return "row"
    return mode


def _tp_group(module: torch.nn.Module):
    return getattr(module, "_tp_group", None) or getattr(module, "tp_group", None)


def _lora_delta(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = module.slime_lora_dropout(x)  # type: ignore[attr-defined]
    delta = F.linear(x, module.slime_lora_A)  # type: ignore[attr-defined]
    delta = F.linear(delta, module.slime_lora_B)  # type: ignore[attr-defined]
    return delta * module.slime_lora_scaling  # type: ignore[attr-defined]


def _maybe_reduce_row_delta(module: torch.nn.Module, delta: torch.Tensor) -> torch.Tensor:
    if getattr(module, "explicit_expert_comm", False):
        return delta
    if getattr(module, "sequence_parallel", False):
        return reduce_scatter_to_sequence_parallel_region(delta, group=_tp_group(module))
    return reduce_from_tensor_model_parallel_region(delta, group=_tp_group(module))


def _maybe_gather_column_delta(module: torch.nn.Module, delta: torch.Tensor) -> torch.Tensor:
    if getattr(module, "gather_output", False):
        return gather_from_tensor_model_parallel_region(delta, group=_tp_group(module))
    return delta


def _lora_forward_hook(module: torch.nn.Module, inputs, output):
    if getattr(module, "slime_lora_merged", False):
        return output
    if not inputs:
        return output

    x = inputs[0]
    delta = _lora_delta(module, x)
    mode = _parallel_mode(module)
    if mode == "row":
        delta = _maybe_reduce_row_delta(module, delta)
    elif mode == "column":
        delta = _maybe_gather_column_delta(module, delta)

    if isinstance(output, tuple):
        if len(output) != 2:
            raise RuntimeError(f"Unsupported LoRA-wrapped output tuple length: {len(output)}")
        return output[0] + delta, output[1]
    return output + delta


def apply_megatron_lora(model: torch.nn.Module, args) -> int:
    """Attach lightweight LoRA adapters to Megatron linear modules in-place."""
    targets = _split_target_modules(getattr(args, "lora_target_modules", None))
    if not targets:
        raise ValueError("--use-megatron-lora requires --lora-target-modules")

    rank = int(getattr(args, "lora_rank", 8))
    if rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {rank}")
    alpha = float(getattr(args, "lora_alpha", rank))
    dropout = float(getattr(args, "lora_dropout", 0.0))
    patched = 0

    for param in model.parameters():
        param.requires_grad_(False)

    for name, module in model.named_modules():
        if not _target_matches(name, targets) or not _is_supported_linear(module):
            continue
        if ".experts." in name and not getattr(args, "megatron_lora_include_experts", False):
            continue
        if getattr(module, "slime_lora_enabled", False):
            continue

        weight = module.weight
        out_features, in_features = weight.shape
        device = weight.device
        dtype = weight.dtype

        lora_A = torch.nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        lora_B = torch.nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))

        mode = _parallel_mode(module)
        average_across_tp = mode in {None, "duplicated"} and mpu.get_tensor_model_parallel_world_size() > 1
        _set_lora_param_attrs(lora_A, average_across_tp=average_across_tp)
        _set_lora_param_attrs(lora_B, average_across_tp=average_across_tp)

        module.register_parameter("slime_lora_A", lora_A)
        module.register_parameter("slime_lora_B", lora_B)
        module.slime_lora_scaling = alpha / rank
        module.slime_lora_dropout = torch.nn.Dropout(dropout)
        module.slime_lora_enabled = True
        module.slime_lora_merged = False
        module.register_forward_hook(_lora_forward_hook)
        patched += 1

    if patched == 0:
        raise ValueError(f"No Megatron modules matched --lora-target-modules={','.join(targets)}")
    logger.info("Applied Megatron LoRA to %s modules: %s", patched, ",".join(targets))
    return patched


def has_megatron_lora(model: torch.nn.Module | list[torch.nn.Module]) -> bool:
    modules = model if isinstance(model, list) else [model]
    return any(
        getattr(module, "slime_lora_enabled", False)
        for model_chunk in modules
        for module in model_chunk.modules()
    )


def iter_lora_modules(model: torch.nn.Module | list[torch.nn.Module]):
    modules = model if isinstance(model, list) else [model]
    for model_chunk in modules:
        for name, module in model_chunk.named_modules():
            if getattr(module, "slime_lora_enabled", False):
                yield name, module


def _local_delta_weight(module: torch.nn.Module) -> torch.Tensor:
    return (module.slime_lora_B @ module.slime_lora_A) * module.slime_lora_scaling


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _dist_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _mp_rank(name: str) -> int:
    try:
        if name == "tp":
            return mpu.get_tensor_model_parallel_rank()
        if name == "pp":
            return mpu.get_pipeline_model_parallel_rank()
        if name == "cp":
            return mpu.get_context_parallel_rank()
        if name == "ep":
            return mpu.get_expert_model_parallel_rank()
    except Exception:
        return 0
    return 0


def _adapter_shard_name() -> str:
    return "adapter_tp{tp:02d}_pp{pp:02d}_cp{cp:02d}_ep{ep:02d}.pt".format(
        tp=_mp_rank("tp"),
        pp=_mp_rank("pp"),
        cp=_mp_rank("cp"),
        ep=_mp_rank("ep"),
    )


def _adapter_shard_path(model_dir: Path) -> Path:
    return model_dir / _adapter_shard_name()


def _is_adapter_writer_rank() -> bool:
    try:
        return mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    except Exception:
        return _dist_rank() == 0


def _resolve_adapter_path(path: str | None) -> Path | None:
    if not path:
        return None

    adapter_path = Path(path).expanduser()
    if adapter_path.is_file():
        return adapter_path

    if adapter_path.is_dir():
        for tracker_name in (ADAPTER_TRACKER_FILE, STANDARD_TRACKER_FILE):
            tracker_path = adapter_path / tracker_name
            if tracker_path.is_file():
                step = int(tracker_path.read_text().strip())
                adapter_path = adapter_path / f"iter_{step:07d}"
                break

    if adapter_path.is_dir() and (adapter_path / "model").is_dir():
        adapter_path = adapter_path / "model"

    if adapter_path.is_dir():
        shard_path = _adapter_shard_path(adapter_path)
        if shard_path.is_file():
            return shard_path
        legacy_path = adapter_path / "adapter_weights.pt"
        if legacy_path.is_file():
            return legacy_path
        return shard_path

    return adapter_path


@contextmanager
def merged_megatron_lora(model: torch.nn.Module | list[torch.nn.Module]):
    merged_modules = []
    with torch.no_grad():
        for _name, module in iter_lora_modules(model):
            if getattr(module, "slime_lora_merged", False):
                continue
            module.weight.add_(_local_delta_weight(module).to(dtype=module.weight.dtype))
            module.slime_lora_merged = True
            merged_modules.append(module)
    try:
        yield
    finally:
        with torch.no_grad():
            for module in reversed(merged_modules):
                module.weight.sub_(_local_delta_weight(module).to(dtype=module.weight.dtype))
                module.slime_lora_merged = False


def save_megatron_lora_checkpoint(model: list[torch.nn.Module], args, iteration: int) -> None:
    if not getattr(args, "save", None):
        return

    step_id = iteration + 1
    checkpoint_dir = Path(args.save).expanduser() / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    if _is_adapter_writer_rank():
        model_dir.mkdir(parents=True, exist_ok=True)
        adapter_state = {}
        for chunk_id, model_chunk in enumerate(model):
            for module_name, module in iter_lora_modules(model_chunk):
                prefix = f"chunks.{chunk_id}.{module_name}"
                adapter_state[f"{prefix}.slime_lora_A"] = module.slime_lora_A.detach().cpu()
                adapter_state[f"{prefix}.slime_lora_B"] = module.slime_lora_B.detach().cpu()
        torch.save(adapter_state, _adapter_shard_path(model_dir))

    if _dist_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "megatron_lora_sharded",
            "checkpoint_type": "adapter_only_warm_start",
            "training_resume_supported": False,
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "lora_rank": getattr(args, "lora_rank", None),
            "lora_alpha": getattr(args, "lora_alpha", None),
            "lora_dropout": getattr(args, "lora_dropout", None),
            "lora_target_modules": getattr(args, "lora_target_modules", None),
        }
        (checkpoint_dir / "meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        (Path(args.save).expanduser() / ADAPTER_TRACKER_FILE).write_text(str(step_id))
        logger.info("Saved Megatron LoRA adapter checkpoint metadata to %s", model_dir)
    _dist_barrier()


def load_megatron_lora_checkpoint(model: list[torch.nn.Module], path: str | None) -> None:
    adapter_path = _resolve_adapter_path(path)
    if not adapter_path:
        return
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Megatron LoRA adapter checkpoint not found: {adapter_path}")
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)

    missing = []
    with torch.no_grad():
        for chunk_id, model_chunk in enumerate(model):
            for module_name, module in iter_lora_modules(model_chunk):
                prefix = f"chunks.{chunk_id}.{module_name}"
                for suffix, param in [("slime_lora_A", module.slime_lora_A), ("slime_lora_B", module.slime_lora_B)]:
                    key = f"{prefix}.{suffix}"
                    if key not in state:
                        missing.append(key)
                        continue
                    param.copy_(state[key].to(device=param.device, dtype=param.dtype))
    if missing:
        raise RuntimeError(f"Megatron LoRA adapter checkpoint missing keys: {missing[:8]}")
    logger.info("Loaded Megatron LoRA adapter checkpoint from %s", adapter_path)
