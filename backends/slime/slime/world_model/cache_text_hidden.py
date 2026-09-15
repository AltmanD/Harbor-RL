from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_state_text(record: dict) -> str:
    text = record.get("context_text")
    if text:
        return str(text)
    # Compatibility fallback for records created before context_text existed.
    return json.dumps(
        {
            "context_hash": record.get("context_hash"),
            "data_source": record.get("data_source"),
            "task_name": record.get("task_name"),
            "task_path": record.get("task_path"),
            "turn_idx": record.get("turn_idx"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _record_action_text(record: dict) -> str:
    return str(record.get("action_text") or record.get("action_hash") or "")


def _record_target_text(record: dict) -> str:
    return str(record.get("next_observation_text") or record.get("next_observation_hash") or "")


def _light_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "schema",
        "hidden_source",
        "task_name",
        "task_path",
        "data_source",
        "uid",
        "group_index",
        "sample_index",
        "rollout_id",
        "train_step",
        "turn_idx",
        "num_turns",
        "status",
        "done",
        "has_tool_result",
        "reward_score",
        "reward_base_score",
        "reward_raw_score",
        "context_hash",
        "action_hash",
        "next_observation_hash",
        "context_token_len",
        "action_token_len",
        "context_text_source",
    ]
    return {key: record.get(key) for key in keys if key in record}


def _stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _hash_encode(texts: Iterable[str], hidden_dim: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for text in texts:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(text))
        vec = torch.randn(hidden_dim, generator=generator)
        rows.append(F.normalize(vec, dim=0))
    if not rows:
        return torch.empty(0, hidden_dim)
    return torch.stack(rows, dim=0)


def _hf_encode(
    texts: list[str],
    *,
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str,
    local_files_only: bool,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].to(hidden.dtype)
        if pooling == "last":
            lengths = enc["attention_mask"].sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        elif pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        rows.append(pooled.detach().cpu().float())
    if not rows:
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        return torch.empty(0, hidden_size)
    return torch.cat(rows, dim=0)


def _encode_texts(args: argparse.Namespace, texts: list[str]) -> torch.Tensor:
    if args.encoder == "hash":
        return _hash_encode(texts, args.hidden_dim)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return _hf_encode(
        texts,
        model_name_or_path=args.hf_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        pooling=args.pooling,
        local_files_only=args.hf_local_files_only,
    )


def _encode_record_texts(
    args: argparse.Namespace,
    state_texts: list[str],
    action_texts: list[str],
    target_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.encoder == "hash":
        return (
            _hash_encode(state_texts, args.hidden_dim),
            _hash_encode(action_texts, args.hidden_dim),
            _hash_encode(target_texts, args.hidden_dim),
        )

    all_texts = state_texts + action_texts + target_texts
    all_hidden = _encode_texts(args, all_texts)
    count = len(state_texts)
    return all_hidden[:count], all_hidden[count : count * 2], all_hidden[count * 2 :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache state/action/target hidden tensors for text JEPA world-model probes.")
    parser.add_argument("--input", required=True, help="World-model records JSONL.")
    parser.add_argument("--output", required=True, help="Torch output with state_hidden/action_hidden/target_hidden tensors.")
    parser.add_argument("--encoder", choices=["hash", "hf"], default="hash")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hash encoder output dim.")
    parser.add_argument("--hf-model", default=None, help="HF model path/name used when --encoder hf.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pooling", choices=["mean", "last", "cls"], default="mean")
    parser.add_argument("--hf-local-files-only", action="store_true", help="Do not download HF model/tokenizer files.")
    args = parser.parse_args()

    if args.encoder == "hf" and not args.hf_model:
        raise ValueError("--hf-model is required when --encoder hf")

    input_path = Path(args.input)
    records = _read_jsonl(input_path)
    if not records:
        raise ValueError(f"No world-model records found in {input_path}. Check filters or input path.")
    state_texts = [_record_state_text(record) for record in records]
    action_texts = [_record_action_text(record) for record in records]
    target_texts = [_record_target_text(record) for record in records]

    state_hidden, action_hidden, target_hidden = _encode_record_texts(args, state_texts, action_texts, target_texts)
    rewards = [record.get("reward_score") for record in records]
    reward_mask = [value is not None for value in rewards]
    has_reward = any(value is not None for value in rewards)

    payload = {
        "state_hidden": state_hidden,
        "action_hidden": action_hidden,
        "target_hidden": target_hidden,
        "record_count": len(records),
        "encoder": args.encoder,
        "input": str(input_path),
        "record_metadata": [_light_record_metadata(record) for record in records],
        "metadata": {
            "schema_version": "openclaw_text_jepa_hidden_cache_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": str(input_path),
            "input_records_sha256": _file_sha256(input_path),
            "record_count": len(records),
            "encoder": args.encoder,
            "hidden_dim": args.hidden_dim if args.encoder == "hash" else int(state_hidden.shape[-1]),
            "hf_model": args.hf_model if args.encoder == "hf" else None,
            "pooling": args.pooling if args.encoder == "hf" else None,
            "hf_local_files_only": bool(args.hf_local_files_only) if args.encoder == "hf" else None,
            "max_length": args.max_length if args.encoder == "hf" else None,
            "batch_size": args.batch_size,
            "state_hidden_shape": tuple(state_hidden.shape),
            "action_hidden_shape": tuple(action_hidden.shape),
            "target_hidden_shape": tuple(target_hidden.shape),
        },
    }
    if has_reward:
        payload["reward"] = torch.tensor([0.0 if value is None else float(value) for value in rewards], dtype=torch.float32)
        payload["reward_mask"] = torch.tensor(reward_mask, dtype=torch.bool)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(
        "cached "
        f"{len(records)} records to {out} "
        f"(state={tuple(state_hidden.shape)} action={tuple(action_hidden.shape)} target={tuple(target_hidden.shape)} encoder={args.encoder})"
    )


if __name__ == "__main__":
    main()
