from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _load_cache_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")
    return payload


def _load_tensor_dataset(payload: dict, path: Path) -> TensorDataset:
    required = ["state_hidden", "action_hidden", "target_hidden"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}. Expected {required}.")
    tensors = [payload[key].float() for key in required]
    count = int(tensors[0].shape[0])
    for key, tensor in zip(required, tensors):
        if int(tensor.shape[0]) != count:
            raise ValueError(f"Inconsistent first dimension for {key}: expected {count}, got {int(tensor.shape[0])}")
    if "reward" in payload:
        reward = payload["reward"].float()
        if int(reward.shape[0]) != count:
            raise ValueError(f"Inconsistent first dimension for reward: expected {count}, got {int(reward.shape[0])}")
        tensors.append(reward)
        if "reward_mask" in payload:
            reward_mask = payload["reward_mask"].bool()
            if int(reward_mask.shape[0]) != count:
                raise ValueError(
                    f"Inconsistent first dimension for reward_mask: expected {count}, got {int(reward_mask.shape[0])}"
                )
            tensors.append(reward_mask)
    return TensorDataset(*tensors)


def _split_dataset(dataset: TensorDataset, *, val_ratio: float, seed: int):
    if val_ratio <= 0.0 or len(dataset) < 2:
        return dataset, None
    val_size = int(round(len(dataset) * val_ratio))
    val_size = max(1, min(val_size, len(dataset) - 1))
    train_size = len(dataset) - val_size
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _run_epoch(
    *,
    model: TextLatentWorldModel,
    loader: DataLoader,
    device: torch.device,
    optim: torch.optim.Optimizer | None,
    sigreg_coef: float,
    action_contrast_coef: float,
    value_coef: float,
) -> float:
    total_loss = 0.0
    total_count = 0
    train = optim is not None
    model.train(train)
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            batch = [x.to(device) for x in batch]
            reward = batch[3] if len(batch) > 3 else None
            reward_mask = batch[4] if len(batch) > 4 else None
            loss, _metrics = model.compute_loss(
                state_hidden=batch[0],
                action_hidden=batch[1],
                target_hidden=batch[2],
                reward=reward,
                reward_mask=reward_mask,
                sigreg_coef=sigreg_coef,
                action_contrast_coef=action_contrast_coef,
                value_coef=value_coef,
            )
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            total_loss += float(loss.detach().cpu()) * batch[0].size(0)
            total_count += batch[0].size(0)
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small JEPA-style text latent world-model probe.")
    parser.add_argument("--input", required=True, help="Torch file with state_hidden/action_hidden/target_hidden tensors.")
    parser.add_argument("--output", required=True, help="Checkpoint path for the trained probe.")
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigreg-coef", type=float, default=0.1)
    parser.add_argument("--action-contrast-coef", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = _load_cache_payload(input_path)
    dataset = _load_tensor_dataset(payload, input_path)
    if len(dataset) == 0:
        raise ValueError(f"No cached world-model records found in {input_path}.")
    first = dataset[0]
    config = TextLatentWorldModelConfig(
        state_hidden_dim=int(first[0].shape[-1]),
        action_hidden_dim=int(first[1].shape[-1]),
        target_hidden_dim=int(first[2].shape[-1]),
        latent_dim=args.latent_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextLatentWorldModel(config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dataset, val_dataset = _split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(
        "dataset "
        f"total={len(dataset)} train={len(train_dataset)} val={0 if val_dataset is None else len(val_dataset)} "
        f"device={device}"
    )

    final_train_loss = None
    final_val_loss = None
    for epoch in range(args.epochs):
        train_loss = _run_epoch(
            model=model,
            loader=loader,
            device=device,
            optim=optim,
            sigreg_coef=args.sigreg_coef,
            action_contrast_coef=args.action_contrast_coef,
            value_coef=args.value_coef,
        )
        if val_loader is not None:
            val_loss = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optim=None,
                sigreg_coef=args.sigreg_coef,
                action_contrast_coef=args.action_contrast_coef,
                value_coef=args.value_coef,
            )
            print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={val_loss:.6f}")
            final_val_loss = val_loss
        else:
            print(f"epoch={epoch + 1} loss={train_loss:.6f}")
        final_train_loss = train_loss

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "openclaw_text_jepa_probe_checkpoint_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "cache_metadata": payload.get("metadata", {}),
        "record_count": len(dataset),
        "has_reward": "reward" in payload,
        "reward_mask_count": int(payload.get("reward_mask", torch.zeros(0, dtype=torch.bool)).sum().item())
        if "reward_mask" in payload
        else None,
        "state_hidden_shape": tuple(payload["state_hidden"].shape),
        "action_hidden_shape": tuple(payload["action_hidden"].shape),
        "target_hidden_shape": tuple(payload["target_hidden"].shape),
        "train_count": len(train_dataset),
        "val_count": 0 if val_dataset is None else len(val_dataset),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "hyperparameters": {
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "sigreg_coef": args.sigreg_coef,
            "action_contrast_coef": args.action_contrast_coef,
            "value_coef": args.value_coef,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
    }
    torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "metadata": metadata}, out)
    print(f"saved probe checkpoint to {out}")


if __name__ == "__main__":
    main()
