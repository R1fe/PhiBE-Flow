from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metric: float | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metric": metric,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location)
    current = model.state_dict()
    for key in ("input_layer.0.weight", "input_projection.0.weight"):
        old_weight = checkpoint["model_state_dict"].get(key)
        new_weight = current.get(key)
        if old_weight is not None and new_weight is not None and old_weight.shape != new_weight.shape:
            raise ValueError(
                f"Checkpoint input shape {tuple(old_weight.shape)} does not match "
                f"{tuple(new_weight.shape)} for {key}. Time-conditioned v(x,t) models "
                "require new training checkpoints; legacy v(x) weights cannot be resumed directly."
            )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
