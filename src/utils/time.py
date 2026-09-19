from __future__ import annotations

import torch


def batch_time(time, reference: torch.Tensor) -> torch.Tensor:
    """Return one explicit, finite time per sample on the state's device."""
    value = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    elif value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.shape != (reference.shape[0],):
        raise ValueError(f"Expected scalar time or [B]/[B, 1], got {tuple(value.shape)}.")
    if not torch.isfinite(value).all():
        raise ValueError("Time must be finite.")
    return value
