"""Frozen KTH f8_small VQ-VAE with the source checkpoint layout."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .vqvae_taming import VQModelInterface, vq_f8_small_ddconfig


class VQVAE(nn.Module):
    def __init__(self, checkpoint_path: str | Path, chunk_size: int = 32):
        super().__init__()
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"KTH VQ-VAE weights not found: {checkpoint_path}")
        if chunk_size < 1:
            raise ValueError("VQ-VAE chunk_size must be positive.")
        self.chunk_size = chunk_size
        self.backbone = VQModelInterface(vq_f8_small_ddconfig, str(checkpoint_path))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # The pretrained image codec must remain frozen during velocity training.
        return super().train(False)

    @torch.no_grad()
    def _sequence(self, value, function):
        if value.ndim not in (4, 5):
            raise ValueError("Expected [B,C,H,W] or [B,T,C,H,W].")
        shape = value.shape
        flat = value.reshape(-1, *shape[-3:])
        output = torch.cat([function(chunk) for chunk in flat.split(self.chunk_size)])
        return output.reshape(*shape[:-3], *output.shape[-3:])

    def encode(self, frames):
        # Continuous pre-quantization latents, as used by the source predictor.
        return self._sequence(frames, self.backbone.encode)

    def decode(self, latents):
        # Source decoding includes nearest-codebook quantization.
        return self._sequence(latents, self.backbone.decode)
