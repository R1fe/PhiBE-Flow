"""Time-conditioned adaptation of the RIVER spatial predictor architecture.

Original spatial architecture credited there to RIVER (Araachie/river).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

from src.utils.time import batch_time


class SpatialPositionEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.row_embed = nn.Embedding(512, width // 2)
        self.col_embed = nn.Embedding(512, width // 2)
        nn.init.uniform_(self.row_embed.weight, -1, 1)
        nn.init.uniform_(self.col_embed.weight, -1, 1)

    def forward(self, z):
        b, _, h, w = z.shape
        rows = self.row_embed(torch.arange(h, device=z.device))
        cols = self.col_embed(torch.arange(w, device=z.device))
        return torch.cat([cols[None].expand(h, -1, -1),
                          rows[:, None].expand(-1, w, -1)], -1).reshape(1, h*w, -1).expand(b, -1, -1)


class SpatialTransformerLayer(nn.TransformerEncoderLayer):
    """Unfused pre-norm attention supports double backward on old/new PyTorch.

    Standard parameter names preserve the original spatial weight layouts.
    """

    def forward(self, src):
        x = self.norm1(src)
        b, n, width = x.shape
        heads = self.self_attn.num_heads
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.reshape(b, n, 3, heads, width // heads).permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1) / math.sqrt(width // heads)).softmax(-1)
        attention = F.dropout(attention, self.self_attn.dropout, self.training)
        x = (attention @ v).transpose(1, 2).reshape(b, n, width)
        x = src + self.dropout1(self.self_attn.out_proj(x))
        y = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
        return x + self.dropout2(y)


class KTHVelocityPredictor(nn.Module):
    def __init__(self, state_size=4, state_res=(8, 8), inner_dim=768,
                 depth=4, mid_depth=5, heads=8, dropout=0.05, time_scale=1.0):
        super().__init__()
        if inner_dim < 2 or inner_dim % 2 or inner_dim % heads:
            raise ValueError("inner_dim must be even and divisible by heads.")
        self.state_shape = (state_size, *state_res)
        self.inner_dim = inner_dim
        self.time_scale = float(time_scale)
        if not math.isfinite(self.time_scale) or self.time_scale <= 0:
            raise ValueError("time_scale must be finite and positive.")
        self.position_encoding = SpatialPositionEmbedding(inner_dim)
        self.project_in = nn.Sequential(Rearrange("b c h w -> b (h w) c"),
                                        nn.Linear(2 * state_size, inner_dim))
        self.time_embedding = nn.Sequential(nn.Linear(inner_dim, inner_dim), nn.SiLU(),
                                             nn.Linear(inner_dim, inner_dim))

        def layer():
            return SpatialTransformerLayer(inner_dim, heads, 4 * inner_dim,
                                           dropout, activation="gelu", norm_first=True,
                                           batch_first=True)

        self.in_blocks = nn.ModuleList([layer() for _ in range(depth)])
        self.mid_blocks = nn.Sequential(*[layer() for _ in range(mid_depth)])
        self.out_blocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(2 * inner_dim, inner_dim), layer()]) for _ in range(depth)])
        self.project_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim), nn.GELU(), nn.LayerNorm(inner_dim),
            Rearrange("b (h w) c -> b c h w", h=state_res[0]),
            nn.Conv2d(inner_dim, state_size, 3, padding=1))

    def forward(self, z, time, ref):
        if tuple(z.shape[1:]) != self.state_shape or ref.shape != z.shape:
            raise ValueError(f"Expected z/ref [B,{self.state_shape}], got {z.shape}/{ref.shape}.")
        time = batch_time(time, z)
        half = self.inner_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=z.device) / half)
        phase = time[:, None] * self.time_scale * freqs[None]
        embedding = torch.cat([phase.cos(), phase.sin()], dim=-1).to(z.dtype)
        x = self.project_in(torch.cat([z, ref], 1)) + self.position_encoding(z)
        x = x + self.time_embedding(embedding)[:, None]
        skips = []
        for block in self.in_blocks:
            x = block(x)
            skips.append(x)
        x = self.mid_blocks(x)
        for skip, (projection, block) in zip(reversed(skips), self.out_blocks):
            x = block(projection(torch.cat([skip, x], -1)))
        return self.project_out(x)
