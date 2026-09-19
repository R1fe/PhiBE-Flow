from __future__ import annotations

# U-Net components adapted from lucidrains/denoising-diffusion-pytorch (MIT).
# See licenses/DENOISING-DIFFUSION-MIT.txt and THIRD_PARTY_NOTICES.md.

import math
from functools import partial

import torch
from einops import rearrange
from torch import einsum, nn
import torch.nn.functional as F

from src.utils.time import batch_time


def _exists(value) -> bool:
    return value is not None


def _default(value, fallback):
    return value if _exists(value) else fallback


class Residual(nn.Module):
    def __init__(self, function: nn.Module) -> None:
        super().__init__()
        self.fn = function

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int, dim_out: int | None = None) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, _default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim: int, dim_out: int | None = None) -> nn.Module:
    return nn.Conv2d(dim, _default(dim_out, dim), 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * math.sqrt(x.shape[1])


class PreNorm(nn.Module):
    def __init__(self, dim: int, function: nn.Module) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        embedding = x[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, is_random: bool = False) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("The learned sinusoidal dimension must be even.")
        self.weights = nn.Parameter(torch.randn(dim // 2), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        frequencies = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        return torch.cat((x, frequencies.sin(), frequencies.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: int,
        classes_emb_dim: int | None = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        conditioning_dim = time_emb_dim + (classes_emb_dim or 0)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(conditioning_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        class_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioning = [time_embedding]
        if class_embedding is not None:
            conditioning.append(class_embedding)
        scale_shift = rearrange(self.mlp(torch.cat(conditioning, dim=-1)), "b c -> b c 1 1").chunk(2, dim=1)
        hidden = self.block1(x, scale_shift=scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q, k, v = map(
            lambda item: rearrange(item, "b (h c) x y -> b h c (x y)", h=self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = einsum("b h d n, b h e n -> b h d e", k, v)
        output = einsum("b h d e, b h d n -> b h e n", context, q)
        output = rearrange(output, "b h c (x y) -> b (h c) x y", x=height, y=width)
        return self.to_out(output)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q, k, v = map(
            lambda item: rearrange(item, "b (h c) x y -> b h c (x y)", h=self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )
        similarity = einsum("b h d i, b h d j -> b h i j", q * self.scale, k)
        output = einsum("b h i j, b h d j -> b h i d", similarity.softmax(dim=-1), v)
        output = rearrange(output, "b h (x y) d -> b (h d) x y", x=height, y=width)
        return self.to_out(output)


class NSEUnet(nn.Module):
    """Conditioned U-Net copied from forecasting_new and cleaned for this project."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        dim: int = 128,
        dim_mults: tuple[int, ...] = (1, 2, 2, 2),
        resnet_block_groups: int = 8,
        learned_sinusoidal_cond: bool = True,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 32,
        attention_dim_head: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)
        dims = [dim, *[dim * multiplier for multiplier in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        block = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        if learned_sinusoidal_cond or random_fourier_features:
            position_embedding = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            embedding_dim = learned_sinusoidal_dim + 1
        else:
            position_embedding = SinusoidalPosEmb(dim)
            embedding_dim = dim
        self.time_mlp = nn.Sequential(
            position_embedding,
            nn.Linear(embedding_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_in, time_emb_dim=time_dim),
                        block(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        if is_last
                        else Downsample(dim_in, dim_out),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim, dim_head=attention_dim_head, heads=attention_heads))
        )
        self.mid_block2 = block(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = index == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        if is_last
                        else Upsample(dim_out, dim_in),
                    ]
                )
            )

        self.final_res_block = block(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        residual = x.clone()
        time_embedding = self.time_mlp(time)
        skips = []

        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time_embedding)
            skips.append(x)
            x = attention(block2(x, time_embedding))
            skips.append(x)
            x = downsample(x)

        x = self.mid_block2(self.mid_attn(self.mid_block1(x, time_embedding)), time_embedding)
        for block1, block2, attention, upsample in self.ups:
            x = block1(torch.cat((x, skips.pop()), dim=1), time_embedding)
            x = block2(torch.cat((x, skips.pop()), dim=1), time_embedding)
            x = upsample(attention(x))

        x = self.final_res_block(torch.cat((x, residual), dim=1), time_embedding)
        return self.final_conv(x)


class NSEDriftModel(nn.Module):
    """Predict the NSE drift at x1 while conditioning on the previous frame x0."""

    def __init__(self, **unet_kwargs) -> None:
        super().__init__()
        self._arch = NSEUnet(**unet_kwargs)

    def forward(
        self,
        x1: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self._arch(torch.cat((x1, condition), dim=1), batch_time(time, x1))
