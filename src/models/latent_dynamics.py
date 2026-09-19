from __future__ import annotations

import torch
from torch import nn

from src.utils.time import batch_time


class LatentResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        expansion: int = 2,
        dropout: float = 0.0,
        layer_scale_init: float = 0.01,
    ):
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )
        self.layer_scale = nn.Parameter(torch.full((width,), layer_scale_init))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layer_scale * self.ffn(self.norm(inputs))


class LatentResidualDynamics(nn.Module):
    """Time-conditioned discrete dynamics for [z1,z2,z3,dz1,dz2,dz3]."""

    def __init__(
        self,
        sequence_length: int,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        acceleration_scale: torch.Tensor,
        width: int = 512,
        num_blocks: int = 8,
        expansion: int = 2,
        dropout: float = 0.0,
        layer_scale_init: float = 0.01,
        acceleration_limit: float = 5.0,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.acceleration_limit = acceleration_limit
        self.register_buffer("state_mean", state_mean.float().view(1, 1, 6))
        self.register_buffer("state_std", state_std.float().view(1, 1, 6))
        self.register_buffer("acceleration_scale", acceleration_scale.float().view(1, 3))
        self.input_projection = nn.Sequential(
            nn.Linear(sequence_length * 6 + 1, width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                LatentResidualBlock(
                    width,
                    expansion=expansion,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 3),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, sequence: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        normalized = (sequence - self.state_mean) / self.state_std
        inputs = torch.cat((normalized.flatten(start_dim=1), batch_time(time, sequence)[:, None]), dim=1)
        hidden = self.input_projection(inputs)
        hidden = self.blocks(hidden)
        raw_acceleration = self.output_head(hidden)
        acceleration = (
            torch.tanh(raw_acceleration)
            * self.acceleration_scale
            * self.acceleration_limit
        )
        current = sequence[:, -1, :]
        current_position = current[:, :3]
        current_velocity = current[:, 3:]
        next_position = current_position + current_velocity
        next_velocity = current_velocity + acceleration
        return torch.cat((next_position, next_velocity), dim=1)


class LatentAnalogResidualRollout(nn.Module):
    """Correct a nearest-training-trajectory rollout with a deep residual MLP."""

    def __init__(
        self,
        sequence_length: int,
        rollout_steps: int,
        state_std: torch.Tensor,
        width: int = 384,
        num_blocks: int = 8,
        expansion: int = 2,
        dropout: float = 0.0,
        layer_scale_init: float = 0.01,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.rollout_steps = rollout_steps
        self.register_buffer("state_std", state_std.float().view(1, 1, 6))
        self.input_projection = nn.Sequential(
            nn.Linear(sequence_length * 6, width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                LatentResidualBlock(
                    width,
                    expansion=expansion,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, rollout_steps * 3),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        context: torch.Tensor,
        analog_context: torch.Tensor,
        analog_future_positions: torch.Tensor,
    ) -> torch.Tensor:
        context_difference = (context - analog_context) / self.state_std
        hidden = self.input_projection(context_difference.flatten(start_dim=1))
        hidden = self.blocks(hidden)
        normalized_correction = self.output_head(hidden).view(-1, self.rollout_steps, 3)
        correction = normalized_correction * self.state_std[:, :, :3]
        return analog_future_positions + correction
