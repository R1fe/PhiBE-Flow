from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.utils.time import batch_time


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    """Residual MLP block used by the Acrobot velocity predictor.

    The default settings preserve the original two-linear-layer block.  The
    deeper experiment enables pre-normalization, a wider feed-forward inner
    layer, dropout, and a small learnable residual scale.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: nn.Module,
        *,
        ffn_expansion: int = 1,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if ffn_expansion < 1:
            raise ValueError("ffn_expansion must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.use_layer_norm = use_layer_norm
        self.norm = nn.LayerNorm(in_dim) if use_layer_norm else nn.Identity()
        inner_dim = out_dim * ffn_expansion
        self.linear1 = nn.Linear(in_dim, inner_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_dim, out_dim)
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.layer_scale = (
            nn.Parameter(torch.full((out_dim,), layer_scale_init))
            if layer_scale_init > 0.0
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        hidden = self.norm(inputs)
        hidden = self.linear1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout(hidden)
        if self.layer_scale is not None:
            hidden = hidden * self.layer_scale
        return self.activation(hidden + residual)


@dataclass
class PredictorConfig:
    seq_length: int
    feature_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [128, 256])
    num_res_blocks: int = 2
    res_block_dim: int = 256
    activation: str = "relu"
    ffn_expansion: int = 1
    dropout: float = 0.0
    use_layer_norm: bool = False
    layer_scale_init: float = 0.0
    periodic_angle_indices: list[int] = field(default_factory=list)
    predict_acceleration_only: bool = False
    physics_backbone: bool = False
    physics_residual_scale: float = 1.0
    kinematic_time_delta: float = 0.0
    zero_init_output: bool = False


class VelocityPredictor(nn.Module):
    """Flattened sequence-to-velocity predictor adapted from the source ResNet MLP."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        activation = get_activation(config.activation)
        self.seq_length = config.seq_length
        self.feature_dim = config.feature_dim
        self.periodic_angle_indices = tuple(config.periodic_angle_indices)
        self.predict_acceleration_only = config.predict_acceleration_only
        self.physics_backbone = config.physics_backbone
        self.physics_residual_scale = config.physics_residual_scale
        self.kinematic_time_delta = config.kinematic_time_delta

        for angle_index in self.periodic_angle_indices:
            if not 0 <= angle_index < config.feature_dim:
                raise ValueError(
                    f"Periodic angle index {angle_index} is outside feature_dim={config.feature_dim}."
                )
        if self.predict_acceleration_only and config.feature_dim != 4:
            raise ValueError(
                "predict_acceleration_only currently requires the four-dimensional "
                "Acrobot state [theta1, theta2, theta1_dot, theta2_dot]."
            )
        if self.physics_backbone and not self.predict_acceleration_only:
            raise ValueError("physics_backbone requires predict_acceleration_only=true.")

        encoded_feature_dim = config.feature_dim + len(self.periodic_angle_indices)
        input_dim = config.seq_length * encoded_feature_dim + 1
        learned_output_dim = 2 if self.predict_acceleration_only else config.output_dim

        input_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            input_layers.append(nn.Linear(prev_dim, hidden_dim))
            input_layers.append(get_activation(config.activation))
            prev_dim = hidden_dim

        input_layers.append(nn.Linear(prev_dim, config.res_block_dim))
        input_layers.append(get_activation(config.activation))
        self.input_layer = nn.Sequential(*input_layers)

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    config.res_block_dim,
                    config.res_block_dim,
                    activation=get_activation(config.activation),
                    ffn_expansion=config.ffn_expansion,
                    dropout=config.dropout,
                    use_layer_norm=config.use_layer_norm,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.num_res_blocks)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(config.res_block_dim, config.res_block_dim // 2),
            get_activation(config.activation),
            nn.Linear(config.res_block_dim // 2, config.res_block_dim // 4),
            get_activation(config.activation),
            nn.Linear(config.res_block_dim // 4, learned_output_dim),
        )
        if config.zero_init_output:
            output_linear = self.output_layer[-1]
            if not isinstance(output_linear, nn.Linear):
                raise TypeError("The output head must end with nn.Linear.")
            nn.init.zeros_(output_linear.weight)
            nn.init.zeros_(output_linear.bias)

    def forward(self, inputs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            state_sequence = inputs.reshape(-1, self.seq_length, self.feature_dim)
        elif inputs.dim() == 3:
            state_sequence = inputs
        else:
            raise ValueError(
                "VelocityPredictor expects [batch, feature] or [batch, time, feature], "
                f"got {tuple(inputs.shape)}."
            )

        if self.periodic_angle_indices:
            encoded_features = []
            periodic_indices = set(self.periodic_angle_indices)
            for feature_index in range(self.feature_dim):
                feature = state_sequence[..., feature_index : feature_index + 1]
                if feature_index in periodic_indices:
                    encoded_features.extend((feature.sin(), feature.cos()))
                else:
                    encoded_features.append(feature)
            model_inputs = torch.cat(encoded_features, dim=-1).flatten(start_dim=1)
        else:
            model_inputs = state_sequence.flatten(start_dim=1)

        model_inputs = torch.cat((model_inputs, batch_time(time, inputs)[:, None]), dim=1)
        hidden = self.input_layer(model_inputs)
        hidden = self.res_blocks(hidden)
        learned_output = self.output_layer(hidden)
        if self.predict_acceleration_only:
            last_state = state_sequence[:, -1, :]
            acceleration = learned_output
            if self.physics_backbone:
                acceleration = self._acrobot_acceleration(last_state) + (
                    self.physics_residual_scale * learned_output
                )
            angular_drift = last_state[:, 2:4]
            if self.kinematic_time_delta > 0.0:
                angular_drift = angular_drift + (
                    0.5 * acceleration * self.kinematic_time_delta
                )
            return torch.cat((angular_drift, acceleration), dim=1)
        return learned_output

    @staticmethod
    def _acrobot_acceleration(state: torch.Tensor) -> torch.Tensor:
        """Noise-free Acrobot acceleration used as a differentiable backbone."""
        theta1, theta2, theta1_dot, theta2_dot = state.unbind(dim=1)
        sin_theta2 = theta2.sin()
        cos_theta2 = theta2.cos()

        d1 = 1.0 + 1.0 + 0.5**2 + (1.0 + 0.5**2 + cos_theta2)
        d2 = 1.0 + (0.5**2 + 0.5 * cos_theta2)
        c1 = (
            -0.5 * sin_theta2 * theta2_dot.square()
            - sin_theta2 * theta1_dot * theta2_dot
            + 1.5 * 9.8 * (theta1 - torch.pi / 2).cos()
            + 0.5 * 9.8 * (theta1 + theta2 - torch.pi / 2).cos()
        )
        c2 = 0.5 * 9.8 * (theta1 + theta2 - torch.pi / 2).cos()
        theta2_ddot = (
            (d2 / d1) * c1 - 0.5 * theta1_dot.square() * sin_theta2 - c2
        ) / (0.5**2 + 1.0 - d2.square() / d1)
        theta1_ddot = -(d2 * theta2_ddot + c1) / d1
        return torch.stack((theta1_ddot, theta2_ddot), dim=1)


def build_predictor(
    *,
    seq_length: int,
    feature_dim: int,
    hidden_dims: list[int],
    num_res_blocks: int,
    res_block_dim: int,
    activation: str = "relu",
    ffn_expansion: int = 1,
    dropout: float = 0.0,
    use_layer_norm: bool = False,
    layer_scale_init: float = 0.0,
    periodic_angle_indices: list[int] | None = None,
    predict_acceleration_only: bool = False,
    physics_backbone: bool = False,
    physics_residual_scale: float = 1.0,
    kinematic_time_delta: float = 0.0,
    zero_init_output: bool = False,
) -> VelocityPredictor:
    config = PredictorConfig(
        seq_length=seq_length,
        feature_dim=feature_dim,
        output_dim=feature_dim,
        hidden_dims=hidden_dims,
        num_res_blocks=num_res_blocks,
        res_block_dim=res_block_dim,
        activation=activation,
        ffn_expansion=ffn_expansion,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        layer_scale_init=layer_scale_init,
        periodic_angle_indices=list(periodic_angle_indices or []),
        predict_acceleration_only=predict_acceleration_only,
        physics_backbone=physics_backbone,
        physics_residual_scale=physics_residual_scale,
        kinematic_time_delta=kinematic_time_delta,
        zero_init_output=zero_init_output,
    )
    return VelocityPredictor(config)
