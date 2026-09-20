from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_last_state(
    sequence: torch.Tensor,
    dataset_name: str | None = None,
) -> torch.Tensor:
    """
    Extract the conditioning state used to predict the next step.

    `acrobot_angles` uses a 3D tensor `(batch, time, feature)`. Other datasets
    may use 4D tensors, so we keep an explicit branch for that case.
    """
    if dataset_name == "acrobot_angles":
        if sequence.dim() != 3:
            raise ValueError(
                f"Expected a 3D tensor for acrobot_angles, got shape {tuple(sequence.shape)}."
            )
        return sequence[:, -1, :]

    if sequence.dim() == 3:
        return sequence[:, -1, :]
    if sequence.dim() == 4:
        return sequence[:, -1, ...]

    raise ValueError(
        f"Unsupported input rank {sequence.dim()} for sequence shape {tuple(sequence.shape)}."
    )


def flatten_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten every sample while keeping the batch dimension."""
    return tensor.reshape(tensor.shape[0], -1)


def compute_true_drift(
    sequence: torch.Tensor,
    target: torch.Tensor,
    time_delta: float,
    dataset_name: str | None = None,
) -> torch.Tensor:
    """Compute the empirical drift between the last input state and the target."""
    last_state = extract_last_state(sequence, dataset_name=dataset_name)
    return (target - last_state) / time_delta


def compute_full_jacobian(
    predicted_velocity: torch.Tensor,
    sequence: torch.Tensor,
    dataset_name: str | None = None,
) -> torch.Tensor:
    """
    Compute the full Jacobian d v_i / d x_j for the last state of each sample.

    Returns a tensor of shape `(batch, output_dim, input_dim)` after flattening
    the last-state coordinates and the model outputs per sample.
    """
    predicted_flat = flatten_per_sample(predicted_velocity)
    output_dim = predicted_flat.shape[1]
    jacobian_rows = []

    for output_index in range(output_dim):
        grad_component = torch.autograd.grad(
            outputs=predicted_flat[:, output_index].sum(),
            inputs=sequence,
            create_graph=True,
            retain_graph=True,
        )[0]
        last_grad = extract_last_state(grad_component, dataset_name=dataset_name)
        jacobian_rows.append(flatten_per_sample(last_grad).unsqueeze(1))

    return torch.cat(jacobian_rows, dim=1)


def velocity_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    time_delta: float = 1.0 / 30.0,
    dataset_name: str | None = None,
    include_diffusion: bool = True,
) -> torch.Tensor:
    """
    SDE-inspired objective from the source Acrobot training code.

    The model predicts a velocity field v(x_t, t). The loss combines the norm of the
    field and its alignment with the empirical drift. Diffusion is enabled by
    default; disabling it is an explicit loss-level ablation. Its Jacobian is
    a spatial derivative at fixed t; time is not a state coordinate.
    """
    sequence, target, time = batch
    sequence = sequence.to(device)
    if include_diffusion:
        sequence = sequence.requires_grad_(True)
    target = target.to(device)

    with torch.backends.cudnn.flags(enabled=False):
        predicted_velocity = model(sequence, time)

    last_state = extract_last_state(sequence, dataset_name=dataset_name)
    drift = compute_true_drift(
        sequence,
        target,
        time_delta,
        dataset_name=dataset_name,
    )
    error = target - last_state

    predicted_flat = flatten_per_sample(predicted_velocity)
    drift_flat = flatten_per_sample(drift)
    error_flat = flatten_per_sample(error)

    velocity_sq = (predicted_flat ** 2).sum(dim=1)
    alignment = 2.0 * (drift_flat * predicted_flat).sum(dim=1)

    if include_diffusion:
        jacobian = compute_full_jacobian(
            predicted_velocity,
            sequence,
            dataset_name=dataset_name,
        )
        sigma_matrix = (error_flat.unsqueeze(2) * error_flat.unsqueeze(1)) / time_delta
        diffusion = (sigma_matrix * jacobian).sum(dim=(1, 2))
        return (velocity_sq - alignment - diffusion).mean()

    return (velocity_sq - alignment).mean()


def validation_mse_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    time_delta: float = 1.0 / 30.0,
    dataset_name: str | None = None,
) -> torch.Tensor:
    """Simple validation metric: predicted velocity against empirical drift."""
    sequence, target, time = batch
    sequence = sequence.to(device)
    target = target.to(device)

    predicted_velocity = model(sequence, time)
    true_drift = compute_true_drift(
        sequence,
        target,
        time_delta,
        dataset_name=dataset_name,
    )
    return F.mse_loss(predicted_velocity, true_drift)


def nse_velocity_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    time_delta: float = 1.0,
    create_graph: bool = True,
) -> torch.Tensor:
    """NSE drift objective with exact spatial diffusion and the data-only ||b||^2 term."""
    if time_delta <= 0:
        raise ValueError("time_delta must be positive.")
    x0, x1, x2, time = (tensor.to(device, dtype=torch.float32) for tensor in batch)
    with torch.enable_grad():
        x1 = x1.detach().requires_grad_(True)
        error = (x2 - x1).detach()
        true_drift = error / time_delta
        predicted_drift = model(x1, time.detach(), condition=x0.detach())
        reduce_dims = tuple(range(1, predicted_drift.ndim))
        velocity_sq = predicted_drift.square().sum(dim=reduce_dims)
        alignment = 2.0 * (true_drift * predicted_drift).sum(dim=reduce_dims)
        drift_sq = true_drift.square().sum(dim=reduce_dims)
        diffusion = spatial_diffusion(predicted_drift, x1, error, time_delta, create_graph)
        objective = (velocity_sq - alignment + drift_sq - diffusion).mean()
        return objective if create_graph else objective.detach()


def spatial_diffusion(velocity, state, delta, time_delta, create_graph=True):
    """Exact delta^T J_v delta / dt, including all cross-coordinate terms."""
    delta = delta.detach()
    if velocity.requires_grad:
        vjp = torch.autograd.grad(velocity, state, grad_outputs=delta,
                                  create_graph=create_graph, retain_graph=True,
                                  allow_unused=True)[0]
    else:
        vjp = None
    if vjp is None:
        return torch.zeros(len(state), device=state.device, dtype=state.dtype)
    return (vjp * delta).flatten(1).sum(1) / time_delta


def nse_prediction_mse(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    time_delta: float = 1.0,
) -> torch.Tensor:
    """Mean pixel error of the one-step NSE forecast ``x1 + dt * v``."""
    x0, x1, x2, time = (tensor.to(device, dtype=torch.float32) for tensor in batch)
    predicted_drift = model(x1, time, condition=x0)
    return F.mse_loss(x1 + time_delta * predicted_drift, x2)


def kth_velocity_loss(model, current, reference, target, time, time_delta=1.0,
                      include_diffusion=True, create_graph=True):
    """KTH source objective with exact full-Jacobian contraction at fixed t.

    For delta = z_next - z, diffusion is delta^T J_v delta / dt.
    The VJP includes all off-diagonal terms without materializing J_v.
    dt=1 reproduces the original discrete-time KTH objective.
    """
    import math

    if not math.isfinite(time_delta) or time_delta <= 0:
        raise ValueError("KTH time_delta must be finite and positive.")
    with torch.enable_grad():
        z = current.detach().requires_grad_(include_diffusion)
        delta = (target.detach() - z.detach())
        velocity = model(z, time, ref=reference.detach())
        if velocity.shape != z.shape:
            raise ValueError("KTH velocity must have the same shape as its latent state.")
        dims = tuple(range(1, z.ndim))
        objective = velocity.square().sum(dims) - 2 * (delta * velocity).sum(dims) / time_delta
        if include_diffusion:
            objective = objective - spatial_diffusion(velocity, z, delta, time_delta, create_graph)
        return objective.mean() if create_graph else objective.mean().detach()


LOSS_REGISTRY = {
    "velocity_loss": velocity_loss,
    "drift_mse": validation_mse_loss,
    "nse_velocity_loss": nse_velocity_loss,
    "nse_prediction_mse": nse_prediction_mse,
    "kth_velocity_loss": kth_velocity_loss,
}
