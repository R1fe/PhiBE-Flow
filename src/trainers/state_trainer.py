from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.method.loss import validation_mse_loss, velocity_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import drift_mse_from_rollout
from src.utils.time import batch_time
from src.utils.visualization import visualize_acrobot_prediction


class StateTrainer:
    """Trainer for datasets that stay in their original state space."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        time_delta: float,
        checkpoint_dir: str | Path,
        result_dir: str | Path,
        log_dir: str | Path,
        logger,
        dataset_name: str = "acrobot_angles",
        update_parameters: bool = True,
        mixed_precision: bool = False,
        rollout_loss_weight: float = 0.0,
        feature_scales: list[float] | None = None,
        integration_method: str = "euler",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.time_delta = time_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.result_dir = Path(result_dir)
        self.log_dir = Path(log_dir)
        self.logger = logger
        self.dataset_name = dataset_name
        self.update_parameters = update_parameters
        self.mixed_precision = mixed_precision and device.type == "cuda"
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.rollout_loss_weight = rollout_loss_weight
        self.feature_scales = (
            None
            if feature_scales is None
            else torch.tensor(feature_scales, dtype=torch.float32, device=device)
        )
        if integration_method not in {"euler", "rk4"}:
            raise ValueError("integration_method must be 'euler' or 'rk4'.")
        self.integration_method = integration_method

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def fit(
        self,
        train_loader,
        epochs: int,
        rollout_loader=None,
        log_every: int = 1,
        visualization_samples: list[dict] | None = None,
        figure_dir: str | Path | None = None,
        transform_angles: bool = False,
        visualize_every: int = 1,
    ) -> dict[str, list[float]]:
        history = {
            "train_velocity_loss": [],
            "train_drift_mse": [],
            "train_eval_velocity_loss": [],
            "visualization_mse": [],
            "train_rollout_mse": [],
        }
        if rollout_loader is None:
            from torch.utils.data import DataLoader
            rollout_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size or 1,
                                        shuffle=False)
        if rollout_loader.dataset is not train_loader.dataset:
            raise ValueError("Epoch rollout must use the training dataset.")
        figure_dir = Path(figure_dir) if figure_dir is not None else None

        for epoch in range(1, epochs + 1):
            train_loss = self._run_train_epoch(train_loader, epoch, epochs)
            train_metrics = self.evaluate_loader(rollout_loader)
            visualization_mse = None

            if (
                visualization_samples
                and figure_dir is not None
                and visualize_every > 0
                and epoch % visualize_every == 0
            ):
                visualization_mse = self._visualize_samples(
                    samples=visualization_samples,
                    epoch=epoch,
                    figure_dir=figure_dir,
                    transform_angles=transform_angles,
                )

            history["train_velocity_loss"].append(train_loss)
            history["train_drift_mse"].append(train_metrics["drift_mse"])
            history["train_eval_velocity_loss"].append(train_metrics["velocity_loss"])
            history["visualization_mse"].append(visualization_mse)
            history["train_rollout_mse"].append(train_metrics["rollout_mse"])

            self.writer.add_scalar("loss/train_velocity", train_loss, epoch)
            self.writer.add_scalar("loss/train_eval_velocity", train_metrics["velocity_loss"], epoch)
            self.writer.add_scalar("loss/train_drift_mse", train_metrics["drift_mse"], epoch)
            self.writer.add_scalar("train/rollout_mse", train_metrics["rollout_mse"], epoch)
            if visualization_mse is not None:
                self.writer.add_scalar("viz/train_rollout_mse", visualization_mse, epoch)

            if epoch % log_every == 0:
                message = "Epoch %s/%s | train_velocity_loss=%.6f" % (epoch, epochs, train_loss)
                message += (
                    " | train_eval_velocity_loss=%.6f | train_drift_mse=%.6f | train_rollout_mse=%.6f"
                    % (train_metrics["velocity_loss"], train_metrics["drift_mse"], train_metrics["rollout_mse"])
                )
                if visualization_mse is not None:
                    message += " | viz_rollout_mse=%.6f" % visualization_mse
                self.logger.info(message)

            if self.update_parameters:
                save_checkpoint(self.checkpoint_dir / "last.pt", self.model, self.optimizer,
                                epoch=epoch, extra={"history": history})
            if self.update_parameters:
                save_checkpoint(
                    path=self.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metric=train_metrics["drift_mse"],
                    extra={"history": history},
                )

        history_path = self.result_dir / ("train_history.json" if self.update_parameters else "frozen_history.json")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        self.writer.close()
        return history

    def evaluate_loader(self, data_loader) -> dict[str, float]:
        self.model.eval()
        total_velocity_loss = 0.0
        total_drift_mse = 0.0
        num_samples = 0

        for batch in data_loader:
            sequence, target, time = batch
            one_step_target = target[:, 0, :] if target.dim() == 3 else target
            with torch.enable_grad():
                velocity_term = velocity_loss(
                    self.model,
                    (sequence, one_step_target, time),
                    self.device,
                    time_delta=self.time_delta,
                    dataset_name=self.dataset_name,
                )
            with torch.no_grad():
                drift_mse = validation_mse_loss(
                    self.model,
                    (sequence, one_step_target, time),
                    self.device,
                    time_delta=self.time_delta,
                    dataset_name=self.dataset_name,
                )

            total_velocity_loss += velocity_term.item() * len(sequence)
            total_drift_mse += drift_mse.item() * len(sequence)
            num_samples += len(sequence)

        if num_samples == 0:
            raise ValueError("Received an empty data loader during evaluation.")

        return {
            "velocity_loss": total_velocity_loss / num_samples,
            "drift_mse": total_drift_mse / num_samples,
            **self.evaluate_rollouts(data_loader.dataset),
        }

    def evaluate_rollouts(self, dataset):
        """Run every held-out trajectory to its end, starting from true context only."""
        total, elements = 0.0, 0
        seq_length = dataset.seq_length
        for features, time in dataset.rollout_trajectories:
            predicted = self.predict_rollout(torch.as_tensor(features[:seq_length]),
                                             len(features) - seq_length, start_time=time).numpy()
            size = features[seq_length:].size
            total += drift_mse_from_rollout(predicted, features, condition_frames=seq_length) * size
            elements += size
        if not elements:
            raise ValueError("No held-out future states for rollout evaluation.")
        return dict(rollout_mse=total/elements, num_rollout_trajectories=len(dataset.rollout_trajectories))

    def predict_rollout(
        self,
        seed_sequence: torch.Tensor,
        rollout_steps: int,
        integrate_velocity: bool = True,
        *,
        start_time: float | torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressively roll out future states from an initial sequence."""
        self.model.eval()

        if seed_sequence.dim() == 2:
            current_sequence = seed_sequence.unsqueeze(0).to(self.device)
        else:
            current_sequence = seed_sequence.to(self.device)

        generated = current_sequence.clone()
        time = batch_time(start_time, current_sequence)
        for _ in range(rollout_steps):
            with torch.no_grad():
                if integrate_velocity:
                    next_state = self._integrate_next_state(current_sequence, time)
                else:
                    next_state = self.model(current_sequence, time)

                next_state = next_state.unsqueeze(1)
                current_sequence = torch.cat([current_sequence[:, 1:, :], next_state], dim=1)
                generated = torch.cat([generated, next_state], dim=1)
                time = time + self.time_delta

        return generated.squeeze(0).cpu()

    def _run_train_epoch(self, train_loader, epoch: int, epochs: int) -> float:
        self.model.train(self.update_parameters)
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch in progress:
            if self.update_parameters:
                self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                sequence, target, time = batch
                if target.dim() == 3 and self.rollout_loss_weight > 0.0:
                    loss = self._rollout_aware_loss(sequence, target, time)
                else:
                    loss = velocity_loss(
                        self.model,
                        (sequence, target[:, 0, :] if target.dim() == 3 else target, time),
                        self.device,
                        time_delta=self.time_delta,
                        dataset_name=self.dataset_name,
                    )
            if self.update_parameters:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(1, len(train_loader))

    def _rollout_aware_loss(
        self,
        sequence: torch.Tensor,
        targets: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Combine one-step drift fitting with differentiable free rollouts."""
        context = sequence.to(self.device)
        time = batch_time(time, context)
        targets = targets.to(self.device)
        scales = (
            self.feature_scales
            if self.feature_scales is not None
            else torch.ones(targets.shape[-1], device=self.device, dtype=targets.dtype)
        )

        one_step_loss = velocity_loss(self.model, (context, targets[:, 0, :], time),
                                      self.device, self.time_delta, self.dataset_name,
                                      include_diffusion=True)

        rollout_terms = []
        for step in range(targets.shape[1]):
            next_state = self._integrate_next_state(context, time + step * self.time_delta)
            normalized_error = (
                (next_state - targets[:, step, :]) / scales / self.time_delta
            )
            rollout_terms.append(normalized_error.square().mean())
            context = torch.cat((context[:, 1:, :], next_state.unsqueeze(1)), dim=1)

        rollout_loss = torch.stack(rollout_terms).mean()
        return one_step_loss + self.rollout_loss_weight * rollout_loss

    def _integrate_next_state(self, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time = batch_time(time, context)
        current_state = context[:, -1, :]
        if self.integration_method == "euler":
            return current_state + self.model(context, time) * self.time_delta

        def velocity_at(state: torch.Tensor, stage_time: torch.Tensor) -> torch.Tensor:
            stage_context = torch.cat(
                (context[:, :-1, :], state.unsqueeze(1)), dim=1
            )
            return self.model(stage_context, stage_time)

        half_dt = 0.5 * self.time_delta
        k1 = velocity_at(current_state, time)
        k2 = velocity_at(current_state + half_dt * k1, time + half_dt)
        k3 = velocity_at(current_state + half_dt * k2, time + half_dt)
        k4 = velocity_at(current_state + self.time_delta * k3, time + self.time_delta)
        return current_state + (self.time_delta / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    def _visualize_samples(
        self,
        samples: list[dict],
        epoch: int,
        figure_dir: Path,
        transform_angles: bool,
    ) -> float:
        epoch_dir = figure_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        rollout_mse_values = []
        for sample in samples:
            features = sample["features"]
            seq_length = int(sample["seq_length"])
            rollout_steps = int(sample["rollout_steps"])
            seed_sequence = torch.tensor(features[:seq_length], dtype=torch.float32)
            predicted = self.predict_rollout(
                seed_sequence=seed_sequence,
                rollout_steps=rollout_steps,
                integrate_velocity=True,
                start_time=sample["start_time"],
            ).numpy()
            ground_truth = np.asarray(features[: seq_length + rollout_steps], dtype=np.float32)

            tag = f"sample_{sample['sample_id']:02d}_traj_{sample['trajectory_index']:04d}"
            visualize_acrobot_prediction(
                true_traj=ground_truth,
                pred_traj=predicted,
                output_dir=epoch_dir,
                tag=tag,
                transform_angles=transform_angles,
            )
            rollout_mse_values.append(drift_mse_from_rollout(predicted, ground_truth,
                                                            condition_frames=seq_length))

        return float(np.mean(rollout_mse_values)) if rollout_mse_values else float("nan")
