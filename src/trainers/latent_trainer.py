from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from src.utils.checkpoint import save_checkpoint


class LatentTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        decoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        state_std: torch.Tensor,
        output_dir: str | Path,
        logger: logging.Logger,
        mixed_precision: bool = True,
        rollout_discount: float = 0.97,
        position_loss_weight: float = 1.0,
    ):
        self.model = model
        self.decoder = decoder.eval()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.state_std = state_std.to(device).view(1, 6)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.figure_dir = self.output_dir / "figures"
        self.result_dir = self.output_dir / "results"
        for directory in (self.checkpoint_dir, self.figure_dir, self.result_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.use_amp = mixed_precision and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.rollout_discount = rollout_discount
        self.loss_weights = torch.tensor(
            [position_loss_weight] * 3 + [1.0] * 3,
            dtype=torch.float32,
            device=device,
        ).view(1, 6)

    def _rollout_loss(self, sequence: torch.Tensor, future: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        current = sequence
        total = torch.zeros((), device=sequence.device)
        weight_sum = 0.0
        for step in range(future.shape[1]):
            prediction = self.model(current, time + step)
            normalized_error = (prediction - future[:, step, :]) / self.state_std
            weight = self.rollout_discount**step
            total = total + weight * (
                normalized_error.square() * self.loss_weights
            ).mean()
            weight_sum += weight
            current = torch.cat((current[:, 1:, :], prediction.unsqueeze(1)), dim=1)
        return total / weight_sum

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        batches = 0
        for sequence, future, time in loader:
            sequence = sequence.to(self.device, non_blocking=True)
            future = future.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                loss = self._rollout_loss(sequence, future, time.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach())
            batches += 1
        return total / max(batches, 1)

    @torch.no_grad()
    def evaluate_loader(self, loader) -> float:
        self.model.eval()
        total = 0.0
        batches = 0
        for sequence, future, time in loader:
            sequence = sequence.to(self.device, non_blocking=True)
            future = future.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
                loss = self._rollout_loss(sequence, future, time.to(self.device))
            total += float(loss)
            batches += 1
        return total / max(batches, 1)

    @torch.no_grad()
    def predict_rollout(self, trajectory: np.ndarray, sequence_length: int) -> np.ndarray:
        self.model.eval()
        seed = torch.tensor(
            trajectory[:sequence_length], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        current = seed
        predictions = [seed.squeeze(0)]
        for step in range(len(trajectory) - sequence_length):
            next_state = self.model(current, sequence_length - 1 + step)
            predictions.append(next_state)
            current = torch.cat((current[:, 1:, :], next_state.unsqueeze(1)), dim=1)
        return torch.cat(predictions, dim=0).float().cpu().numpy()

    @torch.no_grad()
    def rollout_metrics(
        self,
        truth: np.ndarray,
        prediction: np.ndarray,
        sequence_length: int,
    ) -> dict[str, float | list[float]]:
        truth_eval = truth[sequence_length:]
        pred_eval = prediction[sequence_length:]
        error = pred_eval - truth_eval
        normalized_error = error / self.state_std.squeeze(0).cpu().numpy()
        centered = truth_eval - truth_eval.mean(axis=0, keepdims=True)
        r2 = 1.0 - np.sum(error**2, axis=0) / np.maximum(
            np.sum(centered**2, axis=0), 1.0e-12
        )
        true_images = self.decoder(
            torch.tensor(truth_eval[:, :3], dtype=torch.float32, device=self.device)
        )
        pred_images = self.decoder(
            torch.tensor(pred_eval[:, :3], dtype=torch.float32, device=self.device)
        )
        decoded_mse = float((true_images - pred_images).square().mean().cpu())
        return {
            "raw_mse": float(np.mean(error**2)),
            "position_mse": float(np.mean(error[:, :3] ** 2)),
            "normalized_mse": float(np.mean(normalized_error**2)),
            "position_normalized_mse": float(np.mean(normalized_error[:, :3] ** 2)),
            "position_r2_mean": float(np.mean(r2[:3])),
            "state_r2": [float(value) for value in r2],
            "decoded_mse": decoded_mse,
        }

    def save_rollout_figures(
        self,
        truth: np.ndarray,
        prediction: np.ndarray,
        sequence_length: int,
        epoch: int,
        trajectory_name: str,
    ) -> tuple[Path, Path]:
        epoch_dir = self.figure_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        time = np.arange(len(truth))

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for index, axis in enumerate(axes):
            values = np.concatenate((truth[:, index], prediction[:, index]))
            span = max(float(values.max() - values.min()), 1.0e-6)
            axis.plot(
                time,
                truth[:, index],
                color="#2563EB",
                linewidth=2.1,
                label=f"true_z{index + 1}",
            )
            axis.plot(
                time,
                prediction[:, index],
                color="#F04452",
                linewidth=1.9,
                linestyle=(0, (5, 3)),
                label=f"pred_z{index + 1}",
            )
            axis.axvline(
                sequence_length - 1,
                color="#8A94A3",
                linestyle=":",
                linewidth=1.0,
            )
            axis.set_ylim(
                float(values.min()) - 0.06 * span,
                float(values.max()) + 0.23 * span,
            )
            axis.set_ylabel(f"z{index + 1}")
            axis.grid(True, alpha=0.22)
            axis.legend(loc="upper right", fontsize=8, framealpha=0.92)
        axes[-1].set_xlabel("time_step")
        fig.suptitle(f"Latent rollout · {trajectory_name} · epoch {epoch}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        latent_path = epoch_dir / "latent_rollout.png"
        fig.savefig(latent_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        snapshot_indices = np.linspace(0, len(truth) - 1, 8, dtype=int)
        with torch.no_grad():
            true_images = (
                self.decoder(
                    torch.tensor(
                        truth[snapshot_indices, :3],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                * 0.5
                + 0.5
            ).clamp(0, 1).cpu().numpy()
            pred_images = (
                self.decoder(
                    torch.tensor(
                        prediction[snapshot_indices, :3],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                * 0.5
                + 0.5
            ).clamp(0, 1).cpu().numpy()

        fig, axes = plt.subplots(2, len(snapshot_indices), figsize=(16, 4.6))
        for column, frame_index in enumerate(snapshot_indices):
            axes[0, column].imshow(true_images[column, 0], cmap="gray", vmin=0, vmax=1)
            axes[1, column].imshow(pred_images[column, 0], cmap="gray", vmin=0, vmax=1)
            axes[0, column].set_title(f"t={frame_index}", fontsize=10)
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
            if column == 0:
                axes[0, column].set_ylabel("Ground truth", fontsize=11)
                axes[1, column].set_ylabel("Prediction", fontsize=11)
        fig.suptitle(
            f"Decoded GPE rollout · {trajectory_name} · epoch {epoch}", fontsize=14
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.25)
        rollout_path = epoch_dir / "decoded_rollout.png"
        fig.savefig(rollout_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return latent_path, rollout_path

    def fit(
        self,
        train_loader,
        validation_loader,
        validation_trajectory: np.ndarray,
        validation_name: str,
        sequence_length: int,
        epochs: int,
        target_normalized_mse: float,
        target_position_r2: float,
        checkpoint_every: int = 5,
    ) -> dict:
        history: list[dict] = []
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            validation_loss = self.evaluate_loader(validation_loader)
            prediction = self.predict_rollout(validation_trajectory, sequence_length)
            metrics = self.rollout_metrics(
                validation_trajectory, prediction, sequence_length
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            self.save_rollout_figures(
                validation_trajectory,
                prediction,
                sequence_length,
                epoch,
                validation_name,
            )
            self.logger.info(
                "Epoch %d/%d | train=%.6f | val=%.6f | rollout_norm_mse=%.6f | pos_r2=%.5f | decoded_mse=%.6f",
                epoch,
                epochs,
                train_loss,
                validation_loss,
                metrics["normalized_mse"],
                metrics["position_r2_mean"],
                metrics["decoded_mse"],
            )
            metric = float(metrics["position_normalized_mse"])
            for filename in (f"epoch_{epoch:03d}.pt", "last.pt"):
                save_checkpoint(
                    self.checkpoint_dir / filename,
                    self.model,
                    self.optimizer,
                    epoch,
                    metric,
                    extra={"history": history, "validation_name": validation_name},
                )
            (self.result_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            self.scheduler.step()
        return {"history": history}
