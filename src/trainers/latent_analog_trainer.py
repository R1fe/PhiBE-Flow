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


class LatentAnalogTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        decoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        position_std: torch.Tensor,
        output_dir: str | Path,
        logger: logging.Logger,
        mixed_precision: bool = True,
        velocity_loss_weight: float = 0.25,
        correction_penalty: float = 0.001,
    ):
        self.model = model
        self.decoder = decoder.eval()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.position_std = position_std.to(device).view(1, 1, 3)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.figure_dir = self.output_dir / "figures"
        self.result_dir = self.output_dir / "results"
        for directory in (self.checkpoint_dir, self.figure_dir, self.result_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.use_amp = mixed_precision and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.velocity_loss_weight = velocity_loss_weight
        self.correction_penalty = correction_penalty

    def _loss(
        self,
        context: torch.Tensor,
        analog_context: torch.Tensor,
        analog_future: torch.Tensor,
        target_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(context, analog_context, analog_future)
        position_error = (prediction - target_future) / self.position_std
        position_loss = position_error.square().mean()
        predicted_velocity = prediction[:, 1:] - prediction[:, :-1]
        target_velocity = target_future[:, 1:] - target_future[:, :-1]
        velocity_error = (predicted_velocity - target_velocity) / self.position_std
        velocity_loss = velocity_error.square().mean()
        correction = (prediction - analog_future) / self.position_std
        loss = (
            position_loss
            + self.velocity_loss_weight * velocity_loss
            + self.correction_penalty * correction.square().mean()
        )
        return loss, prediction

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        batches = 0
        for context, analog_context, analog_future, target_future in loader:
            tensors = [
                tensor.to(self.device, non_blocking=True)
                for tensor in (context, analog_context, analog_future, target_future)
            ]
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                loss, _ = self._loss(*tensors)
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
        for context, analog_context, analog_future, target_future in loader:
            tensors = [
                tensor.to(self.device, non_blocking=True)
                for tensor in (context, analog_context, analog_future, target_future)
            ]
            with autocast(enabled=self.use_amp):
                loss, _ = self._loss(*tensors)
            total += float(loss)
            batches += 1
        return total / max(batches, 1)

    @torch.no_grad()
    def predict(
        self,
        context: np.ndarray,
        analog_context: np.ndarray,
        analog_future: np.ndarray,
    ) -> np.ndarray:
        self.model.eval()
        inputs = [
            torch.tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
            for value in (context, analog_context, analog_future)
        ]
        return self.model(*inputs).squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def metrics(self, truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
        error = prediction - truth
        centered = truth - truth.mean(axis=0, keepdims=True)
        r2 = 1.0 - np.sum(error**2, axis=0) / np.maximum(
            np.sum(centered**2, axis=0), 1.0e-12
        )
        true_images = self.decoder(
            torch.tensor(truth, dtype=torch.float32, device=self.device)
        )
        pred_images = self.decoder(
            torch.tensor(prediction, dtype=torch.float32, device=self.device)
        )
        return {
            "position_mse": float(np.mean(error**2)),
            "normalized_position_mse": float(
                np.mean((error / self.position_std.cpu().numpy().reshape(1, 3)) ** 2)
            ),
            "position_r2": [float(value) for value in r2],
            "position_r2_mean": float(np.mean(r2)),
            "decoded_mse": float((true_images - pred_images).square().mean().cpu()),
        }

    def save_figures(
        self,
        full_truth: np.ndarray,
        full_prediction: np.ndarray,
        sequence_length: int,
        epoch: int,
        trajectory_name: str,
        analog_name: str,
    ) -> tuple[Path, Path]:
        epoch_dir = self.figure_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        time = np.arange(len(full_truth))
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for index, axis in enumerate(axes):
            values = np.concatenate((full_truth[:, index], full_prediction[:, index]))
            span = max(float(values.max() - values.min()), 1.0e-6)
            axis.plot(time, full_truth[:, index], color="#2563EB", linewidth=2.2, label=f"true_z{index + 1}")
            axis.plot(time, full_prediction[:, index], color="#F04452", linewidth=1.9, linestyle=(0, (5, 3)), label=f"pred_z{index + 1}")
            axis.axvline(sequence_length - 1, color="#8993A4", linestyle=":", linewidth=1.0)
            axis.set_ylim(float(values.min()) - 0.06 * span, float(values.max()) + 0.22 * span)
            axis.set_ylabel(f"z{index + 1}")
            axis.grid(True, alpha=0.22)
            axis.legend(loc="upper right", fontsize=8, framealpha=0.94)
        axes[-1].set_xlabel("time_step")
        fig.suptitle(
            f"Latent rollout · {trajectory_name} · epoch {epoch}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        latent_path = epoch_dir / "latent_rollout.png"
        fig.savefig(latent_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        snapshot_indices = np.linspace(0, len(full_truth) - 1, 15, dtype=int)
        true_tensor = torch.tensor(
            full_truth[snapshot_indices], dtype=torch.float32, device=self.device
        )
        pred_tensor = torch.tensor(
            full_prediction[snapshot_indices], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            true_images = ((self.decoder(true_tensor) + 1.0) * 0.5).clamp(0, 1).cpu().numpy()
            pred_images = ((self.decoder(pred_tensor) + 1.0) * 0.5).clamp(0, 1).cpu().numpy()
        fig, axes = plt.subplots(
            2,
            len(snapshot_indices),
            figsize=(24, 3.85),
            gridspec_kw={"height_ratios": [1, 1]},
        )
        for column, frame_index in enumerate(snapshot_indices):
            axes[0, column].imshow(true_images[column, 0], cmap="gray", vmin=0, vmax=1)
            axes[1, column].imshow(pred_images[column, 0], cmap="gray", vmin=0, vmax=1)
            axes[0, column].set_title(f"t={frame_index}", fontsize=10)
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                axes[row, column].set_box_aspect(1)
        fig.text(0.065, 0.61, "Ground truth", ha="right", va="center", fontsize=11)
        fig.text(0.065, 0.25, "Prediction", ha="right", va="center", fontsize=11)
        fig.suptitle(
            f"Decoded GPE rollout · {trajectory_name} · epoch {epoch}",
            fontsize=14,
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.997,
            bottom=0.025,
            top=0.82,
            wspace=0.025,
            hspace=0.025,
        )
        decoded_path = epoch_dir / "decoded_rollout.png"
        fig.savefig(decoded_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return latent_path, decoded_path

    def fit(
        self,
        train_loader,
        validation_loader,
        fixed_sample: dict[str, object],
        epochs: int,
        target_r2: float,
        checkpoint_every: int = 10,
        plot_every: int = 5,
        minimum_epochs: int = 0,
    ) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        best_mse = float("inf")
        for epoch in range(0, epochs + 1):
            train_loss = None if epoch == 0 else self.train_epoch(train_loader)
            validation_loss = self.evaluate_loader(validation_loader)
            prediction = self.predict(
                fixed_sample["context"],
                fixed_sample["analog_context"],
                fixed_sample["analog_future"],
            )
            metrics = self.metrics(fixed_sample["target_future"], prediction)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            if epoch == 0 or epoch % plot_every == 0:
                full_prediction = np.concatenate(
                    (fixed_sample["context"][:, :3], prediction), axis=0
                )
                full_truth = np.concatenate(
                    (fixed_sample["context"][:, :3], fixed_sample["target_future"]),
                    axis=0,
                )
                self.save_figures(
                    full_truth,
                    full_prediction,
                    len(fixed_sample["context"]),
                    epoch,
                    str(fixed_sample["name"]),
                    str(fixed_sample["analog_name"]),
                )
            self.logger.info(
                "Epoch %d/%d | train=%s | val=%.6f | rollout_mse=%.6f | r2=%.5f | decoded_mse=%.6f",
                epoch,
                epochs,
                "baseline" if train_loss is None else f"{train_loss:.6f}",
                validation_loss,
                metrics["position_mse"],
                metrics["position_r2_mean"],
                metrics["decoded_mse"],
            )
            metric = float(metrics["position_mse"])
            if metric < best_mse:
                best_mse = metric
                save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    epoch,
                    metric,
                    extra={"history": history, "fixed_sample": fixed_sample["name"]},
                )
                (self.result_dir / "best_metrics.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
            if epoch > 0 and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    self.model,
                    self.optimizer,
                    epoch,
                    metric,
                    extra={"history": history, "fixed_sample": fixed_sample["name"]},
                )
            (self.result_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if epoch > 0:
                self.scheduler.step()
            if epoch >= minimum_epochs and float(metrics["position_r2_mean"]) >= target_r2:
                self.logger.info("Similarity target reached at epoch %d.", epoch)
                if epoch % plot_every != 0:
                    full_prediction = np.concatenate((fixed_sample["context"][:, :3], prediction), axis=0)
                    full_truth = np.concatenate((fixed_sample["context"][:, :3], fixed_sample["target_future"]), axis=0)
                    self.save_figures(
                        full_truth,
                        full_prediction,
                        len(fixed_sample["context"]),
                        epoch,
                        str(fixed_sample["name"]),
                        str(fixed_sample["analog_name"]),
                    )
                break
        return history
