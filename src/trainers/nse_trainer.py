from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.nse import resize_nse_input, resize_nse_target
from src.method.loss import nse_prediction_mse, nse_velocity_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.visualization import save_nse_rollout_comparison


class NSETrainer:
    """Image-space trainer for the conditioned NSE drift U-Net."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
        result_dir: str | Path,
        figure_dir: str | Path,
        logger,
        *,
        time_delta: float = 1.0,
        max_grad_norm: float = 1.0,
        update_parameters: bool = True,
        lo_size: int = 64,
        hi_size: int = 64,
        time_lag: int = 2,
        time_origin: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.result_dir = Path(result_dir)
        self.figure_dir = Path(figure_dir)
        self.logger = logger
        self.time_delta = float(time_delta)
        self.max_grad_norm = float(max_grad_norm)
        self.update_parameters = bool(update_parameters)
        self.lo_size = int(lo_size)
        self.hi_size = int(hi_size)
        self.time_lag = int(time_lag)
        self.time_origin = float(time_origin)
        self.global_step = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader,
        test_loader,
        epochs: int,
        *,
        fixed_test_trajectories: list[dict] | None = None,
        visualize_every: int = 1,
        rollout_steps: int = 6,
        rollout_plot_frames: int = 8,
    ) -> dict[str, list[float]]:
        history = {"train_velocity_loss": [], "test_velocity_loss": [], "test_mse": []}
        best_test_mse = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            test_metrics = self.evaluate_loader(test_loader)
            history["train_velocity_loss"].append(train_loss)
            history["test_velocity_loss"].append(test_metrics["velocity_loss"])
            history["test_mse"].append(test_metrics["mse"])

            self.logger.info(
                "Epoch %s/%s | train_velocity_loss=%.6f | test_velocity_loss=%.6f | test_mse=%.6f",
                epoch,
                epochs,
                train_loss,
                test_metrics["velocity_loss"],
                test_metrics["mse"],
            )
            if fixed_test_trajectories and visualize_every > 0 and epoch % visualize_every == 0:
                self.visualize_rollouts(
                    fixed_test_trajectories,
                    epoch=epoch,
                    rollout_steps=rollout_steps,
                    plot_frames=rollout_plot_frames,
                )

            if self.update_parameters:
                for filename in (f"epoch_{epoch:03d}.pt", "last.pt"):
                    save_checkpoint(self.checkpoint_dir / filename, self.model, self.optimizer,
                                    epoch=epoch, metric=test_metrics["mse"],
                                    extra={"global_step": self.global_step, "history": history})
            if self.update_parameters and test_metrics["mse"] < best_test_mse:
                best_test_mse = test_metrics["mse"]
                save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    metric=best_test_mse,
                    extra={"global_step": self.global_step, "history": history},
                )

        history_name = "train_history.json" if self.update_parameters else "frozen_history.json"
        (self.result_dir / history_name).write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        return history

    def _train_epoch(self, data_loader, epoch: int, epochs: int) -> float:
        self.model.train(self.update_parameters)
        total_loss = 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            if self.update_parameters:
                self.optimizer.zero_grad(set_to_none=True)
            loss = nse_velocity_loss(
                self.model, batch, self.device, time_delta=self.time_delta
            )
            if self.update_parameters:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            self.global_step += 1
            total_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        return total_loss / max(len(data_loader), 1)

    @torch.no_grad()
    def evaluate_loader(self, data_loader) -> dict[str, float]:
        self.model.eval()
        velocity_total = 0.0
        mse_total = 0.0
        samples = 0
        for batch in data_loader:
            size = len(batch[0])
            velocity_total += size * float(
                nse_velocity_loss(self.model, batch, self.device, self.time_delta)
            )
            mse_total += size * float(nse_prediction_mse(self.model, batch, self.device, self.time_delta))
            samples += size
        if samples == 0:
            raise ValueError("Received an empty NSE test loader.")
        return {"velocity_loss": velocity_total / samples, "mse": mse_total / samples}

    @torch.no_grad()
    def rollout(self, trajectory: torch.Tensor, rollout_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out on the same temporal grid and coarse-input path as forecasting_new."""
        self.model.eval()
        required_frames = (rollout_steps + 1) * self.time_lag + 1
        if trajectory.shape[0] < required_frames:
            rollout_steps = max(0, (trajectory.shape[0] - 1) // self.time_lag - 1)
        indices = torch.arange(rollout_steps + 2) * self.time_lag
        ground_truth = resize_nse_target(trajectory[indices], self.hi_size).to(self.device)
        x0 = resize_nse_input(trajectory[indices[0]], self.lo_size, self.hi_size).to(self.device)
        x1 = resize_nse_input(trajectory[indices[1]], self.lo_size, self.hi_size).to(self.device)
        generated = [x0, x1]
        time = torch.full((1,), self.time_origin + self.time_delta, device=self.device, dtype=x1.dtype)

        for _ in range(rollout_steps):
            drift = self.model(x1, time, condition=x0)
            prediction = x1 + self.time_delta * drift
            generated.append(prediction)
            x0 = x1
            x1 = resize_nse_input(prediction, self.lo_size, self.hi_size)
            time = time + self.time_delta
        return ground_truth.cpu(), torch.cat(generated, dim=0).cpu()

    def visualize_rollouts(
        self,
        samples: list[dict],
        *,
        epoch: int,
        rollout_steps: int,
        plot_frames: int,
    ) -> None:
        epoch_dir = self.figure_dir / f"epoch_{epoch:03d}"
        for sample in samples:
            ground_truth, generated = self.rollout(sample["trajectory"], rollout_steps)
            save_nse_rollout_comparison(
                ground_truth[:, 0].numpy(),
                generated[:, 0].numpy(),
                epoch_dir / f"sample_{sample['sample_id']:02d}_rollout.png",
                max_frames=plot_frames,
            )
