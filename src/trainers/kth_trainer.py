"""KTH latent velocity training, exact-VJP loss and timed autoregressive rollout."""

from copy import deepcopy
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from src.method.loss import kth_velocity_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import kth_frame_metrics
from src.utils.visualization import save_video_rollout_comparison


class KTHTrainer:
    def __init__(self, model, codec, optimizer, device, config, project_root, logger, data_identity=None):
        from src.utils.config import resolve_path

        self.model, self.codec, self.optimizer = model, codec, optimizer
        self.device, self.config, self.logger = device, config, logger
        if not config.training.get("include_diffusion", True):
            raise ValueError("PhiBE training requires include_diffusion=true.")
        self.dt = float(config.dataset.frame_time_delta) * int(config.dataset.frame_stride)
        self.condition_frames = int(config.dataset.condition_frames)
        self.prediction_frames = int(config.dataset.prediction_frames)
        self.update_parameters = not bool(config.training.load_weights_only)
        self.global_step = 0
        self.start_epoch = 0
        self.data_identity = data_identity
        self.history = []
        self.ema = deepcopy(model).eval().requires_grad_(False)
        self.model.requires_grad_(self.update_parameters)
        for key in ("checkpoint_dir", "result_dir", "figure_dir"):
            value = resolve_path(project_root, config.paths[key])
            value.mkdir(parents=True, exist_ok=True)
            setattr(self, key, value)

    def time_signature(self):
        return {"frame_time_delta": float(self.config.dataset.frame_time_delta),
                "frame_stride": int(self.config.dataset.frame_stride),
                "time_origin": float(self.config.dataset.time_origin),
                "time_scale": float(self.config.model.time_scale)}

    def load(self, path, resume=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", {})
        if not any(k.startswith("time_embedding.") for k in state):
            raise ValueError("Legacy KTH v(z,ref) checkpoints cannot resume v(z,t,ref). "
                             "Train a new time-conditioned predictor; the VQ-VAE checkpoint is unchanged.")
        if checkpoint.get("time_signature") != self.time_signature():
            raise ValueError("KTH checkpoint time grid/embedding scale differs from the config.")
        if self.data_identity is None or checkpoint.get("data_identity") != self.data_identity:
            raise ValueError("KTH checkpoint dataset/split identity is missing or differs. "
                             "Use the original data and subject split; unverified legacy checkpoints are rejected.")
        self.model.load_state_dict(state)
        self.ema.load_state_dict(checkpoint.get("ema_state_dict", state))
        self.global_step = int(checkpoint.get("global_step", 0))
        if resume and self.update_parameters:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = int(checkpoint.get("epoch", 0))
            self.history = checkpoint.get("history", [])
            rng = checkpoint.get("rng_state")
            if rng is not None:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu())
                if self.device.type == "cuda" and rng["cuda"] is not None:
                    torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])

    def prepare_batch(self, frames, times, random_index=True):
        b, n = frames.shape[:2]
        index = (torch.randint(1, n-1, (b,)) if random_index
                 else torch.full((b,), self.condition_frames - 1, dtype=torch.long))
        row = torch.arange(b)
        triplets = torch.stack([frames[row, index-1], frames[row, index], frames[row, index+1]], 1)
        with torch.no_grad():
            latent = self.codec.encode(triplets.to(self.device))
        return latent[:, 1], latent[:, 0], latent[:, 2], times[row, index].to(self.device)

    def _loss(self, batch, training, model=None):
        current, reference, target, time = self.prepare_batch(*batch, random_index=training)
        loss = kth_velocity_loss(self.model if model is None else model, current, reference, target, time, self.dt,
                                 bool(self.config.training.include_diffusion),
                                 create_graph=training and self.update_parameters)
        return loss

    @torch.no_grad()
    def _update_ema(self):
        after = int(self.config.training.ema_start_step)
        decay = float(self.config.training.ema_decay) if self.global_step >= after else 0.0
        for average, weight in zip(self.ema.parameters(), self.model.parameters()):
            average.mul_(decay).add_(weight, alpha=1-decay)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def train_epoch(self, loader, epoch):
        self.model.train(self.update_parameters)
        total, count = 0.0, 0
        for batch in tqdm(loader, desc=f"KTH epoch {epoch}", leave=False):
            if self.update_parameters and self.global_step >= int(self.config.training.max_steps):
                break
            if self.update_parameters:
                warmup = int(self.config.training.warmup_steps)
                scale = min(1.0, (self.global_step+1) / warmup) if warmup > 0 else 1.0
                for group in self.optimizer.param_groups:
                    group["lr"] = float(self.config.training.lr) * scale
                self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(batch, training=self.update_parameters)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite KTH velocity loss.")
            if self.update_parameters:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.training.max_grad_norm),
                                               error_if_nonfinite=True)
                self.optimizer.step()
                self.global_step += 1
                self._update_ema()
            size = len(batch[0])
            total += float(loss.detach()) * size
            count += size
        return total / max(count, 1)

    @torch.no_grad()
    def rollout(self, context, times, steps=None, use_ema=False):
        self.model.eval()
        model = self.ema if use_ema else self.model
        steps = self.prediction_frames if steps is None else int(steps)
        if context.ndim != 5 or context.shape[1] < 2 or steps < 1:
            raise ValueError("Rollout requires [B,T,C,H,W], T>=2, and steps>=1.")
        if times.shape != context.shape[:2]:
            raise ValueError("One timestamp is required per conditioning frame.")
        # Only the last two latents condition the source model, not all 10 frames.
        latent = self.codec.encode(context[:, -2:].to(self.device))
        previous, current = latent[:, 0], latent[:, 1]
        time = times[:, -1].to(self.device)
        predicted = []
        for _ in range(steps):
            following = current + self.dt * model(current, time, ref=previous)
            if not torch.isfinite(following).all():
                raise FloatingPointError("Non-finite KTH rollout.")
            predicted.append(following)
            previous, current = current, following
            time = time + self.dt
        return self.codec.decode(torch.stack(predicted, 1)).clamp(-1, 1)

    @torch.no_grad()
    def evaluate_loader(self, loader, use_ema=False, detector=None):
        from src.utils.fvd import fvd_features, frechet_distance

        self.model.eval()
        sums, count = {}, 0
        real_features, fake_features = [], []
        for frames, times in loader:
            prediction = self.rollout(frames[:, :self.condition_frames], times[:, :self.condition_frames],
                                      use_ema=use_ema)
            truth = frames[:, self.condition_frames:].to(self.device)
            metrics = kth_frame_metrics(prediction, truth, self.config.evaluation.horizons)
            # Exact-VJP needs input autograd even in evaluation; it never updates parameters.
            metrics["velocity_loss"] = float(self._loss((frames, times), training=False,
                                                        model=self.ema if use_ema else self.model))
            size = len(frames)
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value * size
            count += size
            if detector is not None:
                fake = torch.cat([frames[:, :self.condition_frames].to(self.device), prediction], 1)
                real_features.append(fvd_features(frames.to(self.device), detector).cpu())
                fake_features.append(fvd_features(fake, detector).cpu())
        if not count:
            raise ValueError("Empty KTH test loader.")
        result = {key: value / count for key, value in sums.items()}
        result["num_test_videos"] = count
        if detector is not None:
            result["fvd"] = frechet_distance(torch.cat(fake_features), torch.cat(real_features))
        return result

    @torch.no_grad()
    def visualize(self, samples, tag, use_ema=False):
        for sample in samples:
            frames, times = sample["frames"], sample["times"]
            prediction = self.rollout(frames[None, :self.condition_frames], times[None, :self.condition_frames],
                                      use_ema=use_ema)[0].cpu()
            generated = torch.cat([frames[:self.condition_frames], prediction])
            save_video_rollout_comparison(
                frames, generated, times, self.figure_dir / tag / f"sample_{sample['sample_id']:02d}",
                condition_frames=self.condition_frames,
                max_frames=int(self.config.visualization.rollout_plot_frames),
                fps=float(self.config.visualization.fps))

    def fit(self, train_loader, rollout_loader, epochs, fixed_samples):
        if self.data_identity is None:
            raise ValueError("KTH training requires a verified dataset/split identity.")
        if rollout_loader.dataset is not train_loader.dataset:
            raise ValueError("Epoch rollout must use the training dataset.")
        for epoch in range(self.start_epoch+1, self.start_epoch+epochs+1):
            if self.update_parameters and self.global_step >= int(self.config.training.max_steps):
                break
            self._sync()
            start = perf_counter()
            train_loss = self.train_epoch(train_loader, epoch)
            self._sync()
            train_end = perf_counter()
            metrics = self.evaluate_loader(rollout_loader)
            self._sync()
            eval_end = perf_counter()
            every = int(self.config.visualization.every_n_epochs)
            tag = f"epoch_{epoch:03d}" if self.update_parameters else f"frozen_epoch_{epoch:03d}"
            if fixed_samples and every > 0 and epoch % every == 0:
                self.visualize(fixed_samples, tag)
            self._sync()
            record = dict(epoch=epoch, global_step=self.global_step, train_loss=train_loss,
                          train_seconds=train_end-start, train_rollout_seconds=eval_end-train_end,
                          rollout_split="train", visualization_seconds=perf_counter()-eval_end,
                          **{"train_" + key: value for key, value in metrics.items()})
            self.history.append(record)
            self.logger.info("KTH %s", record)
            if self.update_parameters:
                extra = dict(global_step=self.global_step, ema_state_dict=self.ema.state_dict(),
                             data_identity=self.data_identity, history=self.history, config=dict(self.config),
                             time_signature=self.time_signature(),
                             rng_state={"python": random.getstate(), "numpy": np.random.get_state(),
                                        "torch": torch.get_rng_state(),
                                        "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None})
                save_checkpoint(self.checkpoint_dir / f"epoch_{epoch:03d}.pt", self.model, self.optimizer,
                                epoch, metrics["mse"], extra)
                save_checkpoint(self.checkpoint_dir / "last.pt", self.model, self.optimizer,
                                epoch, metrics["mse"], extra)
            name = "train_history.json" if self.update_parameters else "frozen_history.json"
            (self.result_dir / name).write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return self.history
