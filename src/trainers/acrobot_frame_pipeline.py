"""Train an independent image codec, then frozen-codec PhiBE latent forecasting."""

import json
import logging
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.datasets.acrobot_frames import build_acrobot_frame_datasets
from src.method.loss import velocity_loss
from src.models.acrobot_frame_model import AcrobotFrameModel
from src.models.external_codec import build_image_codec
from src.utils.config import resolve_path
from src.utils.logger import build_logger
from src.utils.metrics import kth_frame_metrics
from src.utils.visualization import save_video_rollout_comparison


class AcrobotFrameTrainer:
    def __init__(self, config, device, root, restoring=False):
        self.config, self.device = config, device
        if not config.training.get("include_diffusion", True):
            raise ValueError("PhiBE training requires include_diffusion=true.")
        d, m = config.dataset, config.model
        if d.mode != "window" or not d.normalize:
            raise ValueError("Image training requires window mode and normalize=true.")
        if config.evaluation.get("fvd", False):
            raise NotImplementedError("This image smoke baseline reports pixel metrics, not Acrobot FVD.")
        self.horizons = [int(h) for h in config.evaluation.horizons]
        if not self.horizons or any(h < 1 or h > d.prediction_horizon for h in self.horizons):
            raise ValueError("evaluation.horizons must be within dataset.prediction_horizon.")
        self.dt = float(d.time_delta) * int(d.time_lag)
        self.signature = dict(image_size=d.image_size, seq_length=d.seq_length,
                              latent_dim=m.latent_dim, width=m.width, time_scale=m.time_scale,
                              dt=self.dt, time_origin=d.time_origin)
        codec, identity, self.needs_codec_training = build_image_codec(config, root, device, restoring)
        self.signature["codec"] = identity
        self.model = AcrobotFrameModel(d.image_size, d.seq_length, codec=codec, **dict(m)).to(device)
        self.optimizer = torch.optim.Adam(self.model.velocity.parameters(), lr=config.training.lr)
        self.update_parameters = not config.training.load_weights_only
        self.epoch = 0
        self.dirs = {key: resolve_path(root, value) for key, value in config.paths.items()}
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        self.logger = build_logger("acrobot_frames", self.dirs["log_dir"] / "pipeline.log")

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def freeze_codec(self):
        self.model.codec.eval()
        for parameter in self.model.codec.parameters():
            parameter.requires_grad_(False)

    def save(self, name, split):
        torch.save(dict(format="acrobot_image_pipeline_v2", model=self.model.state_dict(),
                        optimizer=self.optimizer.state_dict(), epoch=self.epoch,
                        signature=self.signature, split=split), self.dirs["checkpoint_dir"] / name)

    def load(self, path, split):
        payload = torch.load(path, map_location=self.device)
        if payload.get("format") not in {"acrobot_image_baseline_v1", "acrobot_image_pipeline_v2"}:
            raise ValueError("Expected a unified image-pipeline checkpoint; configure raw GPE weights under codec instead.")
        if payload["format"] == "acrobot_image_baseline_v1":
            payload["signature"].setdefault("codec", {"type": "mlp"})
        if payload["signature"] != self.signature or payload["split"] != split:
            raise ValueError("Checkpoint architecture/time grid or train/test trajectory split differs.")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.training.lr
        self.epoch = int(payload["epoch"])
        self.freeze_codec()

    def pretrain_codec(self, loader):
        epochs = int(self.config.training.codec_epochs)
        if epochs < 1:
            raise ValueError("Training without weights requires codec_epochs >= 1.")
        optimizer = torch.optim.Adam(self.model.codec.parameters(), lr=self.config.training.codec_lr)
        history = []
        for epoch in range(1, epochs+1):
            self.sync()
            start = perf_counter()
            total, count = 0.0, 0
            self.model.codec.train()
            for context, future, _ in loader:
                frames = torch.cat((context, future), dim=1).flatten(0, 1).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(self.model.codec(frames), frames)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite codec loss.")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(frames)
                count += len(frames)
            self.sync()
            row = dict(codec_epoch=epoch, reconstruction_mse_m11=total/count,
                       seconds=perf_counter()-start)
            history.append(row)
            self.logger.info("Codec %s", row)
        (self.dirs["result_dir"] / "codec_history.json").write_text(json.dumps(history, indent=2))
        self.freeze_codec()

    def train_epoch(self, loader):
        self.model.velocity.train(self.update_parameters)
        total, count = 0.0, 0
        for context, future, time in loader:
            with torch.no_grad():
                sequence = self.model.encode_sequence(context.to(self.device))
                target = self.model.codec.encode(future[:, 0].to(self.device))
            # The codec is fixed: no future images can change the conditioning representation.
            loss = velocity_loss(self.model.velocity, (sequence, target, time), self.device,
                                 self.dt, dataset_name="acrobot_frames",
                                 include_diffusion=bool(self.config.training.include_diffusion))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite latent velocity loss.")
            if self.update_parameters:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.velocity.parameters(),
                                               self.config.training.max_grad_norm)
                self.optimizer.step()
            total += float(loss.detach()) * len(context)
            count += len(context)
        return total/count

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        totals, count = {}, 0
        for context, future, time in loader:
            context, future = context.to(self.device), future.to(self.device)
            prediction = self.model.rollout(context, time, future.shape[1], self.dt)
            metrics = kth_frame_metrics(prediction, future, self.horizons)
            reconstructed = self.model.codec(future.flatten(0, 1)).reshape_as(future)
            metrics["codec_mse"] = float(F.mse_loss((reconstructed+1)/2, (future+1)/2))
            persistence = context[:, -1:].expand_as(future)
            metrics["persistence_mse"] = float(F.mse_loss((persistence+1)/2, (future+1)/2))
            if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
                raise FloatingPointError("Non-finite image evaluation metrics.")
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * len(context)
            count += len(context)
        return dict({key: value/count for key, value in totals.items()}, num_test_windows=count)

    @torch.no_grad()
    def visualize(self, samples, tag):
        self.model.eval()
        for sample in samples:
            context, future, time = sample["batch"]
            prediction = self.model.rollout(context[None].to(self.device), time,
                                            len(future), self.dt)[0].cpu()
            truth = torch.cat((context, future))
            generated = torch.cat((context, prediction))
            times = time + (torch.arange(len(truth)) - len(context)+1) * self.dt
            save_video_rollout_comparison(
                truth, generated, times,
                self.dirs["figure_dir"] / tag / f"sample_{sample['sample_id']:02d}",
                condition_frames=len(context), max_frames=self.config.visualization.rollout_plot_frames,
                fps=self.config.visualization.fps, title="Acrobot Image Rollout Comparison")


def run_acrobot_frames(config, args, device, project_root, evaluation=False):
    try:
        if config.training.load_weights_only and not config.training.checkpoint_path and not evaluation:
            raise ValueError("Frozen mode requires training.checkpoint_path.")
        train, test = build_acrobot_frame_datasets(config, project_root)
        kwargs = dict(batch_size=config.dataset.batch_size, num_workers=config.dataset.num_workers)
        train_loader = DataLoader(train, shuffle=True, **kwargs)
        test_loader = DataLoader(test, shuffle=False, **kwargs)
        rollout_loader = DataLoader(train, shuffle=False, **kwargs)
        checkpoint = config.training.checkpoint_path
        if evaluation:
            checkpoint = getattr(args, "checkpoint", None) or config.evaluation.checkpoint
        trainer = AcrobotFrameTrainer(config, device, project_root, restoring=bool(checkpoint))
        split = dict(train=train.trajectory_names, test=test.trajectory_names)
        if checkpoint:
            if evaluation and not getattr(args, "checkpoint", None):
                path = trainer.dirs["checkpoint_dir"] / checkpoint
            else:
                path = resolve_path(project_root, checkpoint)
            trainer.load(path, split)
        count = int(config.visualization.num_fixed_samples) if config.visualization.enabled else 0
        if count < 0 or int(config.visualization.every_n_epochs) < 1:
            raise ValueError("Invalid visualization count or interval.")
        sample_dataset = test if evaluation else train
        indices = torch.randperm(len(sample_dataset), generator=torch.Generator().manual_seed(config.training.seed+1))[:count]
        samples = []
        for sample_id, index in enumerate(indices.tolist()):
            trajectory, start = sample_dataset.indices[index]
            samples.append(dict(sample_id=sample_id, window_index=index,
                                trajectory=sample_dataset.trajectory_names[trajectory], start_frame=start,
                                batch=sample_dataset[index]))
        result_dir = trainer.dirs["result_dir"]
        (result_dir / "split_manifest.json").write_text(json.dumps(split, indent=2))
        sample_file = "eval_fixed_test_samples.json" if evaluation else "fixed_train_samples.json"
        (result_dir / sample_file).write_text(json.dumps(
            [{k: v for k, v in sample.items() if k != "batch"} for sample in samples], indent=2))
        if evaluation:
            trainer.sync()
            start = perf_counter()
            metrics = trainer.evaluate(test_loader)
            trainer.sync()
            metrics["evaluation_seconds"] = perf_counter()-start
            metrics["codec_identity"] = trainer.signature["codec"]
            (result_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
            trainer.visualize(samples, "evaluation")
            trainer.logger.info("Image test %s", metrics)
            return metrics
        epochs = args.epochs if args.epochs is not None else int(config.training.epochs)
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        if not checkpoint:
            if trainer.needs_codec_training:
                trainer.logger.info("Codec initialization uses train-only reconstruction, not the GPE geometry objective.")
                trainer.pretrain_codec(train_loader)
            else:
                trainer.freeze_codec()
            trainer.save("codec_initialized.pt", split)
        (result_dir / "codec_info.json").write_text(json.dumps(dict(
            identity=trainer.signature["codec"],
            initialization=config.get("codec", {}).get("initialization", "random"),
            reconstruction_pretraining=not bool(checkpoint) and trainer.needs_codec_training), indent=2))
        history = []
        for _ in range(epochs):
            trainer.epoch += 1
            trainer.sync()
            start = perf_counter()
            loss = trainer.train_epoch(train_loader)
            trainer.sync()
            train_end = perf_counter()
            metrics = trainer.evaluate(rollout_loader)
            trainer.sync()
            rollout_end = perf_counter()
            tag = ("epoch" if trainer.update_parameters else "frozen_epoch") + f"_{trainer.epoch:03d}"
            if trainer.epoch % int(config.visualization.every_n_epochs) == 0:
                trainer.visualize(samples, tag)
            trainer.sync()
            row = dict(epoch=trainer.epoch, train_velocity_loss=loss,
                       **{"train_" + key: value for key, value in metrics.items()},
                       rollout_split="train", train_seconds=train_end-start,
                       train_rollout_seconds=rollout_end-train_end,
                       visualization_seconds=perf_counter()-rollout_end)
            history.append(row)
            trainer.logger.info("Forecast %s", row)
            trainer.save(tag + ".pt", split)
            # Frozen inspection must not replace the last trained checkpoint.
            if trainer.update_parameters:
                trainer.save("last.pt", split)
        name = "history.json" if trainer.update_parameters else "frozen_history.json"
        (result_dir / name).write_text(json.dumps(history, indent=2))
        return history
    finally:
        logger = logging.getLogger("acrobot_frames")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
