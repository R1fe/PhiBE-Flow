"""Shared KTH setup for the unified train/eval entrypoints."""

import json
import logging
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.datasets.kth import build_kth_datasets
from src.models.kth_predictor import KTHVelocityPredictor
from src.models.vqvae import VQVAE
from src.trainers.kth_trainer import KTHTrainer
from src.utils.config import resolve_path
from src.utils.logger import build_logger


def fixed_kth_samples(dataset, count, seed):
    if count < 0:
        raise ValueError("Fixed sample count cannot be negative.")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed+1))[:count].tolist()
    samples = []
    for sample_id, index in enumerate(indices):
        start = dataset.clip_start(index)
        frames, times = dataset.get_clip(index, start)
        record = dataset.records[index]
        samples.append(dict(sample_id=sample_id, dataset_index=index, shard=str(record.shard),
                            video_key=record.key, start_frame=start, frames=frames, times=times))
    return samples


def run_kth(config, args, device, project_root, evaluation=False):
    try:
        return _run_kth(config, args, device, project_root, evaluation)
    finally:
        logger = logging.getLogger("kth_eval" if evaluation else "kth_train")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def _run_kth(config, args, device, project_root, evaluation=False):
    horizons = [int(h) for h in config.evaluation.horizons]
    if any(h < 1 or h > int(config.dataset.prediction_frames) for h in horizons):
        raise ValueError("evaluation.horizons must lie within dataset.prediction_frames.")
    if bool(config.training.load_weights_only) and not config.training.checkpoint_path and not evaluation:
        raise ValueError("Frozen KTH mode requires training.checkpoint_path.")
    if int(config.training.max_steps) < 1:
        raise ValueError("training.max_steps must be positive.")
    # Fail early rather than silently using a randomly initialized image codec.
    codec_path = resolve_path(project_root, config.vqvae.checkpoint_path)
    if not codec_path.is_file():
        raise FileNotFoundError(f"Place the pretrained KTH f8_small VQ-VAE at {codec_path}.")
    train, test = build_kth_datasets(config, project_root)
    kwargs = dict(batch_size=int(config.dataset.batch_size), num_workers=int(config.dataset.num_workers),
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, **kwargs)
    test_loader = DataLoader(test, shuffle=False, **kwargs)
    rollout_loader = DataLoader(train, shuffle=False, **kwargs)
    model = KTHVelocityPredictor(**dict(config.model)).to(device)
    codec = VQVAE(codec_path, int(config.vqvae.chunk_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.lr),
                                  weight_decay=float(config.training.weight_decay))
    logger = build_logger("kth_eval" if evaluation else "kth_train",
                          resolve_path(project_root, config.paths.log_dir) / ("eval.log" if evaluation else "train.log"))
    trainer = KTHTrainer(model, codec, optimizer, device, config, project_root, logger,
                         data_identity=train.data_identity)
    manifest = train.split_manifest
    sample_dataset = test if evaluation else train
    samples = fixed_kth_samples(sample_dataset, int(config.visualization.num_fixed_samples),
                                int(config.training.seed)) if config.visualization.enabled else []
    fixed_manifest = [{k: v for k, v in sample.items() if k not in ("frames", "times")} for sample in samples]
    def save_manifests():
        prefix = "eval_" if evaluation else ""
        (trainer.result_dir / (prefix + "split_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        sample_file = "eval_fixed_test_samples.json" if evaluation else "fixed_train_samples.json"
        (trainer.result_dir / sample_file).write_text(json.dumps(fixed_manifest, indent=2), encoding="utf-8")
    logger.info("KTH train=%s test=%s videos | dt=%s | parameters=%s | frozen=%s",
                len(train), len(test), trainer.dt, sum(p.numel() for p in model.parameters()),
                not trainer.update_parameters)
    if evaluation:
        checkpoint = (resolve_path(project_root, args.checkpoint) if args.checkpoint else
                      trainer.checkpoint_dir / str(config.evaluation.checkpoint))
        trainer.load(checkpoint, resume=False)
        save_manifests()
        detector = None
        if bool(config.evaluation.fvd):
            from src.utils.fvd import load_fvd_detector, SERVER_I3D_SHA256

            i3d_path = resolve_path(project_root, config.evaluation.i3d_checkpoint)
            detector, detector_sha256 = load_fvd_detector(
                i3d_path, device, config.evaluation.get("i3d_sha256", SERVER_I3D_SHA256))
        trainer._sync()
        start = perf_counter()
        use_ema = bool(config.evaluation.use_ema)
        metrics = trainer.evaluate_loader(test_loader, use_ema=use_ema, detector=detector)
        trainer._sync()
        metrics["evaluation_seconds"] = perf_counter()-start
        metrics["use_ema"] = use_ema
        metrics["data_identity"] = trainer.data_identity
        if detector is not None:
            metrics["fvd_protocol"] = {
                "backend": "torchscript_i3d",
                "i3d_sha256": detector_sha256,
                "condition_frames": trainer.condition_frames,
                "prediction_frames": int(config.dataset.prediction_frames),
                "reference": "original_pixels",
                "includes_conditioning": True,
                "resize": "bilinear_224_align_corners_false",
            }
        (trainer.result_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        trainer.visualize(samples, "evaluation_ema" if use_ema else "evaluation", use_ema=use_ema)
        logger.info("KTH test metrics: %s", metrics)
        return metrics
    if config.training.checkpoint_path:
        trainer.load(resolve_path(project_root, config.training.checkpoint_path))
    save_manifests()
    default_epochs = int(config.training.epochs) if trainer.update_parameters else 1
    epochs = args.epochs if args.epochs is not None else default_epochs
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    return trainer.fit(train_loader, rollout_loader, epochs, samples)
