from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training entrypoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/acrobot_angles.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    from src.utils.cli import add_runtime_arguments
    add_runtime_arguments(parser)
    return parser.parse_args()


def build_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def split_trajectories(
    trajectories: list[dict],
    train_ratio: float,
    test_ratio: float,
    seed: int,
):
    import torch

    total_ratio = train_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + test_ratio must equal 1.0.")
    if len(trajectories) < 2:
        raise ValueError("At least 2 trajectories are required to build train/test splits.")

    indexed = list(enumerate(trajectories))
    permutation = torch.randperm(len(indexed), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [indexed[index] for index in permutation]

    train_count = int(len(shuffled) * train_ratio)
    test_count = len(shuffled) - train_count

    if train_count <= 0 or test_count <= 0:
        raise ValueError("The configured split leaves train or test empty.")

    train_split = shuffled[:train_count]
    test_split = shuffled[train_count:]
    return train_split, test_split


def build_fixed_visualization_samples(
    indexed_trajectories,
    *,
    seq_length: int,
    transform_angles: bool,
    num_samples: int,
    rollout_steps: int | None,
    seed: int,
    trajectory_indices: list[int] | None = None,
):
    import torch

    from src.datasets.acrobot_angles import trajectory_to_features

    if not indexed_trajectories or num_samples <= 0:
        return []

    if trajectory_indices:
        trajectory_lookup = dict(indexed_trajectories)
        missing_indices = [index for index in trajectory_indices if index not in trajectory_lookup]
        if missing_indices:
            raise ValueError(
                "Requested visualization trajectories are not in this split: "
                f"{missing_indices}"
            )
        selected_trajectories = [
            (index, trajectory_lookup[index]) for index in trajectory_indices
        ]
    else:
        sample_count = min(num_samples, len(indexed_trajectories))
        permutation = torch.randperm(
            len(indexed_trajectories),
            generator=torch.Generator().manual_seed(seed + 1),
        ).tolist()
        selected_trajectories = [
            indexed_trajectories[position] for position in permutation[:sample_count]
        ]

    samples = []
    for sample_id, (trajectory_index, trajectory) in enumerate(selected_trajectories):
        features = trajectory_to_features(
            trajectory,
            transform_angles=transform_angles,
        )
        if len(features) <= seq_length:
            continue
        available_rollout = len(features) - seq_length
        sample_rollout_steps = available_rollout if rollout_steps is None else min(rollout_steps, available_rollout)
        if sample_rollout_steps <= 0:
            continue

        samples.append(
            {
                "sample_id": sample_id,
                "trajectory_index": trajectory_index,
                "seq_length": seq_length,
                "rollout_steps": sample_rollout_steps,
                "features": features,
                "start_time": float(trajectory["t"][seq_length - 1]),
            }
        )

    return samples


def build_fixed_nse_samples(test_dataset, num_samples: int, seed: int):
    import torch

    if num_samples <= 0 or not test_dataset.trajectories:
        return []
    count = min(num_samples, len(test_dataset.trajectories))
    indices = torch.randperm(
        len(test_dataset.trajectories),
        generator=torch.Generator().manual_seed(seed + 1),
    )[:count].tolist()
    return [
        {
            "sample_id": sample_id,
            "trajectory_index": trajectory_index,
            "trajectory": test_dataset.get_trajectory(trajectory_index).clone(),
        }
        for sample_id, trajectory_index in enumerate(indices)
    ]


def train_nse(config, args, device) -> None:
    import torch
    from torch.utils.data import DataLoader

    from src.datasets.nse import (
        NSEForecastDataset,
        compute_nse_center,
        compute_nse_normalization,
        resolve_nse_files,
        split_nse_files,
    )
    from src.models.nse_predictor import NSEDriftModel
    from src.trainers.nse_trainer import NSETrainer
    from src.utils.checkpoint import load_checkpoint
    from src.utils.config import resolve_path
    from src.utils.logger import build_logger

    data_path = resolve_path(PROJECT_ROOT, config.dataset.dataset_path)
    files = resolve_nse_files(data_path)
    train_files, test_files = split_nse_files(
        files,
        train_ratio=float(config.dataset.train_ratio),
        test_ratio=float(config.dataset.test_ratio),
    )

    configured_normalization = config.dataset.get("normalization", "auto")
    normalization = (
        compute_nse_normalization(train_files)
        if str(configured_normalization).lower() == "auto"
        else float(configured_normalization)
    )
    center_data = bool(config.dataset.center_data)
    center = compute_nse_center(train_files, normalization) if center_data else 0.0
    dataset_kwargs = {
        "lo_size": int(config.dataset.lo_size),
        "hi_size": int(config.dataset.hi_size),
        "normalization": normalization,
        "time_lag": int(config.dataset.time_lag),
        "time_delta": float(config.training.time_delta),
        "time_origin": float(config.dataset.get("time_origin", 0.0)),
        "center": center,
        "center_data": center_data,
        "subsampling_ratio": float(config.dataset.subsampling_ratio),
    }
    train_dataset = NSEForecastDataset(train_files, **dataset_kwargs)
    test_dataset = NSEForecastDataset(test_files, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.dataset.batch_size),
        shuffle=True,
        num_workers=int(config.dataset.num_workers),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config.dataset.batch_size),
        shuffle=False,
        num_workers=int(config.dataset.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = NSEDriftModel(
        in_channels=int(config.model.in_channels),
        out_channels=int(config.model.out_channels),
        dim=int(config.model.channels),
        dim_mults=tuple(config.model.dim_mults),
        resnet_block_groups=int(config.model.resnet_block_groups),
        learned_sinusoidal_cond=bool(config.model.learned_sinusoidal_cond),
        random_fourier_features=bool(config.model.random_fourier_features),
        learned_sinusoidal_dim=int(config.model.learned_sinusoidal_dim),
        attention_dim_head=int(config.model.attention_dim_head),
        attention_heads=int(config.model.attention_heads),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )

    log_dir = resolve_path(PROJECT_ROOT, config.paths.log_dir)
    logger = build_logger("train_nse", log_dir / "train.log")
    checkpoint_path = config.training.get("checkpoint_path")
    checkpoint_path = (
        None if checkpoint_path in (None, "", "null") else resolve_path(PROJECT_ROOT, checkpoint_path)
    )
    load_weights_only = bool(config.training.load_weights_only)
    if load_weights_only and checkpoint_path is None:
        raise ValueError("When training.load_weights_only is true, checkpoint_path must be set.")
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
        logger.info("Loaded NSE checkpoint %s at step %s", checkpoint_path, checkpoint.get("step"))

    fixed_samples = []
    if bool(config.visualization.enabled):
        fixed_samples = build_fixed_nse_samples(
            test_dataset,
            int(config.visualization.num_fixed_test_samples),
            int(config.training.seed),
        )
        manifest = [
            {
                "sample_id": sample["sample_id"],
                "trajectory_index": sample["trajectory_index"],
            }
            for sample in fixed_samples
        ]
        manifest_path = resolve_path(PROJECT_ROOT, config.paths.result_dir) / "fixed_test_samples.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trainer = NSETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=resolve_path(PROJECT_ROOT, config.paths.checkpoint_dir),
        result_dir=resolve_path(PROJECT_ROOT, config.paths.result_dir),
        figure_dir=resolve_path(PROJECT_ROOT, config.paths.figure_dir),
        logger=logger,
        time_delta=float(config.training.time_delta),
        max_grad_norm=float(config.training.max_grad_norm),
        update_parameters=not load_weights_only,
        lo_size=int(config.dataset.lo_size),
        hi_size=int(config.dataset.hi_size),
        time_lag=int(config.dataset.time_lag),
        time_origin=float(config.dataset.get("time_origin", 0.0)),
    )
    epochs = args.epochs if args.epochs is not None else int(config.training.epochs)
    logger.info(
        "Starting NSE training on %s | train_files=%s | test_files=%s | "
        "train_windows=%s | test_windows=%s | normalization=%.8f | update_parameters=%s",
        device,
        len(train_files),
        len(test_files),
        len(train_dataset),
        len(test_dataset),
        normalization,
        not load_weights_only,
    )
    trainer.fit(
        train_loader,
        test_loader,
        epochs,
        fixed_test_trajectories=fixed_samples,
        visualize_every=int(config.visualization.every_n_epochs),
        rollout_steps=int(config.visualization.rollout_steps),
        rollout_plot_frames=int(config.visualization.rollout_plot_frames),
    )


def main() -> None:
    args = parse_args()

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.datasets.acrobot_angles import AcrobotAnglesDataset, load_acrobot_pickle
    from src.models.predictor import build_predictor
    from src.trainers.state_trainer import StateTrainer
    from src.utils.checkpoint import load_checkpoint
    from src.utils.config import load_config, resolve_path
    from src.utils.logger import build_logger
    from src.utils.seed import set_seed

    config_path = resolve_path(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    from src.utils.cli import apply_runtime_arguments
    apply_runtime_arguments(config, args)

    set_seed(int(config.training.seed))
    device = build_device(config.training.device)

    if config.dataset.name == "kth":
        from src.trainers.kth_pipeline import run_kth

        run_kth(config, args, device, PROJECT_ROOT)
        return
    if config.dataset.name == "nse":
        train_nse(config, args, device)
        return
    if config.dataset.name != "acrobot_angles":
        raise NotImplementedError(f"Unsupported dataset: {config.dataset.name}")

    dataset_path = resolve_path(PROJECT_ROOT, config.dataset.dataset_path)
    trajectories = load_acrobot_pickle(dataset_path)
    train_trajectories, test_trajectories = split_trajectories(
        trajectories=trajectories,
        train_ratio=float(config.dataset.train_ratio),
        test_ratio=float(config.dataset.test_ratio),
        seed=int(config.training.seed),
    )

    train_dataset = AcrobotAnglesDataset(
        [trajectory for _, trajectory in train_trajectories],
        seq_length=int(config.dataset.seq_length),
        transform_angles=bool(config.dataset.transform_angles),
        prediction_horizon=int(config.dataset.get("prediction_horizon", 1)),
    )
    test_dataset = AcrobotAnglesDataset(
        [trajectory for _, trajectory in test_trajectories],
        seq_length=int(config.dataset.seq_length),
        transform_angles=bool(config.dataset.transform_angles),
        prediction_horizon=int(config.dataset.get("prediction_horizon", 1)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.dataset.batch_size),
        shuffle=True,
        num_workers=int(config.dataset.num_workers),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config.dataset.batch_size),
        shuffle=False,
        num_workers=int(config.dataset.num_workers),
    )

    model = build_predictor(
        seq_length=int(config.dataset.seq_length),
        feature_dim=int(train_dataset.feature_dim),
        hidden_dims=list(config.model.hidden_dims),
        num_res_blocks=int(config.model.num_res_blocks),
        res_block_dim=int(config.model.res_block_dim),
        activation=str(config.model.activation),
        ffn_expansion=int(config.model.get("ffn_expansion", 1)),
        dropout=float(config.model.get("dropout", 0.0)),
        use_layer_norm=bool(config.model.get("use_layer_norm", False)),
        layer_scale_init=float(config.model.get("layer_scale_init", 0.0)),
        periodic_angle_indices=list(config.model.get("periodic_angle_indices", [])),
        predict_acceleration_only=bool(config.model.get("predict_acceleration_only", False)),
        physics_backbone=bool(config.model.get("physics_backbone", False)),
        physics_residual_scale=float(config.model.get("physics_residual_scale", 1.0)),
        kinematic_time_delta=float(config.model.get("kinematic_time_delta", 0.0)),
        zero_init_output=bool(config.model.get("zero_init_output", False)),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )

    log_dir = resolve_path(PROJECT_ROOT, config.paths.log_dir)
    logger = build_logger("train", log_dir / "train.log")
    checkpoint_path = config.training.get("checkpoint_path")
    checkpoint_path = (
        None if checkpoint_path in (None, "", "null") else resolve_path(PROJECT_ROOT, checkpoint_path)
    )
    load_weights_only = bool(config.training.load_weights_only)
    if load_weights_only and checkpoint_path is None:
        raise ValueError("When training.load_weights_only is true, training.checkpoint_path must be set.")
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(config.training.lr)
        logger.info("Loaded checkpoint from %s", checkpoint_path)

    feature_scales = None
    if bool(config.training.get("normalize_rollout_loss", False)):
        import numpy as np

        from src.datasets.acrobot_angles import trajectory_to_features

        train_features = np.concatenate(
            [
                trajectory_to_features(
                    trajectory,
                    transform_angles=bool(config.dataset.transform_angles),
                )
                for _, trajectory in train_trajectories
            ],
            axis=0,
        )
        feature_scales = np.maximum(train_features.std(axis=0), 1.0e-6).tolist()
        logger.info("Rollout feature scales: %s", feature_scales)

    fixed_visualization_samples = []
    if bool(config.visualization.enabled):
        fixed_visualization_samples = build_fixed_visualization_samples(
            test_trajectories,
            seq_length=int(config.dataset.seq_length),
            transform_angles=bool(config.dataset.transform_angles),
            num_samples=int(config.visualization.num_fixed_test_samples),
            rollout_steps=config.visualization.rollout_steps,
            seed=int(config.training.seed),
            trajectory_indices=[
                int(index)
                for index in config.visualization.get("trajectory_indices", [])
            ],
        )

        sample_manifest = [
            {
                "sample_id": sample["sample_id"],
                "trajectory_index": sample["trajectory_index"],
                "seq_length": sample["seq_length"],
                "rollout_steps": sample["rollout_steps"],
            }
            for sample in fixed_visualization_samples
        ]
        sample_manifest_path = resolve_path(PROJECT_ROOT, config.paths.result_dir) / "fixed_test_samples.json"
        sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        sample_manifest_path.write_text(json.dumps(sample_manifest, indent=2), encoding="utf-8")
        logger.info("Fixed test visualization samples: %s", sample_manifest)

    trainer = StateTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        time_delta=float(config.training.time_delta),
        checkpoint_dir=resolve_path(PROJECT_ROOT, config.paths.checkpoint_dir),
        result_dir=resolve_path(PROJECT_ROOT, config.paths.result_dir),
        log_dir=log_dir,
        logger=logger,
        dataset_name=str(config.dataset.name),
        update_parameters=not load_weights_only,
        mixed_precision=bool(config.training.get("mixed_precision", False)),
        rollout_loss_weight=float(config.training.get("rollout_loss_weight", 0.0)),
        feature_scales=feature_scales,
        integration_method=str(config.training.get("integration_method", "euler")),
    )

    epochs = args.epochs if args.epochs is not None else int(config.training.epochs)
    logger.info(
        "Starting training on %s | train=%s windows | test=%s windows | update_parameters=%s",
        device,
        len(train_dataset),
        len(test_dataset),
        not load_weights_only,
    )
    trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        log_every=int(config.training.log_every),
        visualization_samples=fixed_visualization_samples,
        figure_dir=resolve_path(PROJECT_ROOT, config.paths.figure_dir),
        transform_angles=bool(config.dataset.transform_angles),
        visualize_every=int(config.visualization.every_n_epochs),
        checkpoint_every=int(config.training.get("checkpoint_every", 1)),
        selection_metric=str(config.training.get("selection_metric", "drift_mse")),
    )


if __name__ == "__main__":
    main()
