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
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint.")
    from src.utils.cli import add_runtime_arguments
    add_runtime_arguments(parser)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/acrobot_angles.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path override.",
    )
    return parser.parse_args()


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


def evaluate_nse(config, args, device) -> None:
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

    files = resolve_nse_files(resolve_path(PROJECT_ROOT, config.dataset.dataset_path))
    train_files, test_files = split_nse_files(
        files,
        float(config.dataset.train_ratio),
        float(config.dataset.test_ratio),
    )
    configured_normalization = config.dataset.get("normalization", "auto")
    normalization = (
        compute_nse_normalization(train_files)
        if str(configured_normalization).lower() == "auto"
        else float(configured_normalization)
    )
    center = (
        compute_nse_center(train_files, normalization)
        if bool(config.dataset.center_data)
        else 0.0
    )
    test_dataset = NSEForecastDataset(
        test_files,
        lo_size=int(config.dataset.lo_size),
        hi_size=int(config.dataset.hi_size),
        normalization=normalization,
        time_lag=int(config.dataset.time_lag),
        time_delta=float(config.training.time_delta),
        time_origin=float(config.dataset.get("time_origin", 0.0)),
        center=center,
        center_data=bool(config.dataset.center_data),
        subsampling_ratio=float(config.dataset.subsampling_ratio),
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
    logger = build_logger(
        "eval_nse", resolve_path(PROJECT_ROOT, config.paths.log_dir) / "eval.log"
    )
    trainer = NSETrainer(
        model,
        optimizer,
        device,
        resolve_path(PROJECT_ROOT, config.paths.checkpoint_dir),
        resolve_path(PROJECT_ROOT, config.paths.result_dir),
        resolve_path(PROJECT_ROOT, config.paths.figure_dir),
        logger,
        time_delta=float(config.training.time_delta),
        lo_size=int(config.dataset.lo_size),
        hi_size=int(config.dataset.hi_size),
        time_lag=int(config.dataset.time_lag),
        time_origin=float(config.dataset.get("time_origin", 0.0)),
    )
    checkpoint_path = (
        resolve_path(PROJECT_ROOT, args.checkpoint)
        if args.checkpoint
        else resolve_path(PROJECT_ROOT, config.paths.checkpoint_dir)
        / str(config.evaluation.checkpoint)
    )
    load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    metrics = trainer.evaluate_loader(test_loader)
    output_path = resolve_path(PROJECT_ROOT, config.paths.result_dir) / "eval_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("NSE test metrics: %s", metrics)


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

    config = load_config(resolve_path(PROJECT_ROOT, args.config))
    from src.utils.cli import apply_runtime_arguments
    apply_runtime_arguments(config, args)

    set_seed(int(config.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.training.device == "auto" else torch.device(config.training.device)

    if config.dataset.name == "acrobot_frames":
        from src.trainers.acrobot_frame_pipeline import run_acrobot_frames

        run_acrobot_frames(config, args, device, PROJECT_ROOT, evaluation=True)
        return
    if config.dataset.name == "kth":
        from src.trainers.kth_pipeline import run_kth

        run_kth(config, args, device, PROJECT_ROOT, evaluation=True)
        return
    if config.dataset.name == "nse":
        evaluate_nse(config, args, device)
        return
    if config.dataset.name != "acrobot_angles":
        raise NotImplementedError(f"Unsupported dataset: {config.dataset.name}")

    trajectories = load_acrobot_pickle(resolve_path(PROJECT_ROOT, config.dataset.dataset_path))
    _, test_trajectories = split_trajectories(
        trajectories=trajectories,
        train_ratio=float(config.dataset.train_ratio),
        test_ratio=float(config.dataset.test_ratio),
        seed=int(config.training.seed),
    )
    dataset = AcrobotAnglesDataset(
        [trajectory for _, trajectory in test_trajectories],
        seq_length=int(config.dataset.seq_length),
        transform_angles=bool(config.dataset.transform_angles),
        prediction_horizon=int(config.dataset.get("prediction_horizon", 1)),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=int(config.dataset.batch_size),
        shuffle=False,
        num_workers=int(config.dataset.num_workers),
    )

    model = build_predictor(
        seq_length=int(config.dataset.seq_length),
        feature_dim=int(dataset.feature_dim),
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
    optimizer = optim.Adam(model.parameters(), lr=float(config.training.lr), weight_decay=float(config.training.weight_decay))

    logger = build_logger("eval", resolve_path(PROJECT_ROOT, config.paths.log_dir) / "eval.log")
    trainer = StateTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        time_delta=float(config.training.time_delta),
        checkpoint_dir=resolve_path(PROJECT_ROOT, config.paths.checkpoint_dir),
        result_dir=resolve_path(PROJECT_ROOT, config.paths.result_dir),
        log_dir=resolve_path(PROJECT_ROOT, config.paths.log_dir),
        logger=logger,
        dataset_name=str(config.dataset.name),
        integration_method=str(config.training.get("integration_method", "euler")),
    )

    checkpoint_path = (
        resolve_path(PROJECT_ROOT, args.checkpoint)
        if args.checkpoint
        else resolve_path(PROJECT_ROOT, config.paths.checkpoint_dir) / str(config.evaluation.checkpoint)
    )
    load_checkpoint(checkpoint_path, model, optimizer, map_location=device)

    metrics = trainer.evaluate_loader(test_loader)
    output_path = resolve_path(PROJECT_ROOT, config.paths.result_dir) / "eval_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Test metrics: %s", metrics)


if __name__ == "__main__":
    main()
