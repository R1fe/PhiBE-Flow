"""Acrobot frame readers adapted from cv/dataset.py and GPE's Acrobot datasets.

Expected layout: root/traj_000/frame_000.png, root/traj_001/frame_000.png, ...
Relative roots are resolved against the project root, not the working directory.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils.config import resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _natural_key(path: Path):
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", path.name))


def _frame_paths(directory: Path) -> list[Path]:
    return sorted((path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                  key=_natural_key)


def list_acrobot_frame_trajectories(root_dir: str | Path) -> list[str]:
    root = resolve_path(PROJECT_ROOT, root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Acrobot image directory does not exist: {root}")
    names = [path.name for path in sorted(root.iterdir(), key=_natural_key)
             if path.is_dir() and _frame_paths(path)]
    if not names:
        raise ValueError(f"No trajectory folders containing images found in {root}.")
    return names


def split_acrobot_frame_trajectories(
    root_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split whole trajectories before sampling frames/windows to avoid leakage."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    names = list_acrobot_frame_trajectories(root_dir)
    count = int(len(names) * train_ratio)
    if not 0 < count < len(names):
        raise ValueError("The split must leave at least one train and one test trajectory.")
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [names[index] for index in order]
    return shuffled[:count], shuffled[count:]


def load_acrobot_frame(path: str | Path, image_size: int = 32,
                       normalize: bool = True) -> torch.Tensor:
    """GPE preprocessing: grayscale, short-side resize, center crop, [-1, 1]."""
    if image_size < 1:
        raise ValueError("image_size must be positive.")
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        if width <= height:
            size = (image_size, int(image_size * height / width))
        else:
            size = (int(image_size * width / height), image_size)
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
        left = int(round((size[0] - image_size) / 2.0))
        top = int(round((size[1] - image_size) / 2.0))
        image = image.crop((left, top, left + image_size, top + image_size))
        frame = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0) / 255.0
    return frame * 2.0 - 1.0 if normalize else frame


class AcrobotFramesDataset(Dataset):
    """Lazy frame, adjacent-pair, window or whole-trajectory sampling.

    mode='image': (frame [1,H,W], scalar time).
    mode='pair': (frame [1,H,W], future_frame [1,H,W], scalar time).
    mode='window': (context [L,1,H,W], future [P,1,H,W], scalar time).
    mode='trajectory': (frames [T,1,H,W], times [T]).

    Window time belongs to the final context frame. Time units are determined
    by time_delta (per original frame); the default uses frame indices.
    Whole trajectories can vary in length: use batch_size=1 for that mode.
    """

    def __init__(
        self,
        root_dir: str | Path = "data/acrobot_frames",
        *,
        trajectory_names: list[str] | None = None,
        mode: str = "window",
        image_size: int = 32,
        normalize: bool = True,
        seq_length: int = 1,
        prediction_horizon: int = 1,
        time_lag: int = 1,
        stride: int = 1,
        time_delta: float = 1.0,
        time_origin: float = 0.0,
        max_frames: int | None = None,
        preload: bool = False,
    ):
        if mode not in {"image", "pair", "window", "trajectory"}:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        if min(image_size, seq_length, prediction_horizon, time_lag, stride) < 1:
            raise ValueError("Image size, window lengths, time_lag and stride must be positive.")
        if not np.isfinite(time_delta) or time_delta <= 0 or not np.isfinite(time_origin):
            raise ValueError("Time origin must be finite and time_delta finite and positive.")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive or None.")
        self.root = resolve_path(PROJECT_ROOT, root_dir)
        names = list_acrobot_frame_trajectories(self.root)
        if trajectory_names is not None:
            if not trajectory_names or len(set(trajectory_names)) != len(trajectory_names):
                raise ValueError("trajectory_names must be nonempty and contain no duplicates.")
            missing = set(trajectory_names) - set(names)
            if missing:
                raise ValueError(f"Unknown trajectory folders: {sorted(missing)}")
            names = list(trajectory_names)
        self.trajectory_names = names
        self.frames = [_frame_paths(self.root / name)[:max_frames] for name in names]
        self.mode = mode
        self.image_size = image_size
        self.normalize = normalize
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.time_lag = time_lag
        self.time_delta = time_delta
        self.time_origin = time_origin
        self.indices: list[tuple[int, int]] = []
        window_size = seq_length + prediction_horizon if mode == "window" else 2
        for trajectory_index, paths in enumerate(self.frames):
            if mode == "trajectory":
                self.indices.append((trajectory_index, 0))
            else:
                stop = len(paths) if mode == "image" else len(paths) - (window_size - 1) * time_lag
                self.indices.extend((trajectory_index, start) for start in range(0, max(stop, 0), stride))
        if not self.indices:
            raise ValueError("No valid samples; check trajectory lengths and window settings.")
        self.cache = {}
        if preload:
            self.cache = {path: load_acrobot_frame(path, image_size, normalize)
                          for paths in self.frames for path in paths}

    def __len__(self):
        return len(self.indices)

    def _load(self, path: Path) -> torch.Tensor:
        if path in self.cache:
            return self.cache[path].clone()
        return load_acrobot_frame(path, self.image_size, self.normalize)

    def __getitem__(self, index: int):
        trajectory_index, start = self.indices[index]
        paths = self.frames[trajectory_index]
        time = self.time_origin + start * self.time_delta
        if self.mode == "image":
            return self._load(paths[start]), torch.tensor(time, dtype=torch.float32)
        if self.mode == "trajectory":
            frames = torch.stack([self._load(path) for path in paths])
            times = self.time_origin + torch.arange(len(paths), dtype=torch.float32) * self.time_delta
            return frames, times
        if self.mode == "pair":
            return (self._load(paths[start]), self._load(paths[start + self.time_lag]),
                    torch.tensor(time, dtype=torch.float32))
        indices = [start + offset * self.time_lag
                   for offset in range(self.seq_length + self.prediction_horizon)]
        frames = torch.stack([self._load(paths[frame_index]) for frame_index in indices])
        time = self.time_origin + indices[self.seq_length - 1] * self.time_delta
        return frames[:self.seq_length], frames[self.seq_length:], torch.tensor(time, dtype=torch.float32)


def build_acrobot_frame_datasets(config, project_root: str | Path = PROJECT_ROOT):
    """Build train/test datasets from configs/acrobot_frames.yaml."""
    dataset = config.dataset
    if abs(float(dataset.train_ratio) + float(dataset.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + test_ratio must equal 1.")
    root = resolve_path(project_root, dataset.dataset_path)
    train_names, test_names = split_acrobot_frame_trajectories(
        root, float(dataset.train_ratio), int(dataset.seed)
    )
    options = {key: dataset[key] for key in (
        "mode", "image_size", "normalize", "seq_length", "prediction_horizon",
        "time_lag", "stride", "time_delta", "time_origin", "max_frames", "preload"
    ) if key in dataset}
    return (AcrobotFramesDataset(root, trajectory_names=train_names, **options),
            AcrobotFramesDataset(root, trajectory_names=test_names, **options))
