from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def resolve_nse_files(data_path: str | Path) -> list[Path]:
    """Resolve one NSE tensor file or all sorted tensor files in a directory."""
    path = Path(data_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix in {".pt", ".npy"})
        if len({item.stem for item in files}) != len(files):
            raise ValueError("NSE directory contains both .pt and .npy copies of the same shard. Use one format only.")
        if files:
            return files
        raise FileNotFoundError(f"No .pt/.npy files found in NSE data directory: {path}")
    raise FileNotFoundError(f"NSE dataset path does not exist: {path}")


def split_nse_files(
    files: list[Path],
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    """Split whole trajectory files so windows from one file cannot leak across splits."""
    if abs(train_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + test_ratio must equal 1.0.")
    if len(files) < 2:
        raise ValueError("NSE train/test splitting requires at least two .pt files.")

    train_count = int(len(files) * train_ratio)
    train_count = min(max(train_count, 1), len(files) - 1)
    return files[:train_count], files[train_count:]


def load_nse_tensor(path: str | Path) -> torch.Tensor:
    """Load the `[trajectory, time, height, width]` tensor used by forecasting_new."""
    if Path(path).suffix == ".npy":
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 4:
            raise ValueError("NSE .npy must be float32 with shape [N,T,H,W].")
        return torch.from_numpy(array)
    payload: Any = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, (tuple, list)):
        payload = payload[0]
    elif isinstance(payload, dict):
        for key in ("data", "trajectories", "tensor", "x"):
            if key in payload:
                payload = payload[key]
                break

    if not torch.is_tensor(payload) or payload.ndim != 4:
        shape = getattr(payload, "shape", None)
        raise ValueError(
            f"Expected a 4D NSE tensor [N, T, H, W] in {path}, got {type(payload)} {shape}."
        )
    return payload.float()


def compute_nse_normalization(files: list[Path]) -> float:
    """Mean per-frame RMS, matching forecasting_new's normalization definition."""
    total = 0.0
    count = 0
    for path in files:
        data = load_nse_tensor(path)
        for frames in data.reshape(-1, *data.shape[-2:]).split(32):
            frame_rms = frames.square().mean(dim=(-2, -1)).sqrt()
            total += float(frame_rms.double().sum())
            count += frame_rms.numel()
    if count == 0:
        raise ValueError("Cannot compute normalization from an empty NSE dataset.")
    value = total / count
    if value <= 0:
        raise ValueError(f"NSE normalization must be positive, got {value}.")
    return value


def compute_nse_center(files: list[Path], normalization: float) -> float:
    total = 0.0
    count = 0
    for path in files:
        data = load_nse_tensor(path)
        for frames in data.reshape(-1, *data.shape[-2:]).split(32):
            total += float(frames.double().sum()) / normalization
            count += frames.numel()
    return total / max(count, 1)


def resize_nse_input(frame: torch.Tensor, lo_size: int, hi_size: int) -> torch.Tensor:
    """Downsample to the observation grid and nearest-neighbor upsample for conditioning."""
    frame = frame.reshape(-1, 1, frame.shape[-2], frame.shape[-1])
    low = F.interpolate(frame, size=(lo_size, lo_size), mode="bilinear", align_corners=False)
    return F.interpolate(low, size=(hi_size, hi_size), mode="nearest")


def resize_nse_target(frame: torch.Tensor, hi_size: int) -> torch.Tensor:
    frame = frame.reshape(-1, 1, frame.shape[-2], frame.shape[-1])
    return F.interpolate(frame, size=(hi_size, hi_size), mode="bilinear", align_corners=False)


class NSEForecastDataset(Dataset):
    """Consecutive `(x_t, x_t+lag, x_t+2lag)` Navier-Stokes forecast windows."""

    def __init__(
        self,
        data_files: list[str | Path],
        lo_size: int,
        hi_size: int,
        normalization: float,
        time_lag: int = 2,
        center: float = 0.0,
        center_data: bool = False,
        subsampling_ratio: float = 1.0,
        time_delta: float = 1.0,
        time_origin: float = 0.0,
    ) -> None:
        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}.")
        if not 0 < subsampling_ratio <= 1:
            raise ValueError("subsampling_ratio must be in (0, 1].")

        self.lo_size = int(lo_size)
        self.hi_size = int(hi_size)
        self.normalization = float(normalization)
        if not np.isfinite(self.normalization) or self.normalization <= 0:
            raise ValueError("NSE normalization must be finite and positive.")
        self.time_lag = int(time_lag)
        if time_delta <= 0:
            raise ValueError("time_delta must be positive.")
        self.time_delta = float(time_delta)
        self.time_origin = float(time_origin)
        self.center = float(center)
        self.center_data = bool(center_data)
        self.trajectories: list[torch.Tensor] = []
        self.samples: list[tuple[int, int]] = []

        for path in data_files:
            data = load_nse_tensor(path)
            for trajectory in data:
                trajectory_index = len(self.trajectories)
                self.trajectories.append(trajectory.contiguous())
                for t0 in range(max(0, trajectory.shape[0] - 2 * self.time_lag)):
                    self.samples.append((trajectory_index, t0))

        keep = int(len(self.samples) * subsampling_ratio)
        self.samples = self.samples[:keep]
        if not self.samples:
            raise ValueError("NSE dataset contains no valid three-frame forecast windows.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory_index, t0 = self.samples[index]
        trajectory = self.trajectories[trajectory_index]
        x0 = trajectory[t0] / self.normalization
        x1 = trajectory[t0 + self.time_lag] / self.normalization
        x2 = trajectory[t0 + 2 * self.time_lag] / self.normalization

        x0 = resize_nse_input(x0, self.lo_size, self.hi_size)[0]
        x1 = resize_nse_input(x1, self.lo_size, self.hi_size)[0]
        x2 = resize_nse_target(x2, self.hi_size)[0]
        if self.center_data:
            x0 = x0 - self.center
            x1 = x1 - self.center
            x2 = x2 - self.center
        # One forecast step spans time_lag raw frames and time_delta time units.
        time = self.time_origin + (t0 + self.time_lag) * self.time_delta / self.time_lag
        return x0, x1, x2, torch.tensor(time, dtype=torch.float32)

    def get_trajectory(self, index: int) -> torch.Tensor:
        trajectory = self.trajectories[index] / self.normalization
        if self.center_data:
            trajectory = trajectory - self.center
        return trajectory
