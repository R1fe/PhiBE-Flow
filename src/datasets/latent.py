from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def load_latent_trajectories(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Latent dataset must map trajectory names to arrays.")

    trajectories: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 6:
            raise ValueError(f"Trajectory {key!r} has shape {array.shape}; expected [T, 6].")
        trajectories[str(key)] = array
    return trajectories


def split_latent_trajectories(
    trajectories: dict[str, np.ndarray],
    train_ratio: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    items = list(trajectories.items())
    split_index = int(len(items) * train_ratio)
    if split_index <= 0 or split_index >= len(items):
        raise ValueError("train_ratio leaves the train or validation split empty.")
    return dict(items[:split_index]), dict(items[split_index:])


def compute_latent_statistics(
    trajectories: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.concatenate(list(trajectories.values()), axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0), 1.0e-6).astype(np.float32)
    accelerations = np.concatenate(
        [np.diff(trajectory[:, 3:], axis=0) for trajectory in trajectories.values()],
        axis=0,
    )
    acceleration_scale = np.maximum(accelerations.std(axis=0), 1.0e-6).astype(np.float32)
    return mean, std, acceleration_scale


class LatentRolloutDataset(Dataset):
    def __init__(
        self,
        trajectories: dict[str, np.ndarray],
        sequence_length: int,
        prediction_horizon: int,
        stride: int = 1,
    ):
        self.trajectories = list(trajectories.values())
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.indices: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(self.trajectories):
            stop = len(trajectory) - sequence_length - prediction_horizon + 1
            self.indices.extend(
                (trajectory_index, start) for start in range(0, max(stop, 0), stride)
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory_index, start = self.indices[index]
        trajectory = self.trajectories[trajectory_index]
        split = start + self.sequence_length
        sequence = trajectory[start:split]
        future = trajectory[split : split + self.prediction_horizon]
        return torch.from_numpy(sequence), torch.from_numpy(future), torch.tensor(split - 1, dtype=torch.float32)
