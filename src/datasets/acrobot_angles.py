from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset
from tqdm import tqdm
from src.physics import AcrobotSystem


def transform_state(state: np.ndarray, transform_angles: bool = False) -> np.ndarray:
    """Optionally convert angles to a sin/cos representation."""
    if not transform_angles:
        return np.asarray(state, dtype=np.float32)

    theta1, theta2, theta1_dot, theta2_dot = state
    return np.asarray(
        [
            np.cos(theta1),
            np.sin(theta1),
            np.cos(theta2),
            np.sin(theta2),
            theta1_dot,
            theta2_dot,
        ],
        dtype=np.float32,
    )


def trajectory_to_features(
    trajectory: dict,
    transform_angles: bool = False,
) -> np.ndarray:
    """Convert a raw Acrobot trajectory dict into a frame-by-frame feature array."""
    states = np.asarray(trajectory["y"], dtype=np.float32).T
    return np.asarray(
        [transform_state(state, transform_angles=transform_angles) for state in states],
        dtype=np.float32,
    )


def load_acrobot_pickle(path: str | Path) -> list[dict]:
    """Load safe numeric NPZ data or the legacy trusted-only pickle format."""
    if Path(path).suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            states, times = data["y"], data["t"]
            if states.ndim != 3 or states.shape[1] != 4 or times.ndim != 1 or states.shape[2] != len(times):
                raise ValueError("Expected Acrobot NPZ y=[N,4,T], t=[T].")
            return [{"t": times.copy(), "y": state.copy()} for state in states]
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


class AcrobotAnglesDataset(Dataset):
    """
    Sliding-window dataset for Acrobot angle trajectories.

    Each sample is `(sequence, next_state, time)`, where time belongs to the
    last conditioning state. The input sequence contains
    `seq_length` consecutive states and the target is the immediate next state.
    """

    def __init__(
        self,
        trajectories: Iterable[dict],
        seq_length: int = 1,
        transform_angles: bool = False,
        prediction_horizon: int = 1,
    ) -> None:
        if seq_length < 1:
            raise ValueError("seq_length must be at least 1.")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1.")

        self.seq_length = seq_length
        self.transform_angles = transform_angles
        self.prediction_horizon = prediction_horizon
        self.samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.rollout_trajectories = []

        for trajectory in trajectories:
            time_steps = trajectory["t"]
            features = trajectory_to_features(
                trajectory,
                transform_angles=transform_angles,
            )

            if len(time_steps) < seq_length + prediction_horizon:
                continue
            self.rollout_trajectories.append((features, float(time_steps[seq_length - 1])))

            window_count = len(time_steps) - seq_length - prediction_horizon + 1
            for start in range(window_count):
                sequence = features[start : start + seq_length]
                future = features[
                    start + seq_length : start + seq_length + prediction_horizon
                ]
                target = future[0] if prediction_horizon == 1 else future
                self.samples.append((sequence, target, float(time_steps[start + seq_length - 1])))

        if not self.samples:
            raise ValueError("No valid samples were created from the Acrobot dataset.")

        self.feature_dim = self.samples[0][0].shape[-1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, target, time = self.samples[index]
        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(time, dtype=torch.float32),
        )




class AcrobotDatasetGenerator:
    """Generator for synthetic Acrobot trajectories in the original pickle format."""

    def __init__(self, system: AcrobotSystem, duration: float = 8.0, fps: int = 30):
        self.system = system
        self.duration = duration
        self.fps = fps

    def generate_initial_state(self) -> list[float]:
        return [
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
        ]

    def generate_single_trajectory(self) -> dict:
        initial_state = self.generate_initial_state()
        t_eval = np.linspace(0, self.duration, int(self.duration * self.fps))
        solution = solve_ivp(
            self.system.dynamics,
            (0, self.duration),
            initial_state,
            t_eval=t_eval,
        )
        return {
            "initial_state": initial_state,
            "t": solution.t,
            "y": solution.y,
        }

    def generate_dataset(self, num_samples: int = 600) -> list[dict]:
        samples = []
        for _ in tqdm(range(num_samples), desc="Generating Acrobot trajectories"):
            samples.append(self.generate_single_trajectory())
        return samples


def save_generated_dataset(
    dataset: list[dict],
    output_dir: str | Path,
    filename: str = "acrobot_data.pkl",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with output_path.open("wb") as handle:
        pickle.dump(dataset, handle)
    return output_path
