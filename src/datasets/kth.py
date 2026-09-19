"""Lazy HDF5 KTH clips with video-level splits and explicit frame times."""

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class KTHVideo:
    shard: Path
    key: str
    length: int


def _frame_store(handle, key):
    value = handle[key]
    return value["frames"] if isinstance(value, h5py.Group) and "frames" in value else value


def discover_kth_videos(root):
    root = Path(root)
    shards = [root] if root.is_file() else sorted(
        set(root.rglob("*.hdf5")) | set(root.rglob("*.h5")))
    if not shards:
        raise FileNotFoundError(f"No KTH HDF5 shards found under {root}.")
    records = []
    for shard in shards:
        with h5py.File(shard, "r") as handle:
            keys = list(handle["len"].keys()) if "len" in handle else [k for k in handle if k.isdigit()]
            for key in sorted(keys, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else k)):
                store = _frame_store(handle, key)
                length = int(handle["len"][key][()]) if "len" in handle else len(store)
                if len(store) < length:
                    raise ValueError(f"Invalid video length in {shard}:{key}.")
                records.append(KTHVideo(shard.resolve(), key, length))
    if not records:
        raise ValueError("KTH shards contain no videos.")
    return records


def split_kth_videos(records, train_ratio=0.8, test_ratio=0.2, seed=0):
    if not 0 < train_ratio < 1 or not math.isclose(train_ratio + test_ratio, 1.0):
        raise ValueError("KTH train/test ratios must be positive and sum to 1.")
    count = int(len(records) * train_ratio)
    if not 0 < count < len(records):
        raise ValueError("At least two eligible videos are needed for train/test splitting.")
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed)).tolist()
    return [records[i] for i in order[:count]], [records[i] for i in order[count:]]


class KTHDataset(Dataset):
    def __init__(self, records, frames_per_sample=40, frame_stride=1, image_size=64,
                 frame_time_delta=1.0, time_origin=0.0, random_time=False,
                 horizontal_flip=False, seed=0):
        self.records = list(records)
        self.frames_per_sample = int(frames_per_sample)
        self.frame_stride = int(frame_stride)
        self.image_size = int(image_size)
        self.frame_time_delta = float(frame_time_delta)
        self.time_origin = float(time_origin)
        self.random_time = random_time
        self.horizontal_flip = horizontal_flip
        self.seed = int(seed)
        if min(self.frames_per_sample, self.frame_stride, self.image_size) < 1:
            raise ValueError("KTH frame count, stride and image size must be positive.")
        if self.frame_time_delta <= 0 or not math.isfinite(self.frame_time_delta + self.time_origin):
            raise ValueError("KTH requires positive finite dt and finite time_origin.")
        self.span = (self.frames_per_sample - 1) * self.frame_stride + 1
        if not self.records or any(record.length < self.span for record in self.records):
            raise ValueError("KTH dataset is empty or contains videos shorter than a clip.")

    def __len__(self):
        return len(self.records)

    def clip_start(self, index):
        high = self.records[index].length - self.span + 1
        generator = None if self.random_time else torch.Generator().manual_seed(self.seed + index)
        return int(torch.randint(high, (1,), generator=generator))

    def __getitem__(self, index):
        return self.get_clip(index, self.clip_start(index))

    def get_clip(self, index, start):
        record = self.records[index]
        if start < 0 or start + self.span > record.length:
            raise ValueError("Requested KTH clip is outside the video.")
        indices = start + np.arange(self.frames_per_sample) * self.frame_stride
        flip = self.horizontal_flip and bool(torch.randint(2, (1,)))
        frames = []
        # No open HDF5 handles are retained across DataLoader workers.
        with h5py.File(record.shard, "r") as handle:
            store = _frame_store(handle, record.key)
            for i in indices:
                frame = np.asarray(store[int(i)] if isinstance(store, h5py.Dataset) else store[str(i)][()])
                if frame.dtype != np.uint8:
                    raise ValueError("KTH frames must be uint8 in [0,255].")
                if frame.ndim == 2:
                    frame = frame[..., None]
                if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
                    raise ValueError(f"Expected HWC grayscale/RGB frames, got {frame.shape}.")
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                h, w = frame.shape[:2]
                scale = self.image_size / min(h, w)
                if scale != 1:
                    frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
                h, w = frame.shape[:2]
                top, left = (h - self.image_size) // 2, (w - self.image_size) // 2
                frame = frame[top:top+self.image_size, left:left+self.image_size]
                if flip:
                    frame = frame[:, ::-1]
                frames.append(torch.from_numpy(frame.copy()).permute(2, 0, 1))
        time = self.time_origin + torch.as_tensor(indices, dtype=torch.float32) * self.frame_time_delta
        return torch.stack(frames).float().div(127.5).sub(1.0), time


def build_kth_datasets(config, project_root):
    from src.utils.config import resolve_path

    data = config.dataset
    frames = int(data.condition_frames) + int(data.prediction_frames)
    if int(data.condition_frames) < 2 or int(data.prediction_frames) < 1:
        raise ValueError("KTH needs at least 2 conditioning frames and 1 predicted frame.")
    stride = int(data.frame_stride)
    records = discover_kth_videos(resolve_path(project_root, data.dataset_path))
    eligible = [record for record in records if record.length >= (frames-1)*stride+1]
    train, test = split_kth_videos(eligible, float(data.train_ratio), float(data.test_ratio), int(config.training.seed))
    kwargs = dict(frames_per_sample=frames, frame_stride=stride, image_size=int(data.image_size),
                  frame_time_delta=float(data.frame_time_delta), time_origin=float(data.time_origin),
                  seed=int(config.training.seed))
    return (KTHDataset(train, random_time=True, horizontal_flip=bool(data.horizontal_flip), **kwargs),
            KTHDataset(test, random_time=False, **kwargs))
