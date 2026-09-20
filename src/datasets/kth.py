"""Lazy HDF5 KTH clips with video-level splits and explicit frame times."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import hashlib
import json
import re

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
    subject: int | None = None
    source: str = ""


# Subject IDs from the official 00sequences.txt, not inferred from shard names.
OFFICIAL_TRAIN = frozenset(range(11, 19))
OFFICIAL_VALIDATION = frozenset((1, 4, 19, 20, 21, 23, 24, 25))
OFFICIAL_TEST = frozenset((2, 3, 5, 6, 7, 8, 9, 10, 22))


def subject_from_source(source):
    match = re.search(r"(?:^|[/\\])person(\d{2})_", str(source))
    return int(match.group(1)) if match else None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        sidecar = shard.with_suffix(".json")
        metadata = {}
        if sidecar.is_file():
            metadata = {str(item["key"]): item for item in
                        json.loads(sidecar.read_text(encoding="utf-8")).get("videos", [])}
        with h5py.File(shard, "r") as handle:
            keys = list(handle["len"].keys()) if "len" in handle else [k for k in handle if k.isdigit()]
            for key in sorted(keys, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else k)):
                store = _frame_store(handle, key)
                length = int(handle["len"][key][()]) if "len" in handle else len(store)
                if len(store) < length:
                    raise ValueError(f"Invalid video length in {shard}:{key}.")
                source = handle[key].attrs.get("source", metadata.get(key, {}).get("source", ""))
                source = source.decode() if isinstance(source, bytes) else str(source)
                inferred = subject_from_source(source or key)
                subject = handle[key].attrs.get("subject_id", inferred)
                subject = None if subject is None else int(subject)
                if subject is not None and (subject not in range(1, 26) or
                                             (inferred is not None and inferred != subject)):
                    raise ValueError(f"Invalid or conflicting subject metadata for video {key}.")
                records.append(KTHVideo(shard.resolve(), key, length, subject, source))
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
    from src.utils.download import sha256_file

    data = config.dataset
    frames = int(data.condition_frames) + int(data.prediction_frames)
    if int(data.condition_frames) < 2 or int(data.prediction_frames) < 1:
        raise ValueError("KTH needs at least 2 conditioning frames and 1 predicted frame.")
    stride = int(data.frame_stride)
    root = resolve_path(project_root, data.dataset_path).resolve()
    records = discover_kth_videos(root)
    protocol = data.get("split_protocol", "official_subjects")
    if protocol == "official_subjects":
        if any(record.subject is None for record in records):
            raise ValueError("Official KTH splitting requires subject metadata. Re-run prepare_kth.py "
                             "or supply the converter's matching JSON sidecar; numeric IDs alone are insufficient.")
        train_subjects = OFFICIAL_TRAIN
        if data.get("merge_official_validation", False):
            train_subjects = train_subjects | OFFICIAL_VALIDATION
        train = [record for record in records if record.subject in train_subjects]
        test = [record for record in records if record.subject in OFFICIAL_TEST]
    elif protocol == "random_video":
        train, test = split_kth_videos(records, float(data.get("train_ratio", 0.8)),
                                      float(data.get("test_ratio", 0.2)), int(config.training.seed))
    else:
        raise ValueError("split_protocol must be official_subjects or random_video.")
    base = root.parent if root.is_file() else root
    def describe(record):
        return dict(shard=record.shard.relative_to(base).as_posix(), key=record.key,
                    length=record.length, subject=record.subject, source=record.source)
    partitions = {"train": [describe(r) for r in train], "test": [describe(r) for r in test]}
    shards = {p.relative_to(base).as_posix(): sha256_file(p) for p in sorted({r.shard for r in records})}
    identity = dict(version=1, protocol=protocol,
                    dataset_sha256=_digest(dict(shards=shards, videos=[describe(r) for r in records])),
                    split_sha256=_digest(partitions), image_size=int(data.image_size))
    # Membership is fixed before horizon-dependent eligibility filtering.
    span = (frames - 1) * stride + 1
    train = [record for record in train if record.length >= span]
    test = [record for record in test if record.length >= span]
    kwargs = dict(frames_per_sample=frames, frame_stride=stride, image_size=int(data.image_size),
                  frame_time_delta=float(data.frame_time_delta), time_origin=float(data.time_origin),
                  seed=int(config.training.seed))
    datasets = (KTHDataset(train, random_time=True, horizontal_flip=bool(data.horizontal_flip), **kwargs),
                KTHDataset(test, random_time=False, **kwargs))
    for dataset in datasets:
        dataset.data_identity = identity
        dataset.split_manifest = dict(identity=identity, partitions=partitions,
                                     eligible_train=[describe(r) for r in train],
                                     eligible_test=[describe(r) for r in test])
    return datasets
