"""Convert official KTH AVI files/ZIPs to one lazy-loading HDF5 shard.

Each full AVI is one video; source filenames and subject IDs are retained.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import zipfile
import re

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def video_sources(directory):
    sources = [(path, None) for path in sorted(directory.rglob("*.avi"))]
    for path in sorted(directory.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.filename.lower().endswith(".avi") and not info.is_dir():
                    sources.append((path, info.filename))
    names = [Path(member).name if member else path.name for path, member in sources]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate AVI names found. Use archives OR extracted files, not both.")
    if not sources:
        raise FileNotFoundError("No AVI files or ZIP archives found.")
    return sources


@contextmanager
def local_video(path, member):
    if member is None:
        yield path
        return
    # Never use archive entry paths as filesystem destinations (ZIP traversal).
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as archive:
        target = Path(tmp) / "video.avi"
        with archive.open(member) as source, target.open("wb") as destination:
            while True:
                chunk = source.read(1024*1024)
                if not chunk:
                    break
                destination.write(chunk)
        yield target


def convert(source, output, image_size=64, min_frames=1):
    if image_size < 1 or min_frames < 1:
        raise ValueError("image_size and min_frames must be positive.")
    sources = video_sources(Path(source))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".hdf5.part")
    if temporary.exists():
        raise FileExistsError("An incomplete HDF5 conversion exists; choose a new output path.")
    records, skipped = [], []
    try:
        with h5py.File(temporary, "x") as handle:
            lengths = handle.create_group("len")
            for path, member in sources:
                name = Path(member).name if member else path.name
                with local_video(path, member) as local:
                    capture = cv2.VideoCapture(str(local))
                    if not capture.isOpened():
                        raise ValueError(f"OpenCV cannot decode {name}.")
                    frames = []
                    try:
                        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        while True:
                            valid, frame = capture.read()
                            if not valid:
                                break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w = frame.shape[:2]
                            scale = image_size / min(h, w)
                            frame = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_LINEAR)
                            h, w = frame.shape[:2]
                            y, x = (h-image_size)//2, (w-image_size)//2
                            frames.append(frame[y:y+image_size, x:x+image_size])
                    finally:
                        capture.release()
                    if expected > 0 and len(frames) != expected:
                        raise ValueError(f"Incomplete AVI decode: {name} ({len(frames)}/{expected}).")
                if len(frames) < min_frames:
                    skipped.append(name)
                    continue
                key = str(len(records))
                store = handle.create_dataset(key, data=np.stack(frames), compression="gzip", compression_opts=1,
                                              chunks=(1, image_size, image_size, 3))
                store.attrs["source"] = name
                subject = re.match(r"person(\d{2})_", name)
                if subject:
                    store.attrs["subject_id"] = int(subject.group(1))
                lengths[key] = len(frames)
                records.append(dict(key=key, source=name, frames=len(frames)))
                print(f"Converted {len(records)}/{len(sources)}: {name}", flush=True)
            if len(records) < 2:
                raise ValueError("At least two eligible videos are required.")
        temporary.rename(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    manifest = dict(protocol="whole_avi_with_subject_metadata", image_size=image_size,
                    videos=records, skipped_short_videos=skipped)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/kth_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/kth/videos.hdf5")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
