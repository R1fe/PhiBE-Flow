"""Convert trusted NSE .pt shards to float32 memory-mapped .npy files."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.datasets.nse import load_nse_tensor


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    files = [source] if source.is_file() else sorted(source.glob("*.pt"))
    if not files:
        raise FileNotFoundError("No NSE .pt shards found.")
    if any((destination / f"{path.stem}.npy").exists() for path in files):
        raise FileExistsError("A converted shard already exists; refusing to overwrite.")
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        print(f"Converting {path.name}; allow RAM for one full .pt shard.", flush=True)
        tensor = load_nse_tensor(path)
        output = destination / f"{path.stem}.npy"
        temporary = output.with_suffix(".npy.part")
        if temporary.exists():
            raise FileExistsError(f"An incomplete conversion exists: {temporary.name}")
        try:
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=tuple(tensor.shape))
            for index in range(len(tensor)):
                value = tensor[index].numpy()
                if not np.isfinite(value).all():
                    raise ValueError("NSE input contains non-finite values.")
                array[index] = value
            array.flush()
            del array
            temporary.rename(output)
        except Exception:
            if "array" in locals():
                del array
            temporary.unlink(missing_ok=True)
            raise
        del tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/nse")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
