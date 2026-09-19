from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import struct
import sys
import zipfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _StorageMetadata:
    def __init__(self, key: str, size: int) -> None:
        self.key = key
        self.size = size


class _TensorMetadata:
    def __init__(self, storage, offset, shape, stride, *_args) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = tuple(shape)
        self.stride = tuple(stride)


def _rebuild_tensor(storage, offset, shape, stride, *_args):
    return _TensorMetadata(storage, offset, shape, stride)


class _TorchMetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return name
        return super().find_class(module, name)

    def persistent_load(self, persistent_id):
        kind, _storage_type, key, _location, size = persistent_id
        if kind != "storage":
            raise ValueError(f"Unsupported persistent object: {kind}")
        return _StorageMetadata(str(key), int(size))


def _zip_entry_data_offset(path: Path, info: zipfile.ZipInfo) -> int:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise ValueError(f"Invalid ZIP local header in {path}")
    return info.header_offset + 30 + fields[-2] + fields[-1]


def mmap_first_tensor(path: str | Path):
    """Memory-map the first tensor in a standard uncompressed torch ZIP checkpoint."""
    import io
    import numpy as np

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        pickle_info = next(info for info in archive.infolist() if info.filename.endswith("data.pkl"))
        payload = _TorchMetadataUnpickler(io.BytesIO(archive.read(pickle_info))).load()
        metadata = payload[0] if isinstance(payload, (tuple, list)) else payload
        if not isinstance(metadata, _TensorMetadata):
            raise ValueError(f"The first object in {path} is not a tensor.")
        storage_suffix = f"data/{metadata.storage.key}"
        storage_info = next(
            info for info in archive.infolist() if info.filename.endswith(storage_suffix)
        )
        if storage_info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Memory-mapped overfit mode requires an uncompressed torch storage.")
        storage_offset = _zip_entry_data_offset(path, storage_info)

    expected_stride = []
    running = 1
    for dimension in reversed(metadata.shape):
        expected_stride.append(running)
        running *= dimension
    if metadata.stride != tuple(reversed(expected_stride)):
        raise ValueError("Only contiguous NSE tensors are supported by this diagnostic.")

    return np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=storage_offset + metadata.offset * 4,
        shape=metadata.shape,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a tiny fixed NSE sample.")
    parser.add_argument("--data", type=Path, required=True, help="One NSE .pt file.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--time-lag", type=int, default=2)
    parser.add_argument("--time-delta", type=float, default=1.0)
    parser.add_argument("--time-origin", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--normalization", type=float, default=3.0679163932800293)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "nse_overfit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_lag < 1 or args.time_delta <= 0:
        raise ValueError("time-lag and time-delta must be positive.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from src.datasets.nse import resize_nse_input, resize_nse_target
    from src.method.loss import nse_prediction_mse, nse_velocity_loss
    from src.models.nse_predictor import NSEDriftModel

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = mmap_first_tensor(args.data)
    max_windows = data.shape[1] - 2 * args.time_lag
    if not 0 <= args.trajectory < data.shape[0]:
        raise ValueError(f"trajectory must be in [0, {data.shape[0] - 1}].")
    if not 1 <= args.windows <= max_windows:
        raise ValueError(f"windows must be in [1, {max_windows}].")

    trajectory = args.trajectory
    starts = np.arange(args.windows)
    x0_raw = torch.from_numpy(np.array(data[trajectory, starts], copy=True)) / args.normalization
    x1_raw = torch.from_numpy(
        np.array(data[trajectory, starts + args.time_lag], copy=True)
    ) / args.normalization
    x2_raw = torch.from_numpy(
        np.array(data[trajectory, starts + 2 * args.time_lag], copy=True)
    ) / args.normalization
    batch = (
        resize_nse_input(x0_raw, args.image_size, args.image_size),
        resize_nse_input(x1_raw, args.image_size, args.image_size),
        resize_nse_target(x2_raw, args.image_size),
        torch.tensor(args.time_origin + (starts + args.time_lag) * args.time_delta / args.time_lag, dtype=torch.float32),
    )

    model = NSEDriftModel(
        dim=args.channels,
        dim_mults=(1, 2, 2),
        resnet_block_groups=8,
        learned_sinusoidal_dim=16,
        attention_dim_head=16,
        attention_heads=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    model.eval()
    with torch.no_grad():
        initial_mse = float(nse_prediction_mse(model, batch, device, args.time_delta))
    history = []
    checkpoints = {1, 10, 25, 50, 100, 200, args.steps}
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nse_velocity_loss(model, batch, device, args.time_delta)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            mse = float(nse_prediction_mse(model, batch, device, args.time_delta))
        history.append({"step": step, "loss": float(loss.detach()), "mse": mse})
        if step in checkpoints or step % 100 == 0:
            print(
                f"step={step:04d} loss={float(loss.detach()):.8f} "
                f"mse={mse:.10f} grad_norm={float(grad_norm):.6f}",
                flush=True,
            )

    final_mse = history[-1]["mse"]
    reduction = final_mse / initial_mse
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "data": str(args.data),
        "tensor_shape": list(data.shape),
        "device": str(device),
        "steps": args.steps,
        "windows": args.windows,
        "trajectory": args.trajectory,
        "time_delta": args.time_delta,
        "time_origin": args.time_origin,
        "time_lag": args.time_lag,
        "model_channels": args.channels,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "final_to_initial_ratio": reduction,
        "converged": reduction < 0.05,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "summary": summary},
        args.output_dir / "overfit.pt",
    )

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy([item["step"] for item in history], [item["mse"] for item in history])
    axis.axhline(initial_mse, color="tab:red", linestyle="--", label="Initial MSE")
    axis.set_xlabel("Optimization Step")
    axis.set_ylabel("Fixed-batch MSE")
    axis.set_title("NSE Small-Sample Overfit")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)

    print(
        f"initial_mse={initial_mse:.10f} final_mse={final_mse:.10f} "
        f"ratio={reduction:.6f} converged={summary['converged']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
