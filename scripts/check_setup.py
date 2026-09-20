"""Validate local assets and data shape before allocating a training model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.cli import add_runtime_arguments, apply_runtime_arguments
from src.utils.config import load_config, resolve_path


def check(config, root=ROOT, load_codec=False):
    import numpy as np
    import torch

    data = config.dataset
    path = resolve_path(root, data.dataset_path)
    report = {"dataset": data.name, "python": sys.version.split()[0], "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "warnings": []}
    if str(config.training.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu or install matching CUDA PyTorch.")
    if data.name == "acrobot_angles":
        from src.datasets.acrobot_angles import load_acrobot_pickle, AcrobotAnglesDataset
        from scripts.train import split_trajectories
        records = load_acrobot_pickle(path)
        train, test = split_trajectories(records, float(data.train_ratio), float(data.test_ratio), int(config.training.seed))
        for record in records:
            times = np.asarray(record["t"])
            states = np.asarray(record["y"])
            if states.shape != (4, len(times)) or not np.isfinite(states).all() or not np.isfinite(times).all():
                raise ValueError("Invalid Acrobot shape or non-finite values.")
            if not np.all(np.diff(times) > 0):
                raise ValueError("Acrobot times must increase strictly.")
            if not np.allclose(np.diff(times), float(config.training.time_delta), rtol=0.01, atol=1e-6):
                raise ValueError("Acrobot timestamps disagree with training.time_delta.")
        dataset = AcrobotAnglesDataset([r for _, r in train], int(data.seq_length), bool(data.transform_angles))
        report.update(train_trajectories=len(train), test_trajectories=len(test), sample_shape=list(dataset[0][0].shape))
    elif data.name == "acrobot_frames":
        from src.datasets.acrobot_frames import build_acrobot_frame_datasets
        train, test = build_acrobot_frame_datasets(config, root)
        context, future, time = test[0]
        report.update(train_trajectories=len(train.trajectory_names),
                      test_trajectories=len(test.trajectory_names),
                      sample_shape=list(context.shape), future_shape=list(future.shape),
                      codec_type=config.get("codec", {}).get("type", "mlp"))
        if load_codec:
            from src.models.external_codec import build_image_codec
            codec, identity, needs_training = build_image_codec(config, root, torch.device("cpu"))
            with torch.no_grad():
                reconstructed = codec(context)
            report.update(codec_identity=identity, reconstruction_shape=list(reconstructed.shape),
                          needs_reconstruction_pretraining=needs_training)
        report["warnings"].append("External source/weights require compatible architecture and independent provenance checks.")
    elif data.name == "nse":
        from src.datasets.nse import resolve_nse_files, split_nse_files, load_nse_tensor
        files = resolve_nse_files(path)
        train, test = split_nse_files(files, float(data.train_ratio), float(data.test_ratio))
        for shard in files:
            tensor = load_nse_tensor(shard)
            if min(tensor.shape) < 1 or tensor.shape[1] <= 2*int(data.time_lag):
                raise ValueError("NSE shard has no valid three-frame windows.")
            if not torch.isfinite(tensor[0, :3]).all():
                raise ValueError("NSE sample contains non-finite values.")
            del tensor
        report.update(train_files=len(train), test_files=len(test))
        if any(shard.suffix == ".pt" for shard in files):
            report["warnings"].append(".pt files load into RAM. Convert large data to .npy with prepare_nse.py.")
    elif data.name == "kth":
        from src.datasets.kth import build_kth_datasets
        from src.utils.download import sha256_file
        train, test = build_kth_datasets(config, root)
        weight = resolve_path(root, config.vqvae.checkpoint_path)
        if not weight.is_file():
            raise FileNotFoundError("Missing KTH VQ-VAE. Run download_assets.py kth_vqvae --accept-terms.")
        expected = json.loads((Path(root)/"assets.json").read_text())["assets"]["kth_vqvae"]["sha256"]
        digest = sha256_file(weight)
        if digest != expected:
            raise ValueError("KTH codec differs from the declared public checkpoint. Verify your custom asset before training.")
        clip, times = test[0]
        if data.image_size != 64 or tuple(config.model.state_res) != (8, 8) or config.model.state_size != 4:
            raise ValueError("The supplied KTH pretrained codec requires 64px RGB / 4x8x8 latent settings.")
        if any(h < 1 or h > int(data.prediction_frames) for h in config.evaluation.horizons):
            raise ValueError("Metric horizons must not exceed prediction_frames.")
        report.update(train_videos=len(train), test_videos=len(test), sample_shape=list(clip.shape),
                      codec_sha256=digest, data_identity=train.data_identity)
        if load_codec:
            from src.models.vqvae import VQVAE
            codec = VQVAE(weight, chunk_size=1)
            with torch.no_grad():
                z = codec.encode(clip[:1])
                reconstructed = codec.decode(z)
            if not torch.isfinite(reconstructed).all():
                raise ValueError("KTH codec returned non-finite values.")
            report.update(latent_shape=list(z.shape), reconstruction_shape=list(reconstructed.shape))
    else:
        raise ValueError("Expected acrobot_angles, acrobot_frames, nse or kth.")
    report["status"] = "local_assets_validated_not_a_convergence_claim"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-codec", action="store_true", help="Also load the selected KTH/Acrobot codec and check reconstruction.")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    config = apply_runtime_arguments(load_config(resolve_path(ROOT, args.config)), args)
    try:
        print(json.dumps(check(config, load_codec=args.load_codec), indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Setup check failed: {error}\n")


if __name__ == "__main__":
    main()
