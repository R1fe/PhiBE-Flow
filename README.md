# Time-Conditioned Forecasting

Anonymous source artifact for state-space, image-space and latent-space
forecasting with explicit time-dependent velocity fields.

**Release status: prepared source draft, not yet a complete benchmark release.**
NSE and the original Acrobot benchmark still need public assets/metadata, and
project-wide licensing and venue-specific anonymity checks remain pending.
See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md). Do not interpret smoke tests
as evidence that paper results have been reproduced.

## Supported Pipelines

| Dataset | Train / test | Data readiness |
| --- | --- | --- |
| Acrobot angles | Implemented; direct state prediction | Deterministic demo generator included; original benchmark download pending |
| NSE | Implemented; conditioned image-space U-Net | Requires external simulation shards; exact public benchmark pending |
| KTH | Implemented; frozen VQ-VAE + time-conditioned Transformer | Public video/codec download and conversion commands included |
| Acrobot frames | Dataset loader only | Unified GPE image training/evaluation is not implemented |

No separate validation split is created. Acrobot splits trajectories, NSE splits
sorted whole files, and KTH splits whole videos 80:20. Small datasets are rounded
down for training. KTH is **not** the official subject-based benchmark protocol.

## Installation

Use Python 3.10 for a new environment. The conservative dependency profile is
pinned in `requirements.txt`; tests have also run in Python 3.8 / PyTorch 1.13.1.
The old PyTorch profile is for compatibility, not a claim of current security
support. Only load trusted, checksummed pickle/checkpoint inputs.

```bash
conda env create -f environment.yml
conda activate anonymous-forecasting
python -m unittest discover -s tests -v
```

For a CPU-only installation without Conda:

```bash
python -m venv .venv
# Activate .venv using your shell's normal activation command.
python -m pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA, install the appropriate PyTorch 1.13.1 CUDA wheel for your driver before
the remaining requirements. Training defaults to CUDA when available, otherwise
CPU. `--device cpu`, `--batch-size N` and `--num-workers N` are supported by
both unified entrypoints. Full KTH training is GPU-oriented; CPU smoke tests use
small models. No training run downloads files or uses a telemetry service.

Run commands below from this directory. YAML paths are relative to this project,
not to an author's machine. `--data` overrides the dataset path.

## First End-to-End Run: Acrobot Demonstration

This generates **deterministic, zero-noise demonstration data**, not the original
stochastic benchmark. It is intended to verify installation and the pipeline.

```bash
python scripts/generate_acrobot.py
python scripts/check_setup.py --config configs/acrobot_angles.yaml --data data/acrobot_angles/demo.npz
python scripts/train.py --config configs/acrobot_angles.yaml --data data/acrobot_angles/demo.npz --epochs 1 --batch-size 32 --num-workers 0
python scripts/eval.py --config configs/acrobot_angles.yaml --data data/acrobot_angles/demo.npz --batch-size 32 --num-workers 0
```

For the original benchmark, put the trusted published dataset at
`data/acrobot_angles/acrobot_data.pkl` and omit `--data`. Its public download
has not yet been supplied; the release checklist records this blocker.

## KTH: Download, Prepare, Train, Test

Read the [KTH terms](https://www.csc.kth.se/cvap/actions/) and
[RIVER model card](https://huggingface.co/cvg-unibe/river_kth_64) first. The
official KTH data are available for non-commercial use. Downloads are explicit
and are not redistributed in this source artifact.

```bash
python scripts/download_assets.py kth_raw --accept-terms
python scripts/prepare_kth.py
python scripts/download_assets.py kth_vqvae --accept-terms
python scripts/check_setup.py --config configs/kth.yaml --load-codec
python scripts/train.py --config configs/kth.yaml --batch-size 4 --epochs 1
python scripts/eval.py --config configs/kth.yaml --batch-size 4
```

The commands above are a first run, not full training. The reference config uses
batch 128, learning rate 2e-4, 10,000 warmup steps and at most 400,000 updates.
A one-epoch run with warmup is not a convergence test. Reduce batch size for
memory limits, but record the changed setting. The codec is approximately
858 MB; raw videos total about 1.2 GB, with additional space needed for HDF5 and
large training checkpoints. Public codec SHA-256 is pinned in `assets.json`.

For a real-codec pipeline smoke test, substitute `configs/smoke/kth.yaml` in
the train/eval commands. It uses a small predictor, one update and two future
frames. This tests integration, not benchmark quality.

The converter stores each full AVI as one video and preserves original frame
order. This preparation is not asserted to reproduce any unpublished HDF5
subsequence selection. Use an explicitly documented matching manifest when
reproducing a specific experiment.

## NSE: Bring Simulation Shards

The exact simulation data and generation parameters must be released before
this benchmark is independently reproducible. Once available:

1. Put trusted `.pt` files with shape `[N,T,H,W]` into a temporary input directory.
2. Convert to float32 `.npy` shards for memory-mapped loading, or use `.pt` directly
   if RAM can hold all selected files.
3. Set the dataset path, normalization and physical time grid for that data.

```bash
python scripts/prepare_nse.py --source downloaded_nse --output data/nse
python scripts/check_setup.py --config configs/nse.yaml
python scripts/train.py --config configs/nse.yaml --epochs 1 --batch-size 4
python scripts/eval.py --config configs/nse.yaml --batch-size 4
```

Conversion loads one `.pt` shard at a time; allow RAM for that shard. Converted
`.npy` training normalizes sampled frames on demand. Do not mix original and
converted copies in one dataset directory. Use at least two shards; five equally
sized shards give an exact 4:1 file split. The provided normalization constant
belongs to the original experiment, not arbitrary downloaded NSE data.

## Configuration and Outputs

| Setting | Meaning |
| --- | --- |
| KTH `dataset.prediction_frames` | Number of generated future frames, default 30 |
| KTH `evaluation.horizons` | Report exact-step and prefix-average MSE/PSNR/SSIM |
| KTH `dataset.frame_time_delta` / `frame_stride` | Define time and integration step |
| NSE `dataset.time_lag` / `training.time_delta` | Sampling stride and forecast-time interval |
| NSE `visualization.rollout_steps` | Plot horizon, not a multi-step numeric evaluation setting |
| `visualization.num_fixed_test_samples` | Samples selected before training, unchanged each epoch |
| `training.checkpoint_path` | Load a compatible predictor checkpoint |
| `training.load_weights_only: true` | Run without parameter updates; checkpoint_path is required |
| `evaluation.checkpoint: last.pt` | Evaluate final weights without selecting on test performance |

KTH supports exact spatial VJP loss, EMA, resume, per-stage timers and optional
FVD. Enable FVD only after supplying the separate local I3D detector.
Acrobot/NSE checkpoint loading is a warm start, not exact RNG/scheduler resume.
Legacy velocity checkpoints without time conditioning are not interchangeable.

Outputs go to `experiments/<dataset>/{checkpoints,logs,results,figures}`.
KTH/Acrobot include fixed-test comparison GIFs and rollout images; NSE currently
provides rollout PNGs and one-step metrics. GIFs need an animation-capable viewer.
Older `best.pt` files select by test metrics and are diagnostic only; report
`last.pt` or a training budget fixed in advance. Monitoring test plots repeatedly
also limits the claim that the test set is untouched.

## Verification and Publication

```bash
python -m unittest discover -s tests -v
python scripts/audit_release.py
python scripts/audit_release.py --require-ready
python scripts/export_release.py
```

The strict readiness check is expected to fail while release blockers remain.
The exporter creates a clean `.release/anonymous-code/` directory and ZIP using
a positive file allowlist and fixed ZIP timestamps. It excludes history,
private notes, old experiment scripts/configs, datasets, checkpoints and logs.
Publish only that reviewed directory, not the entire working folder.

Tests cover time conditioning, off-diagonal VJP correctness, checkpoint loading,
frozen mode, HDF5 variants, preprocessing and tiny train/test entrypoints. They
are not full benchmark convergence tests. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for limitations and [DATASETS.md](DATASETS.md) for exact input formats.

Retain [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
Anonymity is not a reason to remove third-party attribution. Project-wide
license selection is pending in [LICENSE_STATUS.md](LICENSE_STATUS.md).
