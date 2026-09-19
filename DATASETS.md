# Data and Model Assets

Data are not included in the source ZIP. Review upstream usage terms before
downloading. Use only trusted pickle/checkpoint files; SHA-256 detects changes
but does not make an unknown pickle safe.

## Acrobot Angles

The default benchmark input is a pickle containing a list of dictionaries:
`t` is an increasing array `[T]`; `y` is `[4,T]` containing two angles and two
angular velocities. Uniform frame spacing must match `training.time_delta`.
The supplied original recipe can be run with
`python scripts/generate_acrobot_benchmark.py`. Its defaults are 10,000
trajectories, duration 8, fps argument 30 and noise 0.5. A new explicit seed is
recorded; the original source did not seed its RNG, so this is not a promise of
bitwise reproduction of historical paper data. Pass the output via
`--data data/acrobot_angles/benchmark.npz` to train/eval.

The source adds independent Gaussian draws to all four RHS components inside
adaptive `solve_ivp`/RK45. This is preserved for compatibility, **not** presented
as a standard SDE solver. Changing it to Euler-Maruyama would change the data
generation protocol and requires a separate experiment decision. The source's
`linspace(0,8,240)` has frame interval `8/239`, **not** `1/30`; set
`training.time_delta` accordingly and run `check_setup.py` before training.
The numeric NPZ contains solver/library versions and RNG seed as metadata.

For installation checks, `generate_acrobot.py` produces a numeric NPZ with
`t=[T]`, `y=[N,4,T]`, seed 42, no noise and 30 frames per second. It uses the
included two-link ODE with a fixed time grid. This demo is not a substitute for
the stochastic benchmark and must not be used to claim reproduction of it.

## NSE

The supplied source is [Stochastic Navier-Stokes dataset for probabilistic
forecasting](https://zenodo.org/records/10939479), DOI
`10.5281/zenodo.10939479` (Mengjian Hua, 2024), licensed CC BY 4.0 on the record.
Readers can manually download `data_file.pt`, `data_file02.pt`,
`data_file03.pt`, `data_file04.pt` and `data_file05.pt` (about 26.2 GB total).
The record lists identical MD5 values for the first and third files; verify
the downloads and check for duplicates before fixing train/test membership.
Do not silently deduplicate a paper benchmark or put duplicate trajectories
on opposite sides of a split. Download/schema verification and fixed SHA-256
values are still pending, so the automatic downloader remains blocked.

Each `.pt` shard contains `[N,T,H,W]` data, either directly, as the first tuple
element, or under `data`, `trajectories`, `tensor` or `x`. The converted `.npy`
format is float32 with exactly that shape and is memory-mapped on load.
No axes are guessed or transposed automatically.

Publish the viscosity, forcing, domain, boundary conditions, solver, grid,
integration step, saved-frame interval, simulation duration, initial-condition
distribution, seeds, shard ordering and checksums with the exact data. None of
these can safely be inferred from a directory of tensors.

Default normalization is the original experiment's mean frame RMS. Set
`dataset.normalization: auto` to compute this statistic from training shards
only when using new data. Time belongs to the middle sampled frame:
`t = time_origin + (window_start + time_lag) * time_delta / time_lag`.
PT conversion loads one full file; NPY training avoids materializing the whole
normalized dataset. Keep `num_workers: 0` for the lowest memory use on Windows.

## KTH

The [official data page](https://www.csc.kth.se/cvap/actions/) provides six AVI
ZIP archives. It specifies non-commercial use and requests citation of Schuldt,
Laptev and Caputo, ICPR 2004. The downloader records local checksums because
the official archive page does not publish fixed SHA-256 values.

`prepare_kth.py` decodes each whole AVI, converts BGR to RGB, resizes the shorter
side to 64 using bilinear interpolation and center-crops to 64x64. HDF5 stores
uint8 `[T,H,W,C]` arrays under numeric video IDs and lengths under `len/<id>`.
ZIP entry paths are never used as output filesystem paths. Short clips and
conversion provenance are reported in the companion JSON manifest.

The loader additionally accepts `video_id/frames` stores and numbered frame
groups. Training maps pixels to [-1,1]. A seeded video-level 80:20 split is used,
including any shards under subdirectories; original train/val folder names do
not preserve upstream splits. This is not a subject-disjoint official protocol.

The [RIVER KTH image codec](https://huggingface.co/cvg-unibe/river_kth_64/blob/main/vqvae.ckpt)
is `f8_small`, 64x64 RGB -> 4x8x8 continuous latent. Decoding includes vector
quantization. Its 857703197-byte checkpoint has SHA-256:

`dd3c614d100181b0c3fe5c31675763a9340b448cfbcd1b3650bc9b1fb566a04f`

The compatibility loader ignores the stored Lightning ModelCheckpoint class
used only by training metadata, so Lightning is not required for inference.
This is not an untrusted-pickle sandbox.

## Optional Assets and Missing Support

FVD needs an external I3D TorchScript model configured with
`evaluation.i3d_checkpoint`; it is not silently downloaded. The actual server
detector SHA-256 is pinned in `configs/kth.yaml` and checked before loading.
See FVD.md for the server's download reference and protocol. The OneDrive
state-dictionary link is not a drop-in substitute for TorchScript.

Acrobot frame loading accepts `trajectory_name/frame_000.png`, uses grayscale
grayscale preprocessing and provides image/pair/window/trajectory modes.
The unified from-scratch image baseline trains its own lightweight codec and
needs no pretrained weights. It is not GPE training or paper reproduction.
GPE code and weights are not bundled; see EXTERNAL_CODECS.md for the upstream
link, reader-supplied interface and the precise remaining integration work.
