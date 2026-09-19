# Data and Model Assets

Data are not included in the source ZIP. Review upstream usage terms before
downloading. Use only trusted pickle/checkpoint files; SHA-256 detects changes
but does not make an unknown pickle safe.

## Acrobot Angles

The default benchmark input is a pickle containing a list of dictionaries:
`t` is an increasing array `[T]`; `y` is `[4,T]` containing two angles and two
angular velocities. Uniform frame spacing must match `training.time_delta`.
The exact original dataset needs a public URL, checksum and generation metadata.

For installation checks, `generate_acrobot.py` produces a numeric NPZ with
`t=[T]`, `y=[N,4,T]`, seed 42, no noise and 30 frames per second. It uses the
included two-link ODE with a fixed time grid. This demo is not a substitute for
the stochastic benchmark and must not be used to claim reproduction of it.

## NSE

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
`evaluation.i3d_checkpoint`; it is not silently downloaded. Its authoritative
redistribution terms and checksum still need to be finalized for release.

Acrobot frame loading accepts `trajectory_name/frame_000.png`, uses grayscale
GPE preprocessing and provides image/pair/window/trajectory modes. It does not
provide end-to-end raw-frame GPE training. Do not advertise that branch as a
supported paper experiment until the missing model/weight pipeline is supplied.
