# Data Formats

Data and weights are not bundled. Review source terms and load only trusted
pickle/checkpoint inputs.

## Acrobot Angles

The loader accepts a pickle list of dictionaries with `t: [T]` and `y: [4,T]`
(two angles and two angular velocities), or numeric NPZ with `t: [T]` and
`y: [N,4,T]`. Frame spacing must match `training.time_delta`.

- `python scripts/generate_acrobot.py` generates deterministic zero-noise demo
  data at 30 fps with zero noise.
- `python scripts/generate_acrobot_benchmark.py` generates the noisy RK45 recipe.
  Defaults: 10,000 trajectories, duration 8, 240 saved frames, noise 0.5. Pass
  `--data data/acrobot_angles/benchmark.npz` to train/eval and set
  `training.time_delta: 0.03347280334728033` (`8/239`, not `1/30`).

The benchmark recipe draws noise inside adaptive RK45 RHS evaluations; it is
not a standard SDE discretization. Its seed and solver metadata are recorded.

## Acrobot Frames

Use one directory per trajectory:

```text
data/acrobot_frames/
  traj_000/frame_000.png
  traj_000/frame_001.png
  traj_001/frame_000.png
  ...
```

Images are converted to grayscale, resized and normalized to [-1,1]. Use at
least two trajectories; five give an exact four-train/one-test split. Codec
setup is documented in [EXTERNAL_CODECS.md](EXTERNAL_CODECS.md).

## NSE

Source: [Stochastic Navier-Stokes dataset for probabilistic forecasting](https://zenodo.org/records/10939479),
DOI `10.5281/zenodo.10939479` (Mengjian Hua, 2024), CC BY 4.0.
Download `data_file.pt`, `data_file02.pt`, `data_file03.pt`, `data_file04.pt`
and `data_file05.pt` (approximately 26.2 GB). The first and third files have
identical MD5 entries on the record: check for duplicates before splitting.
Download NSE files manually from the linked record.

Each PT file must contain `[N,T,H,W]`, either directly, as the first tuple
element, or under `data`, `trajectories`, `tensor` or `x`. Converted NPY files
are float32 with the same shape and use memory-mapped loading. Axes are not
guessed. Do not mix original and converted copies in one directory.

Splitting uses sorted whole files, so use at least two disjoint shards. Set
`dataset.normalization: auto` for a training-only mean frame RMS estimate on
new data. Configure sampling times consistently with the simulation; preserve
the physical parameters, time grid, seeds, file ordering and checksums with
the experiment. Use `num_workers: 0` for low memory use on Windows.

## KTH

The [official dataset](https://www.csc.kth.se/cvap/actions/) provides six AVI
archives for non-commercial use and requests citation of Schuldt, Laptev and
Caputo, ICPR 2004. The downloader records local archive hashes.

`prepare_kth.py` converts BGR to RGB, resizes the shorter side to 64 and
center-crops to 64x64. HDF5 contains uint8 `[T,H,W,C]` arrays under numeric
video IDs, with lengths under `len/<id>`. The loader also supports
`video_id/frames` and numbered-frame groups. Pixels are mapped to [-1,1].
Splitting follows the [official subject lists](https://www.csc.kth.se/cvap/actions/00sequences.txt):

- Train: 11, 12, 13, 14, 15, 16, 17, 18.
- Test: 2, 3, 5, 6, 7, 8, 9, 10, 22.
- Validation subjects 1, 4, 19, 20, 21, 23, 24, 25 are excluded by default.

`dataset.merge_official_validation: true` explicitly merges validation subjects
into training while keeping official test subjects held out. The default is false.
This uses whole AVI clips, not the recognition benchmark's subsequence annotations.

Conversion stores `source` and `subject_id` attributes on each video. Older
shards may use the matching converter JSON sidecar with original filenames.
If subject identity is unavailable, reconvert the official AVI files; numeric
video keys cannot identify subjects and are rejected by the official split.
Membership is assigned before filtering clip lengths. Startup hashes each full
HDF5 shard and records the unfiltered partition identity in checkpoints; changing
data or split prevents loading, while changing prediction length cannot move
training videos into test. `random_video` is available only as an explicit
alternative for synthetic tests, not the default experiment protocol.

The [RIVER KTH codec](https://huggingface.co/cvg-unibe/river_kth_64) maps 64x64
RGB to a 4x8x8 continuous latent; decoding includes vector quantization.
Its checkpoint is 857,703,197 bytes with SHA-256:

`dd3c614d100181b0c3fe5c31675763a9340b448cfbcd1b3650bc9b1fb566a04f`

Optional I3D weights and video construction are described in [FVD.md](FVD.md).
