# Source Code Guide

This guide covers the source-release allowlist, not local datasets, outputs,
external GPE source, or retired experiment entrypoints. Paths are relative to
the repository root. All `__init__.py` files mark Python packages; they are not
standalone training programs.

## 1. Pipeline Overview

Unified entrypoints read YAML, construct data and models, and select a trainer.
Active velocity predictors take explicit time: `v(x,t)` or `v(z_history,t)`.
Autoregressive rollout advances both state and time.

| Experiment | Input and model | Main training/testing implementation |
| --- | --- | --- |
| Acrobot angles | Low-dimensional states, direct velocity prediction | `state_trainer.py`; no image codec |
| Acrobot frames | Grayscale images, codec, latent prediction, decoding | `acrobot_frame_pipeline.py`; MLP, external GPE or TorchScript |
| NSE | Flow-field images, time-conditioned U-Net | `nse_trainer.py`; no latent codec |
| KTH | RGB video, frozen VQ-VAE, time-conditioned Transformer | `kth_pipeline.py` and `kth_trainer.py` |

There is no separate validation set. Splits are 80:20 by Acrobot trajectory,
NSE file, or KTH video, with rounding for small datasets. KTH does not use the
official subject-disjoint protocol. Visualization samples are selected before
training and held fixed. Repeated tuning against test plots compromises an
independent-test interpretation.

## 2. Configuration Files

| File | Purpose |
| --- | --- |
| `configs/acrobot_angles.yaml` | State data, velocity model, PhiBE objective, training and visualization |
| `configs/acrobot_frames.yaml` | Independent MLP image codec configuration; one future frame by default |
| `configs/acrobot_frames_gpe.yaml` | External GPE first-run configuration; random initialization, reconstruction warmup, then frozen codec; not a paper training recipe |
| `configs/nse.yaml` | NSE sampling, normalization, U-Net and image-space objective |
| `configs/kth.yaml` | Reference KTH model size, codec, training budget, rollout and optional FVD |
| `configs/smoke/acrobot_frames.yaml` | Small image pipeline check: two conditioning and six future frames |
| `configs/smoke/kth.yaml` | Reduced-model KTH integration check; not benchmark training |

For Acrobot images, change `dataset.prediction_horizon` for rollout length and
`evaluation.horizons` for metric horizons. KTH uses `dataset.prediction_frames`
for rollout length. NSE `visualization.rollout_steps` changes plots only; numeric
evaluation currently remains one-step. Set `training.load_weights_only: true`
and supply `training.checkpoint_path` to inspect without parameter updates.

## 3. Command Entrypoints: scripts/

| File | Purpose |
| --- | --- |
| `train.py` | Unified dataset-specific training dispatch; supports data/device/codec overrides |
| `eval.py` | Unified checkpoint evaluation, metrics and applicable visualizations |
| `check_setup.py` | Validate data/configuration; `--load-codec` loads the actual codec and checks shapes |
| `generate_acrobot.py` | Deterministic zero-noise demonstration data, not the original benchmark |
| `generate_acrobot_benchmark.py` | Original-style noisy RK45 recipe; not a standard SDE discretization |
| `prepare_kth.py` | Convert AVI files into HDF5 while retaining within-video frame order |
| `prepare_nse.py` | Convert trusted PT shards into float32 memory-mappable NPY files |
| `download_assets.py` | Explicit asset downloads with terms acknowledgement and available checksum verification |
| `overfit_nse.py` | Small-sample overfitting diagnostic for NSE optimization |
| `audit_release.py` | Check release files, common identity/path patterns and readiness blockers; not an anonymity certification |
| `release_files.py` | Positive release allowlist excluding data, weights, external code and retired entrypoints |
| `export_release.py` | Export a clean source directory, ZIP and hash manifest without Git history |

There is no separate demo entrypoint. Training and evaluation generate outputs.

## 4. Method and Models

| File | Purpose |
| --- | --- |
| `src/method/loss.py` | Core Acrobot/PhiBE velocity objectives, separate NSE objective, and KTH VJP objective; spatial derivatives include off-diagonal terms with time held fixed |
| `src/models/predictor.py` | Low-dimensional velocity networks and optional physical parameterization |
| `src/models/acrobot_frame_model.py` | Independent MLP codec, time-conditioned latent velocity, external codec injection and rollout |
| `src/models/external_codec.py` | Reader-owned GPE source/T-S weights and TorchScript adapters, strict shape/weight/source identity checks; no GPE architecture definitions |
| `src/models/nse_predictor.py` | Time-conditioned NSE U-Net predicting image-space changes |
| `src/models/kth_predictor.py` | Time-conditioned latent Transformer retaining higher derivatives required by the objective |
| `src/models/vqvae.py` | Frozen KTH codec loading, chunked encoding/decoding and quantization |
| `src/models/vqvae_taming.py` | VQ-VAE network and quantizer components; retain third-party notices |
| `src/models/latent_dynamics.py` | Legacy latent residual/analog dynamics components; not the unified image default, and not uniformly time-conditioned |
| `src/physics.py` | Acrobot physical dynamics calculations |

The PhiBE objective is not an ordinary nonnegative MSE. A negative loss alone
does not imply an implementation error. Random external-GPE initialization uses
reconstruction warmup, not the original GPE geometric training objective.

## 5. Data Loading: src/datasets/

| File | Purpose |
| --- | --- |
| `acrobot_angles.py` | State trajectories, trajectory splitting, sample construction and time |
| `acrobot_frames.py` | Trajectory image directories, grayscale preprocessing, image/pair/window/trajectory modes |
| `nse.py` | PT/NPY flow-field shards, file-level splitting, sampling times and image pairs |
| `kth.py` | Supported HDF5 layouts, video-level splitting, conditioning/target clips and time |
| `latent.py` | Legacy precomputed latent trajectories, not the new image-to-image entrypoint |

Data are not bundled. See `DATASETS.md` and `data/README.md` for layouts.

## 6. Trainers: src/trainers/

| File | Purpose |
| --- | --- |
| `state_trainer.py` | Acrobot state optimization, testing, checkpoints and fixed-sample visualization |
| `nse_trainer.py` | NSE optimization, one-step numeric evaluation, rollout plots and timers |
| `acrobot_frame_pipeline.py` | Codec initialization/warmup/freezing, latent dynamics training, pixel metrics, GIF/PNG, checkpoints and no-update mode |
| `kth_pipeline.py` | KTH configuration, data, codec, model, train/eval and optional FVD orchestration |
| `kth_trainer.py` | KTH optimization, learning-rate schedule, EMA, restoration, rollout and metrics |
| `latent_trainer.py` | Legacy precomputed-latent training retained for research compatibility |
| `latent_analog_trainer.py` | Legacy analog latent training; not the unified image reproduction pipeline |

Local legacy `train_latent.py` / `train_latent_analog.py` entrypoints are excluded
from the release. Validation terminology in old components does not establish
a separate validation split in unified entrypoints. KTH restores optimizer,
EMA and RNG state; Acrobot/NSE checkpoint loading is a warm start, not exact resume.

## 7. Utilities: src/utils/

| File | Purpose |
| --- | --- |
| `config.py` | YAML loading and project-relative path resolution |
| `cli.py` | Runtime overrides shared by train, eval and setup checks |
| `checkpoint.py` | Common checkpoint saving/loading helpers |
| `logger.py` | Training log output |
| `seed.py` | Python, NumPy and PyTorch random seeds |
| `time.py` | Time-input shape and broadcasting helpers |
| `metrics.py` | MSE, PSNR and SSIM |
| `visualization.py` | State plots, ground-truth/prediction comparisons, English-labeled GIFs and rollout images |
| `fvd.py` | Compatible TorchScript I3D features and FVD; currently integrated into KTH |
| `download.py` | Download and SHA-256 helpers |
| `serialization.py` | Compatible checkpoint reading and restricted deserialization helpers; not an untrusted-file sandbox |

## 8. Tests, Dependencies and Release Files

| File | Purpose |
| --- | --- |
| `tests/test_time_conditioning.py` | Time-input, prediction and spatial-derivative regressions |
| `tests/test_acrobot_frames.py` | Image data organization, splitting and preprocessing |
| `tests/test_acrobot_frame_pipeline.py` | Independent-codec train/eval, fixed visualizations and frozen mode |
| `tests/test_gpe_pipeline.py` | External source/raw weights/TorchScript switching, freezing, restoration, fingerprints and fail-closed behavior; automated fixtures use independent tiny models |
| `tests/test_external_dependencies.py` | External-dependency boundaries and interfaces |
| `tests/test_acrobot_generation.py` | Acrobot generation regressions |
| `tests/test_kth.py` | KTH data, models, VJP, codec, rollout and training mechanisms |
| `tests/test_release_pipeline.py` | Data conversion, downloads, entrypoints and clean exports |
| `.github/workflows/tests.yml` | GitHub Actions CPU tests, not full benchmark convergence |
| `requirements.txt` / `environment.yml` | Base dependencies and Conda setup |
| `requirements-gpe.txt` | Optional external-GPE architecture adapter dependencies; follow its dedicated installation instructions |
| `assets.json` | Asset URLs, destinations, hashes and terms metadata |
| `.gitignore` / `.gitattributes` | Exclude data/cache/weights and normalize text behavior |
| `SOURCE_MANIFEST.json` | Export-generated source hashes for release verification |
| `licenses/RIVER-GPL-3.0.txt` | RIVER third-party license |
| `licenses/TAMING-TRANSFORMERS-MIT.txt` | Taming Transformers third-party license |
| `licenses/DENOISING-DIFFUSION-MIT.txt` | Diffusion-network third-party license |

## 9. Documentation and Remaining Work

| Document | Contents |
| --- | --- |
| `README.md` | Installation, data preparation, train/eval commands and main limitations |
| `DATASETS.md` | Formats, download sources, normalization and splitting |
| `EXTERNAL_CODECS.md` | GPE download, no-weight startup, T/S loading, TorchScript and switching |
| `FVD.md` | I3D acquisition, formats, hashes and current KTH evaluation protocol |
| `REPRODUCIBILITY.md` | Tested scope, numerical conventions and reproducibility limits |
| `RELEASE_CHECKLIST.md` | Outstanding publication checks |
| `THIRD_PARTY_NOTICES.md` / `LICENSE_STATUS.md` | Third-party sources, licenses and unresolved permissions |
| `CODE_GUIDE.md` | This file-by-file guide |

A runnable integration is not full paper reproduction. Remaining work includes:

- The paper's geometric GPE training recipe, publicly obtainable compatible
  weights and their data provenance. The external codec interface is connected.
- Acrobot image FVD integration and protocol verification; KTH FVD support does
  not imply Acrobot FVD support.
- NSE data version, checksums, normalization and protocol verification; numeric
  multi-step NSE evaluation is not yet connected.
- Full-budget convergence, multiple seeds and paper-table reproduction. Smoke
  tests only establish that the pipeline runs.
- Project-wide licensing, historical GPE/identity content and manual anonymity
  review before submission.

Outputs are stored under `experiments/<name>/{checkpoints,logs,results,figures}`
and excluded from source exports. Do not treat the whole development folder or
old Git history as an already reviewed anonymous release.
