# PhiBE-Flow

Time-conditioned forecasting with velocity fields `v(x,t)` in state, image or
latent space.

| Dataset | Model |
| --- | --- |
| Acrobot angles | Direct state-space velocity predictor |
| Acrobot frames | MLP / external GPE codec with latent velocity prediction |
| NSE | Time-conditioned image-space U-Net |
| KTH | Frozen VQ-VAE with a time-conditioned latent Transformer |

## Installation

```bash
conda env create -f environment.yml
conda activate anonymous-forecasting
```

Python 3.10 is recommended for a new environment. Python 3.8 / PyTorch 1.13.1
has also been tested. For an existing environment, install `requirements.txt`
after selecting a compatible CPU/CUDA PyTorch build. Load only trusted data
and checkpoints. Commands below run from the repository root; paths in YAML
are project-relative. Use `--data`, `--device`, `--batch-size` and
`--num-workers` to override runtime settings.

## Data and Training

Data and pretrained weights are not bundled. Formats and sources are listed in
[DATASETS.md](DATASETS.md).

### Acrobot angles

Generate a small deterministic demonstration and run train/test:

```bash
python scripts/generate_acrobot.py
python scripts/train.py --config configs/acrobot_angles.yaml --data data/acrobot_angles/demo.npz --epochs 1
python scripts/eval.py --config configs/acrobot_angles.yaml --data data/acrobot_angles/demo.npz
```

For the noisy benchmark recipe, use `scripts/generate_acrobot_benchmark.py`
and follow its time-grid requirements in DATASETS.md.

### Acrobot frames

Put images under `data/acrobot_frames/traj_000/frame_000.png`, etc. The small
MLP-codec configuration requires no pretrained weights:

```bash
python scripts/check_setup.py --config configs/smoke/acrobot_frames.yaml
python scripts/train.py --config configs/smoke/acrobot_frames.yaml
python scripts/eval.py --config configs/smoke/acrobot_frames.yaml
```

Use `configs/acrobot_frames.yaml` for full trajectories. To switch to downloaded
GPE source or T/S weights, follow [EXTERNAL_CODECS.md](EXTERNAL_CODECS.md).

### NSE

Download [the NSE shards](https://zenodo.org/records/10939479), verify their
format and check for duplicates, then convert them or use trusted PT files directly:

```bash
python scripts/prepare_nse.py --source downloaded_nse --output data/nse
python scripts/check_setup.py --config configs/nse.yaml
python scripts/train.py --config configs/nse.yaml
python scripts/eval.py --config configs/nse.yaml
```

Set normalization and the physical time interval for your data. Conversion
requires RAM for one complete PT shard.

### KTH

Review the dataset/model terms before downloading:

```bash
python scripts/download_assets.py kth_raw --accept-terms
python scripts/prepare_kth.py
python scripts/download_assets.py kth_vqvae --accept-terms
python scripts/check_setup.py --config configs/kth.yaml --load-codec
python scripts/train.py --config configs/kth.yaml
python scripts/eval.py --config configs/kth.yaml
```

The reference training configuration is GPU-oriented. Use
`configs/smoke/kth.yaml` for a reduced integration check. Optional FVD setup is
documented in [FVD.md](FVD.md).

## Configuration and Outputs

| Setting | Purpose |
| --- | --- |
| Acrobot frames `dataset.prediction_horizon` | Future frames to generate |
| KTH `dataset.prediction_frames` | Future frames to generate |
| Acrobot frames / KTH `evaluation.horizons` | Exact-step and prefix-average metrics |
| NSE `visualization.rollout_steps` | Plot length; numeric rollout covers every test trajectory to its end |
| `visualization.num_fixed_test_samples` | Fixed samples reused across epochs |
| `training.checkpoint_path` | Load a compatible checkpoint |
| `training.load_weights_only: true` | Run without parameter updates; requires a checkpoint |

Outputs are written to `experiments/<name>/{checkpoints,logs,results,figures}`.
Acrobot and NSE use 80:20 trajectory/file splits. KTH uses the official subject
IDs, excluding validation subjects by default; see DATASETS.md. No validation
loader or test-based model selection is used.
Test visualizations reuse samples fixed before training. Each epoch evaluates
all test trajectories/videos with autoregressive rollout and saves an independent
`epoch_NNN.pt` plus `last.pt`. No `best.pt` is generated. Acrobot angle/NSE rollout
metrics exclude the conditioning prefix and cover each trajectory to its end;
image/KTH rollout uses the configured future horizon.

Image metrics use future pixels in [0,1]. `*_at_h` measures frame h;
`*_first_h` averages frames 1 through h. GIFs require an animation-capable viewer.
Logs include timing. Acrobot/NSE checkpoint loading is a warm start; KTH also
restores optimizer, EMA and RNG state. Old time-independent checkpoints are
not interchangeable with the current predictors.

KTH checkpoints include full-shard data hashes and the unfiltered split hash.
Changing the forecast horizon only filters within a fixed split. Loading fails
if the data, split or required identity metadata differ; old unverified KTH
checkpoints must be retrained under the recorded protocol.

## Code and Tests

| Directory | Purpose |
| --- | --- |
| `configs/` | Dataset and experiment settings |
| `src/datasets/` | Loading, splitting and sampling |
| `src/method/loss.py` | PhiBE objectives with full spatial diffusion; NSE retains its data-only drift-square term |
| `src/models/` | Predictors and codec interfaces |
| `src/trainers/` | Optimization, rollout, evaluation and checkpointing |
| `src/utils/` | Metrics, visualization, time, logging and I/O |
| `scripts/` | Train/eval, data preparation and source export |
| `tests/` | Regression and small pipeline tests |

```bash
python -m unittest discover -s tests -v
python scripts/audit_release.py
python scripts/export_release.py
```

The exporter excludes data, weights, local experiments and Git history.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing
terms. GPE source and weights must be obtained separately.
