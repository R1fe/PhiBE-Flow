# External Acrobot Image Codec

GPE source, architecture definitions and weights are **not bundled**. The
previously copied encoder/decoder have been removed from the current tree.
Download upstream yourself from [GPE](https://github.com/wonjunee/GPE_codes)
and read its LICENSE before use. Its terms require prior approval for
redistribution. This project does not grant that approval.

Keep third-party source outside this repository (or in the ignored `external/`
directory). Do not include it, model weights, or exported models in an anonymous
release. Removing current files does not remove copies from old Git commits.

## Three interchangeable backends

The unified `scripts/train.py`, `scripts/eval.py` and `scripts/check_setup.py`
support `codec.type: mlp`, `gpe` or `torchscript`. All use the same trajectory-level
80:20 split, time-conditioned latent predictor, pixel metrics and fixed-test
comparison GIF/rollout PNG outputs. Only the encoder/decoder backend changes.
Do not reuse a trained predictor across different codecs or latent dimensions.

## Download source and run without pretrained weights

Install the base project requirements first. After reviewing upstream terms:

```bash
git clone https://github.com/wonjunee/GPE_codes.git external/GPE_codes
python -m pip install --no-deps -r requirements-gpe.txt
python scripts/check_setup.py --config configs/acrobot_frames_gpe.yaml --trust-external-code --load-codec
python scripts/train.py --config configs/acrobot_frames_gpe.yaml --trust-external-code
python scripts/eval.py --config configs/acrobot_frames_gpe.yaml --trust-external-code
```

Put trajectories in `data/acrobot_frames/`, or add `--data path/to/frames` to
all commands. Use `--gpe-source path/to/GPE_codes` for a different source folder.
The GPE config is a small first-run configuration: 24 frames per trajectory,
two input frames, six future frames, three codec epochs and three predictor
epochs. Set `dataset.max_frames: null` and choose an explicit training budget
for a full experiment. Outputs use `experiments/acrobot_frames_gpe/`.

This imports the reader's `TransportT` / `TransportG` definitions and initializes
them randomly. Reconstruction MSE is optimized on **training images only**, then
the codec is frozen and the PhiBE predictor is trained. This is an external-GPE
**architecture integration check, not GPE's geometric training objective** and
not paper reproduction. Downloading source does not provide trained Acrobot weights.

The adapter-only optional requirements pin TorchCFM 1.0.7 and POT 0.9.4 because
the upstream architecture module imports TorchCFM even when its flow model is
unused. `--no-deps` avoids replacing the base PyTorch/NumPy environment. This is
not the environment for running every upstream GPE/CFM training script.

## Load original T.pth and S.pth directly

Use matching, trusted Acrobot weights and set these fields in the GPE config:

```yaml
codec:
  type: gpe
  initialization: pretrained
  trust_external_code: false
  source_dir: external/GPE_codes
  source_file: transportmodules/transportsMNIST.py
  encoder_class: TransportT
  decoder_class: TransportG
  encoder_checkpoint: data/acrobot_frames/codec/T.pth
  decoder_checkpoint: data/acrobot_frames/codec/S.pth
```

Keep `model.latent_dim` consistent with those weights (the supplied GPE config
uses 3). Run the same commands with `--trust-external-code`. Reconstruction
pretraining is skipped; the loaded codec remains frozen. Plain tensor state
dictionaries, a `state_dict` wrapper, and a uniform `module.` prefix are supported.
Key and tensor-shape matching is strict: incompatible weights are never partially
loaded or silently replaced with random weights.

Some older Acrobot experiments used a modified decoder called `TransportG_mod`.
Those weights require that exact class in the reader's external source and
`codec.decoder_class: TransportG_mod`. The public upstream default `TransportG`
has a different architecture. This repository does not distribute the modified
implementation or those checkpoints, and cannot promise their public availability.

## Optional TorchScript backend

If independently exported compatible artifacts are available:

```yaml
codec:
  type: torchscript
  initialization: pretrained
  trust_external_code: false
  encoder_checkpoint: data/acrobot_frames/codec/encoder.ts
  decoder_checkpoint: data/acrobot_frames/codec/decoder.ts
```

No GPE source or TorchCFM is needed for this backend. The encoder must map float
`[B,1,H,W]` in [-1,1] to `[B,model.latent_dim]`; the decoder must map back to
finite pixels `[B,1,H,W]` in [-1,1]. The adapter checks batch sizes 1 and 2 before
training. It expects tensors, not tuples or dictionaries. T.pth/S.pth state
dictionaries cannot be converted to TorchScript by renaming the files.

## Switching, checkpoints and trust

- `--codec mlp|gpe|torchscript` overrides the backend, but its YAML paths,
  initialization and latent dimension must still match the selected backend.
- `--trust-external-code` permits Python/TorchScript execution after your review;
  it is not a security sandbox or a substitute for upstream permission.
- Pipeline v2 checkpoints contain both codec and predictor tensors. Restoring
  GPE requires the matching external source module and class names, but no longer
  needs the original separate T.pth/S.pth files.
- TorchScript restoration still needs both exported files. Their SHA-256 values
  are checked against the saved pipeline identity.
- The GPE architecture file SHA-256 and class names are recorded and checked.
  This does not hash imported dependencies; record the complete environment too.
- Existing v1 MLP pipeline checkpoints remain supported. Acrobot loading is a
  warm start, not exact optimizer/RNG resume.

Integration was tested with upstream commit
`3c38a4e00a4bb2de7ec5166c0ebba32a291a770e`; architecture module SHA-256:
`ae9aa1f5ea4f9fde794c096fd8fedb12c8267c72f0e4f71f85c9e7ff229abea7`.
Pin that upstream revision to reproduce this interface check rather than assuming
future upstream changes are compatible. Source import, random initialization,
reconstruction warmup, forecast training and independent evaluation were exercised.
Matching reader-supplied modified-class T/S weights were also exercised locally.
These checks establish interoperability, not public weight provenance or convergence.

The paper-specific geometric codec training recipe and Acrobot FVD remain
unverified/unimplemented here. No future frames are used as predictor inputs;
older whole-trajectory smoothing and forward-difference preprocessing are not
part of this unified image pipeline.
