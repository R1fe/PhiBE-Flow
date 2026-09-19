# External Acrobot Image Codec

GPE source, architecture definitions and weights are **not bundled**. The
previously copied encoder/decoder have been removed from the current tree.
Download upstream yourself from [GPE](https://github.com/wonjunee/GPE_codes)
and read its LICENSE before use. Its terms require prior approval for
redistribution. This project does not grant that approval.

Keep third-party source outside this repository (or in the ignored `external/`
directory). Do not include it, model weights, or exported models in an anonymous
release. Removing current files does not remove copies from old Git commits.

## Reader-supplied interface

The local legacy latent experiments can load frozen **TorchScript** modules
through `src/models/external_codec.py`. They do not reconstruct GPE's architecture
from a state dictionary. Only load models from a trusted source.

- `encoder.ts`: float tensor `[B,1,32,32]` in [-1,1] to `[B,3]`.
- `decoder.ts`: float tensor `[B,3]` to `[B,1,32,32]` in [-1,1].
- Place both in the directory selected by `gpe.weights_dir` for those local
  experiments. `T.pth` / `S.pth` are not TorchScript and cannot simply be renamed.

Train/load a compatible codec in your separately obtained source environment,
then export those modules there with PyTorch's TorchScript tools. The upstream
default model is not claimed to be a drop-in Acrobot model: image dimensions,
normalization, latent size and the training objective must match. No compatible
public Acrobot checkpoint or complete codec training recipe is provided here yet.

## Current image-experiment limitation

`AcrobotFramesDataset` reads images, but the unified `scripts/train.py` and
`scripts/eval.py` do not yet implement the Acrobot image-input experiment.
The local legacy latent scripts use precomputed latent trajectories; they are
not an end-to-end raw-image pipeline and are not included in the release.

The missing integration is: external codec setup/training, causal latent
sequence preparation, time-conditioned prediction, decoding against original
test pixels, fixed-test visualization and the image-experiment FVD protocol.
Whole-trajectory smoothing and forward differences from older preprocessing
must not leak future targets into conditioning inputs. These steps require
verification before claiming reproduction of the paper's image experiment.
