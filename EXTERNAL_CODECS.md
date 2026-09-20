# Acrobot Image Codecs

Select `codec.type: mlp`, `gpe` or `torchscript`. All backends share the image
train/eval pipeline. GPE source and weights are not bundled; obtain them from
[upstream](https://github.com/wonjunee/GPE_codes) under its terms.

## Reader-Trained GPE

After installing the base project requirements and reviewing upstream code:

```bash
git clone https://github.com/wonjunee/GPE_codes.git data/acrobot_frames/external/GPE_codes
python -m pip install --no-deps -r requirements-gpe.txt
```

Train a compatible codec in that external project using the Acrobot training
trajectories only. Follow the upstream training procedure and check reconstruction
quality before freezing it. Place the resulting weights at:

```text
data/acrobot_frames/codec/T.pth
data/acrobot_frames/codec/S.pth
```

Then run the interface provided here:

```bash
python scripts/check_setup.py --config configs/acrobot_frames_gpe.yaml --trust-external-code --load-codec
python scripts/train.py --config configs/acrobot_frames_gpe.yaml --trust-external-code
python scripts/eval.py --config configs/acrobot_frames_gpe.yaml --trust-external-code
```

Use `--data path/to/frames` and `--gpe-source path/to/GPE_codes` for other
locations. The configuration uses full trajectories, two input frames and six
future frames. The pretrained codec remains frozen during predictor training.

Optional requirements support importing the architecture, not every upstream
training script. Set class names and latent dimensions to match your trained
codec. Modified GPE source is not distributed here. The separate MLP configuration
trains a lightweight codec locally; it does not replace external GPE training.

## Original T/S Checkpoints

Set the following in the GPE config and run the same commands:

```yaml
codec:
  type: gpe
  initialization: pretrained
  trust_external_code: false
  source_dir: data/acrobot_frames/external/GPE_codes
  source_file: transportmodules/transportsMNIST.py
  encoder_class: TransportT
  decoder_class: TransportG
  encoder_checkpoint: data/acrobot_frames/codec/T.pth
  decoder_checkpoint: data/acrobot_frames/codec/S.pth
```

Match `model.latent_dim` to the weights (the provided config uses 3).
Pretrained codecs are frozen and skip reconstruction training. Plain tensor
state dictionaries, `state_dict` wrappers and uniform `module.` prefixes are
supported; key and shape matching is strict.

Weights trained with `TransportG_mod` require that class in the reader-owned
source and `codec.decoder_class: TransportG_mod`; upstream `TransportG` is not
compatible. Modified source and compatible pretrained Acrobot weights are not
provided by this repository.

## TorchScript

```yaml
codec:
  type: torchscript
  initialization: pretrained
  trust_external_code: false
  encoder_checkpoint: data/acrobot_frames/codec/encoder.ts
  decoder_checkpoint: data/acrobot_frames/codec/decoder.ts
```

No GPE source or TorchCFM is needed. The encoder maps `[B,1,H,W]` in [-1,1]
to `[B,model.latent_dim]`; the decoder returns finite `[B,1,H,W]` pixels in
[-1,1]. Both must return tensors. Renaming T.pth/S.pth does not create TorchScript.

## Compatibility

- `--codec mlp|gpe|torchscript` overrides the backend; paths, initialization and
  latent dimensions must still match. Do not reuse a predictor across codecs.
- `--trust-external-code` permits reviewed Python/TorchScript execution; it is
  not a sandbox or a grant of redistribution permission.
- Pipeline checkpoints contain codec and predictor tensors. GPE restoration
  needs matching source/classes, but not the original separate T/S files.
  TorchScript restoration still needs both exported artifacts.
- Source-module or artifact hashes are checked on restoration. Imported
  dependencies are not hashed. Existing v1 MLP checkpoints remain supported.

The tested upstream revision is `3c38a4e00a4bb2de7ec5166c0ebba32a291a770e`.
Pin it for the same interface; future upstream revisions may differ.
