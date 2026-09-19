# KTH FVD Evaluation

FVD uses a reader-supplied TorchScript I3D detector. No weights are bundled or
automatically downloaded. Acrobot image FVD is not implemented.

## Setup

Obtain the compatible [TorchScript I3D model](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt)
under its upstream terms and place it at `data/kth/i3d_torchscript.pt`.
Expected size: 51,235,320 bytes. Expected SHA-256:

`bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`

The hash is checked before loading; do not change it simply to bypass a mismatch.
The URL is the reference used by the original evaluator; a fresh download has
not been independently verified. Set these fields in `configs/kth.yaml`, keeping
the remaining evaluation options:

```yaml
evaluation:
  fvd: true
  i3d_checkpoint: data/kth/i3d_torchscript.pt
  i3d_sha256: bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2
```

```bash
python scripts/eval.py --config configs/kth.yaml
```

## Protocol

The current configuration uses 10 true conditioning frames followed by 30
predicted frames. Ground truth uses the corresponding original 40-frame clip,
not codec reconstructions. Pixel metrics use future frames only.

RGB videos `[N,T,3,H,W]` in [-1,1] are bilinearly resized to 224x224 with
`align_corners=False`, rearranged to `[N,3,T,224,224]`, and passed to the frozen
detector with `rescale=False, resize=False, return_features=True`. Features
have 400 elements. Tiny negative numerical Frechet results are clamped to zero.

Keep sample IDs, clip starts, stride, context/future lengths, sample count, EMA
selection and detector hash fixed across methods. The current 80:20 video split
does not reproduce an external test shard or the official subject split.
`eval_metrics.json` records the detector hash and clip construction.

## Implementation References

This backend follows [fvd-comparison](https://github.com/universome/fvd-comparison)
and [StyleGAN-V](https://github.com/cvpr2022-stylegan-v/stylegan-v). It does not
execute the [Google Research TensorFlow implementation](https://github.com/google-research/google-research/tree/master/frechet_video_distance).
[DeepMind Kinetics-I3D](https://github.com/google-deepmind/kinetics-i3d) provides
the architecture reference. A state-dictionary I3D checkpoint, including the
previously referenced OneDrive weights, is not a drop-in TorchScript replacement.
