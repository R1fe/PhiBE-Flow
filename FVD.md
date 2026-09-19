# FVD Reproduction and Reader-Supplied Weights

The paper reports FVD, so reproducing that result is a required release task,
not an optional scientific check. It remains optional to *run* FVD when only
testing installation or computing pixel metrics. No detector is bundled or
automatically downloaded by this project.

## Do not mix these three implementations

1. [Google Research FVD](https://github.com/google-research/google-research/tree/master/frechet_video_distance)
   is a TensorFlow reference implementation. Its source uses TensorFlow Hub's
   `deepmind/i3d-kinetics-400/1` and the `RGB/inception_i3d/Mean:0` embedding.
   Inputs before preprocessing are `[N,T,H,W,3]` in [0,255]. The current
   reference source fixes embedding batches at 16. This is not the backend
   invoked by this repository's `scripts/eval.py`.
2. The author-supplied [OneDrive weight link](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI)
   occurs in the **commented-out** older MCVD loader, which uses
   `InceptionI3d(400)` and `load_state_dict`, not `torch.jit.load`.
   The live download, file hash and actual payload have not been verified.
   Do not rename that file to `i3d_torchscript.pt` and assume compatibility.
3. This repository currently uses a **TorchScript** I3D interface, following
   the [fvd-comparison](https://github.com/universome/fvd-comparison) /
   [StyleGAN-V](https://github.com/cvpr2022-stylegan-v/stylegan-v) route.
   The upstream comparison concerns matched, already-preprocessed videos;
   it does not establish equivalence of arbitrary evaluation protocols.

The I3D architecture/training reference is
[DeepMind Kinetics-I3D](https://github.com/google-deepmind/kinetics-i3d).
Architecture provenance alone does not identify a checkpoint, its feature
layer, its file format or the evaluation protocol used for a paper table.

## Manual setup for the currently implemented backend

The inspected server evaluator uses this
[TorchScript I3D download reference](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt).
Download it yourself after checking the upstream terms, and store it as
`data/kth/i3d_torchscript.pt`. No model is redistributed by this repository.
The download URL was read from the server source; a fresh download has not
been verified. Compare your file against the actual server model:

- Size: 51,235,320 bytes.
- SHA-256: `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`.

The evaluator verifies this checksum **before** loading TorchScript. Do not
replace the checksum merely to silence a mismatch. Inspect your local file:

```bash
python -c "import hashlib; print(hashlib.sha256(open('data/kth/i3d_torchscript.pt','rb').read()).hexdigest())"
```

Set the following in `configs/kth.yaml` only after verifying that model:

```yaml
evaluation:
  fvd: true
  i3d_checkpoint: data/kth/i3d_torchscript.pt
  i3d_sha256: bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2
```

Keep the other evaluation settings. Then run the normal KTH evaluation entry:

```bash
python scripts/eval.py --config configs/kth.yaml
```

The detector is evaluated without training. Current inputs are RGB square
videos `[N,T,3,H,W]` in [-1,1], bilinearly resized to 224 with
`align_corners=False`, then passed in `[N,3,T,224,224]` layout with
`rescale=False, resize=False, return_features=True`. FVD currently includes
the conditioning frames concatenated with future frames. It uses the test
loader's clips, not automatically the official KTH subject split. Preserve
the exact sample list, clip start indices, frame stride, number of samples,
context/future length, EMA selection and detector hash across methods.

## Verified server implementation

Read-only inspection of `evaluate_kth_ours.py` and `models/fvd/fvd.py` confirmed:

- The active loader uses `torch.jit.load`; the OneDrive state-dictionary loader
  is commented out. The implementation refers to MCVD / universome / StyleGAN-V,
  not a live invocation of Google Research's TensorFlow evaluator.
- The real clip contains 40 frames; the generated clip concatenates 10 original
  conditioning frames with 30 predicted frames. FVD features have 400 elements.
- Grayscale is repeated into three RGB channels. FVD inputs are resized with
  bilinear interpolation, shorter side 224 then center crop, `align_corners=False`,
  scaled to [-1,1], and passed with the same detector flags described above.
- The script defaults to seed 0, clip start 0, all videos from the selected test
  shard, one prediction per video, and EMA weights. Pixel metrics use future
  frames only; FVD uses the complete 40-frame clips.
- Top-level FVD compares against original ground-truth pixels. A separately
  named result compares against VQ-VAE reconstructions; these are not interchangeable.

The inspected evaluator source SHA-256 is
`aeda4853b56244b811dee19f74d98ffb2af5877a1d808c5132262cf56c651f41`;
the inspected FVD helper SHA-256 is
`422dc295e7e7bc2b92cb09a55a509eb9e0b90b974f830932d18b69c4a80d34e4`.
No server credentials, absolute paths or model weights are included here.

**Remaining differences:** this project's requested 80:20 re-split is not the
server evaluator's supplied test shard; seed, clip selection and resize-to-64
preprocessing must also match for a numerical comparison. The local Frechet
wrapper clamps tiny negative numerical results to zero, unlike the server.
Verifying source and a detector hash is not a full rerun of paper results.
`eval_metrics.json` records the detector hash and context/future construction
when FVD is enabled. Do not describe this wrapper as executing Google TensorFlow
code or using the OneDrive state dictionary. The paper's implementation wording
should reflect the actual evaluation that produced its tables.

FVD for the Acrobot image-input experiment is not connected yet. In particular,
grayscale-to-RGB handling and conditioning/forecast clip construction must be
specified, not silently borrowed from the KTH protocol.
