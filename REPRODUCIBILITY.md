# Reproducibility Contract

## What Is Tested

- Deterministic trajectory/video splitting without overlapping sample records.
- Explicit current-state time passed through each velocity predictor and rollout.
- Full off-diagonal spatial Jacobian contraction in the KTH VJP objective.
- Source-equivalent unfused transformer attention and second derivatives.
- KTH codec checkpoint loading, frozen inference, quantized decoding and shapes.
- Synthetic HDF5/NPZ/NPY inputs, conversion checks and fixed comparison GIF frames.
- Tiny CPU training/evaluation, checkpoint loading, KTH resume and no-update mode.
- A clean source export independent of sibling development directories.
- External Acrobot codec source/state-dictionary/TorchScript switching, strict
  identity checks, frozen parameters and standalone checkpoint evaluation.

The test suite deliberately uses small synthetic data/models. Real KTH codec
weights can be checked separately with `check_setup.py --load-codec`. Passing
tests does not demonstrate convergence, match a paper table or validate an
optional FVD detector.

## Experiment Protocol

Choose a seed, training budget, data version, normalization and metric horizon
before inspecting test results. Save the executed configuration and command with
the experiment. CLI batch/device/worker overrides do not change the stored YAML;
record those overrides explicitly. No separate validation split is created.

Default evaluation uses `last.pt`. `best.pt` is retained as a diagnostic artifact
selected on test error and must not be presented as an unbiased held-out result.
Per-epoch test plots are intended for debugging; repeatedly tuning to them is
also test-set adaptation. Independent final evaluation requires a separately
fixed protocol/data split outside this development loop.

KTH test clips and visualization samples are deterministic for the same shard
ordering, lengths and seed. Replacing shards or changing prediction length can
change eligibility and split membership. Preserve split/sample manifests with
the run; do not upload them unreviewed if local paths are present.

NSE train/test splitting is by sorted file order, not randomly by frame. Shards
must have disjoint trajectories. Duplicate data files or correlated repeated
simulations can invalidate a held-out evaluation even when file IDs differ.

## Numerical Conventions

KTH: `t = time_origin + original_frame_index * frame_time_delta`; prediction
step `dt = frame_stride * frame_time_delta`. The default dt is one frame unit,
not one second. Time is held fixed when differentiating spatial coordinates.
Loss is `||v||^2 - 2*delta.v/dt - delta^T J_v delta/dt`; it need not be positive.

NSE: the active loss omits the diffusion/Jacobian term and includes the data-only
drift-square term. Numeric evaluation currently reports one-step pixel MSE, not
the plotted multi-step horizon. Acrobot uses its separate state-space objective.

KTH MSE, PSNR and SSIM use pixels in [0,1] and omit all conditioning frames.
`*_at_h` reports frame h; `*_first_h` averages frames 1..h. SSIM uses a valid
11x11 Gaussian window (sigma 1.5). PSNR clamps exact-zero MSE to 120 dB. FVD,
when enabled, includes the true conditioning prefix and generated suffix.

## Known Limits

No full-scale convergence or paper-table reproduction is asserted by this
artifact. Exact NSE/Acrobot benchmark assets are pending. Acrobot frames has a
from-scratch MLP and reader-supplied GPE/TorchScript codec backends, not a verified
reproduction of the paper's GPE experiment. Its PhiBE loss trains a predictor on
frozen latents; optional codec reconstruction uses only training frames and does
not implement GPE's geometric objective. Actual upstream source was exercised
separately; see EXTERNAL_CODECS.md for the revision and interface contract.
No Acrobot FVD result is claimed. KTH CPU integration tests use reduced
predictors; full reference settings require substantially more memory/compute.

KTH restores optimizer, EMA, epoch, global step and RNG states. Resume after a
partial epoch is not guaranteed sample-exact, and device/worker/library changes
can change numerical behavior. Acrobot/NSE are warm-start loaders, not exact
resume implementations. No cross-platform bitwise reproducibility is promised.

The tested dependency profile is deliberately conservative. A fresh supported
environment and the optional real-data smoke checks should be run before making
a public reproducibility claim. Check the workflow result for the exact release
commit; passing CPU tests does not establish benchmark accuracy.
