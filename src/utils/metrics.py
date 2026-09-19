from __future__ import annotations

import numpy as np


def kth_frame_metrics(prediction, truth, horizons):
    """Pixel metrics on [0,1], excluding all conditioning frames.

    Report both exact-step and prefix-mean metrics to make horizons explicit.
    SSIM uses an 11x11 Gaussian window, sigma=1.5, and valid convolution.
    """
    import torch
    from torch.nn import functional as F

    prediction = (prediction.clamp(-1, 1) + 1) / 2
    truth = (truth.clamp(-1, 1) + 1) / 2
    if prediction.shape != truth.shape:
        raise ValueError("KTH prediction and truth shapes differ.")
    b, t, c, h, w = truth.shape
    mse = (prediction-truth).square().mean((2, 3, 4))
    psnr = -10 * mse.clamp_min(1e-12).log10()
    window = min(11, h, w)
    if window % 2 == 0:
        window -= 1
    coords = torch.arange(window, device=truth.device, dtype=truth.dtype) - window // 2
    gaussian = torch.exp(-coords.square() / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(c, 1, -1, -1).contiguous()
    x, y = prediction.reshape(b*t, c, h, w), truth.reshape(b*t, c, h, w)
    def mean(value):
        return F.conv2d(value, kernel, groups=c)
    mx, my = mean(x), mean(y)
    vx, vy, covariance = mean(x*x)-mx*mx, mean(y*y)-my*my, mean(x*y)-mx*my
    ssim = ((2*mx*my+0.01**2)*(2*covariance+0.03**2) /
            ((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))).mean((1, 2, 3)).reshape(b, t)
    result = {}
    for name, values in (("mse", mse), ("psnr", psnr), ("ssim", ssim)):
        result[name] = float(values.mean())
        for horizon in horizons:
            horizon = int(horizon)
            if not 1 <= horizon <= t:
                raise ValueError(f"Metric horizon {horizon} exceeds the {t}-frame rollout.")
            result[f"{name}_at_{horizon}"] = float(values[:, horizon-1].mean())
            result[f"{name}_first_{horizon}"] = float(values[:, :horizon].mean())
    return result


def drift_mse_from_rollout(
    predicted_rollout: np.ndarray,
    ground_truth_rollout: np.ndarray,
) -> float:
    """Mean squared error between predicted and ground-truth state trajectories."""
    length = min(len(predicted_rollout), len(ground_truth_rollout))
    if length == 0:
        raise ValueError("Cannot compute metrics on empty trajectories.")
    diff = np.asarray(predicted_rollout[:length], dtype=np.float64) - np.asarray(
        ground_truth_rollout[:length],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(diff)):
        return float("inf")

    squared_error = np.square(diff, dtype=np.float64)
    if not np.all(np.isfinite(squared_error)):
        return float("inf")
    return float(np.mean(squared_error))
