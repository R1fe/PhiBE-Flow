"""Optional KTH FVD with a local I3D TorchScript detector (no downloads).

Preprocessing follows the FVD helper credited to fvd-comparison;
Frechet statistics follow its StyleGAN-V implementation.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
import torch
from torch.nn import functional as F


SERVER_I3D_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"


def load_fvd_detector(path, device, expected_sha256=SERVER_I3D_SHA256):
    """Verify a reader-downloaded detector before loading executable TorchScript."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FVD requires local I3D weights: {path}. See FVD.md.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not expected_sha256 or actual != expected_sha256.lower():
        raise ValueError("I3D SHA-256 mismatch. Use the trusted detector documented in FVD.md.")
    detector = torch.jit.load(str(path), map_location=device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector, actual


@torch.no_grad()
def fvd_features(video, detector):
    # [B,T,C,H,W] in [-1,1]; KTH frames are already square center crops.
    b, t, c, h, w = video.shape
    if h != w:
        raise ValueError("FVD expects the square KTH preprocessing output.")
    frames = F.interpolate(video.reshape(b*t, c, h, w), size=(224, 224),
                           mode="bilinear", align_corners=False)
    value = frames.reshape(b, t, c, 224, 224).permute(0, 2, 1, 3, 4).contiguous()
    return detector(value, rescale=False, resize=False, return_features=True)


def frechet_distance(fake, real):
    fake, real = np.asarray(fake, dtype=np.float64), np.asarray(real, dtype=np.float64)
    if min(len(fake), len(real)) < 2:
        raise ValueError("FVD requires at least two real and generated videos.")
    mu_fake, mu_real = fake.mean(0), real.mean(0)
    cov_fake, cov_real = np.cov(fake, rowvar=False), np.cov(real, rowvar=False)
    root = sqrtm(cov_fake @ cov_real)
    distance = np.real(((mu_fake-mu_real)**2).sum() + np.trace(cov_fake+cov_real-2*root))
    if not np.isfinite(distance):
        raise FloatingPointError("FVD covariance computation is non-finite.")
    return float(max(0, distance))
