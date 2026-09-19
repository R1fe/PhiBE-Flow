"""Load reader-supplied frozen codecs without redistributing their implementation."""

from pathlib import Path

import torch


def load_external_codec(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"External codec not found: {path}. GPE code/weights are not bundled. "
            "See EXTERNAL_CODECS.md; supply a trusted TorchScript module, not T.pth/S.pth."
        )
    model = torch.jit.load(str(path), map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
