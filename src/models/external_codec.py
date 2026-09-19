"""Load reader-supplied frozen codecs without redistributing their implementation."""

from pathlib import Path
import importlib.util
import sys

import torch
from torch import nn

from src.utils.config import resolve_path
from src.utils.download import sha256_file


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


class ExternalImageCodec(nn.Module):
    """Interface adapter only: network definitions remain in reader-owned files."""

    def __init__(self, encoder, decoder, image_size, latent_dim):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.image_size, self.latent_dim = image_size, latent_dim

    def encode(self, frames):
        value = self.encoder(frames)
        if not isinstance(value, torch.Tensor) or value.shape != (len(frames), self.latent_dim):
            raise ValueError("External encoder must return [B, model.latent_dim].")
        if not torch.isfinite(value).all():
            raise ValueError("External encoder returned non-finite values.")
        return value

    def decode(self, latent):
        value = self.decoder(latent)
        expected = (len(latent), 1, self.image_size, self.image_size)
        if not isinstance(value, torch.Tensor) or value.shape != expected:
            raise ValueError(f"External decoder must return {expected}; check image size/channels.")
        if not torch.isfinite(value).all() or value.min() < -1.0001 or value.max() > 1.0001:
            raise ValueError("External decoder must return finite pixels in [-1,1].")
        return value

    def forward(self, frames):
        return self.decode(self.encode(frames))


def _source_module(directory, relative_file):
    directory = Path(directory).resolve()
    path = (directory / relative_file).resolve()
    if directory not in path.parents or not path.is_file():
        raise FileNotFoundError("GPE source_file must name a file inside codec.source_dir. See EXTERNAL_CODECS.md.")
    digest = sha256_file(path)
    spec = importlib.util.spec_from_file_location("_reader_gpe_" + digest, path)
    module = importlib.util.module_from_spec(spec)
    # Import the architecture module, never the upstream training entrypoints.
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        raise RuntimeError(f"External GPE dependency missing: {error.name}. Install upstream dependencies; see EXTERNAL_CODECS.md.") from error
    finally:
        sys.dont_write_bytecode = old
    return module, digest


def _load_state(module, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(isinstance(v, torch.Tensor) for v in payload.values()):
        raise ValueError("Expected a tensor state_dict in the external codec checkpoint.")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key[7:]: value for key, value in payload.items()}
    try:
        module.load_state_dict(payload, strict=True)
    except RuntimeError as error:
        raise ValueError("External weights do not match the selected GPE class/latent size. No partial loading is performed.") from error


def build_image_codec(config, root, device, restoring=False):
    """Return codec, checkpoint identity, and whether reconstruction pretraining is needed."""
    from src.models.acrobot_frame_model import FrameAutoencoder

    options = config.get("codec", {})
    kind = options.get("type", "mlp")
    size, dim = int(config.dataset.image_size), int(config.model.latent_dim)
    if kind == "mlp":
        return FrameAutoencoder(size, dim, config.model.width).to(device), {"type": "mlp"}, True
    if kind not in {"gpe", "torchscript"}:
        raise ValueError("codec.type must be mlp, gpe or torchscript.")
    if not options.get("trust_external_code", False):
        raise ValueError("External Python/TorchScript executes code. Review it, then pass --trust-external-code.")
    initialization = options.get("initialization", "pretrained")
    if initialization not in {"random", "pretrained"}:
        raise ValueError("codec.initialization must be random or pretrained.")
    identity = {"type": kind}
    if kind == "gpe":
        module, digest = _source_module(resolve_path(root, options["source_dir"]),
                                         options.get("source_file", "transportmodules/transportsMNIST.py"))
        encoder_name = options.get("encoder_class", "TransportT")
        decoder_name = options.get("decoder_class", "TransportG")
        try:
            encoder = getattr(module, encoder_name)(input_shape=[1, size, size], zdim=dim)
            decoder = getattr(module, decoder_name)(output_shape=[1, size, size], zdim=dim)
        except AttributeError as error:
            raise ValueError("Selected GPE class is missing from the external module.") from error
        identity.update(source_sha256=digest, encoder_class=encoder_name, decoder_class=decoder_name)
        if initialization == "pretrained" and not restoring:
            for role, network in (("encoder", encoder), ("decoder", decoder)):
                checkpoint = options.get(role + "_checkpoint")
                if not checkpoint:
                    raise ValueError(f"Pretrained GPE requires codec.{role}_checkpoint; random mode must be explicitly selected.")
                _load_state(network, resolve_path(root, checkpoint))
    else:
        if initialization != "pretrained":
            raise ValueError("TorchScript codecs are frozen/pretrained only.")
        encoder_path = resolve_path(root, options["encoder_checkpoint"])
        decoder_path = resolve_path(root, options["decoder_checkpoint"])
        encoder = load_external_codec(encoder_path, device)
        decoder = load_external_codec(decoder_path, device)
        identity.update(encoder_sha256=sha256_file(encoder_path), decoder_sha256=sha256_file(decoder_path))
    codec = ExternalImageCodec(encoder, decoder, size, dim).to(device)
    needs_pretraining = kind == "gpe" and initialization == "random"
    codec.eval()
    with torch.no_grad():
        for batch_size in (1, 2):
            codec(torch.zeros(batch_size, 1, size, size, device=device))
    if not needs_pretraining or restoring:
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
    return codec, identity, needs_pretraining
