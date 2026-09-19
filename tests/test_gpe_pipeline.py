"""External-codec integration using independent toy fixtures, never GPE source."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.models.external_codec import build_image_codec
from src.trainers.acrobot_frame_pipeline import AcrobotFrameTrainer, run_acrobot_frames
from src.utils.config import load_config


TOY_SOURCE = '''import torch
from torch import nn
class TransportT(nn.Module):
    def __init__(self, input_shape, zdim):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_shape[1]**2, zdim))
    def forward(self, x):
        return self.net(x)
class TransportG(nn.Module):
    def __init__(self, output_shape, zdim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim, output_shape[1]**2), nn.Tanh(),
                                 nn.Unflatten(1, tuple(output_shape)))
    def forward(self, x):
        return self.net(x)
'''


class GPEPipelineTests(unittest.TestCase):
    def setup_config(self, root):
        project = Path(__file__).resolve().parents[1]
        config = load_config(project / "configs/smoke/acrobot_frames.yaml")
        config.dataset.update(image_size=8, prediction_horizon=2, max_frames=5, stride=1)
        config.model.update(latent_dim=3, width=16)
        config.training.update(codec_epochs=1, epochs=1)
        config.evaluation.horizons = [1, 2]
        config.visualization.enabled = False
        config.paths = {key: str(root / key) for key in config.paths}
        config.dataset.dataset_path = str(root / "frames")
        for index in range(5):
            directory = root / "frames" / f"traj_{index}"
            directory.mkdir(parents=True)
            for frame in range(5):
                Image.fromarray(np.full((8, 8), 20*frame, dtype=np.uint8)).save(directory / f"{frame}.png")
        source = root / "reader_source"
        source.mkdir()
        (source / "toy.py").write_text(TOY_SOURCE)
        config.codec.update(type="gpe", source_dir=str(source), source_file="toy.py",
                            initialization="random", trust_external_code=True)
        return config

    def test_source_and_torchscript_train_eval_frozen_and_identity(self):
        for kind in ("gpe", "torchscript"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.setup_config(root)
                codec, _, _ = build_image_codec(config, root, torch.device("cpu"))
                if kind == "gpe":
                    torch.save(codec.encoder.state_dict(), root / "T.pth")
                    # Support upstream DataParallel state dictionaries, without partial loads.
                    torch.save({"module."+k: v for k, v in codec.decoder.state_dict().items()}, root / "S.pth")
                else:
                    torch.jit.trace(codec.encoder, torch.zeros(2, 1, 8, 8)).save(str(root / "T.pth"))
                    torch.jit.trace(codec.decoder, torch.zeros(2, 3)).save(str(root / "S.pth"))
                config.codec.update(type=kind, initialization="pretrained",
                                    encoder_checkpoint=str(root / "T.pth"), decoder_checkpoint=str(root / "S.pth"))
                args = SimpleNamespace(epochs=1, checkpoint=None)
                with patch.object(AcrobotFrameTrainer, "pretrain_codec", side_effect=AssertionError("Codec must stay frozen")):
                    run_acrobot_frames(config, args, torch.device("cpu"), root)
                last = root / "checkpoint_dir/last.pt"
                payload = torch.load(last, map_location="cpu")
                for key, value in codec.state_dict().items():
                    torch.testing.assert_close(value, payload["model"]["codec."+key], rtol=0, atol=0)
                if kind == "gpe":
                    (root / "T.pth").unlink()
                    (root / "S.pth").unlink()
                run_acrobot_frames(config, args, torch.device("cpu"), root, evaluation=True)
                metrics = json.loads((root / "result_dir/eval_metrics.json").read_text())
                self.assertEqual(metrics["codec_identity"]["type"], kind)
                config.training.update(checkpoint_path=str(last), load_weights_only=True)
                run_acrobot_frames(config, args, torch.device("cpu"), root)
                frozen = torch.load(root / "checkpoint_dir/frozen_epoch_002.pt", map_location="cpu")
                for key, value in payload["model"].items():
                    torch.testing.assert_close(value, frozen["model"][key], rtol=0, atol=0)
                if kind == "gpe":
                    with (root / "reader_source/toy.py").open("a") as handle:
                        handle.write("\n# different reader source version\n")
                    with self.assertRaisesRegex(ValueError, "architecture/time grid"):
                        run_acrobot_frames(config, args, torch.device("cpu"), root, evaluation=True)

    def test_random_source_training_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.setup_config(root)
            config.codec.trust_external_code = False
            with patch("src.models.external_codec._source_module") as importer:
                with self.assertRaisesRegex(ValueError, "trust-external-code"):
                    build_image_codec(config, root, torch.device("cpu"))
                importer.assert_not_called()
            config.codec.trust_external_code = True
            run_acrobot_frames(config, SimpleNamespace(epochs=1), torch.device("cpu"), root)
            self.assertTrue((root / "result_dir/codec_history.json").is_file())
            config.codec.initialization = "pretrained"
            with self.assertRaisesRegex(ValueError, "encoder_checkpoint"):
                build_image_codec(config, root, torch.device("cpu"))
            config.codec.update(encoder_checkpoint=str(root / "wrong.pt"), decoder_checkpoint=str(root / "wrong.pt"))
            torch.save(nn.Linear(9, 2).state_dict(), root / "wrong.pt")
            with self.assertRaisesRegex(ValueError, "do not match"):
                build_image_codec(config, root, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
