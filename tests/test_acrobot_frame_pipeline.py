import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from scripts import train, eval as evaluate
from src.models.acrobot_frame_model import AcrobotFrameModel
from src.trainers.acrobot_frame_pipeline import AcrobotFrameTrainer


class ImagePipelineTests(unittest.TestCase):
    def test_time_and_autoregressive_rollout(self):
        torch.manual_seed(9)
        model = AcrobotFrameModel(image_size=8, latent_dim=3, width=16)
        context = torch.randn(2, 2, 1, 8, 8)
        latent = model.encode_sequence(context)
        self.assertFalse(torch.allclose(model.velocity(latent, 0.), model.velocity(latent, 5.)))
        observed = []
        hook = model.velocity.register_forward_pre_hook(
            lambda module, args: observed.append(args[1].detach().clone()))
        prediction = model.rollout(context, torch.tensor([1., 2.]), 3, 0.5)
        hook.remove()
        self.assertEqual(prediction.shape, (2, 3, 1, 8, 8))
        for index, time in enumerate(observed):
            torch.testing.assert_close(time, torch.tensor([1., 2.]) + 0.5*index)

    def test_unified_from_scratch_train_eval_and_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "images"
            for trajectory in range(5):
                folder = data / f"traj_{trajectory}"
                folder.mkdir(parents=True)
                for frame in range(7):
                    pixels = np.full((8, 8), 240, dtype=np.uint8)
                    pixels[frame % 8, (frame+trajectory) % 8] = 0
                    Image.fromarray(pixels).save(folder / f"frame_{frame:03d}.png")
            project = Path(__file__).resolve().parents[1]
            config = yaml.safe_load((project / "configs/smoke/acrobot_frames.yaml").read_text())
            config["dataset"].update(dataset_path=str(data), image_size=8, prediction_horizon=3,
                                     max_frames=None, stride=1, batch_size=4)
            config["model"].update(latent_dim=3, width=16)
            config["training"].update(codec_epochs=1, epochs=2)
            config["evaluation"]["horizons"] = [1, 3]
            config["paths"] = {key: str(root / key) for key in config["paths"]}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            evaluated_splits = []
            original_evaluate = AcrobotFrameTrainer.evaluate
            def checked_evaluate(instance, loader):
                evaluated_splits.append(set(loader.dataset.trajectory_names))
                self.assertFalse(loader.drop_last)
                return original_evaluate(instance, loader)
            with patch.object(AcrobotFrameTrainer, "evaluate", checked_evaluate), patch.object(sys, "argv", ["train.py", "--config", str(path)]):
                train.main()
            checkpoint_dir = root / "checkpoint_dir"
            first = torch.load(checkpoint_dir / "epoch_001.pt", map_location="cpu")
            final = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
            codec_before = torch.load(checkpoint_dir / "codec_initialized.pt", map_location="cpu")
            self.assertTrue(any(not torch.equal(first["model"][k], final["model"][k])
                                for k in first["model"] if k.startswith("velocity.")))
            for key in final["model"]:
                if key.startswith("codec."):
                    torch.testing.assert_close(codec_before["model"][key], final["model"][key], rtol=0, atol=0)
            manifest = (root / "result_dir/fixed_train_samples.json").read_text()
            split = json.loads((root / "result_dir/split_manifest.json").read_text())
            self.assertEqual((len(split["train"]), len(split["test"])), (4, 1))
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertEqual(evaluated_splits, [set(split["train"])] * 2)
            self.assertTrue(all(sample["trajectory"] in split["train"] for sample in json.loads(manifest)))
            history = json.loads((root / "result_dir/history.json").read_text())
            self.assertTrue(all(row["rollout_split"] == "train" and "train_mse" in row for row in history))
            for epoch in (1, 2):
                figure = root / f"figure_dir/epoch_{epoch:03d}/sample_00"
                self.assertTrue(figure.with_suffix(".png").is_file())
                with Image.open(figure.with_suffix(".gif")) as gif:
                    self.assertEqual(gif.n_frames, 5)
            with patch.object(AcrobotFrameTrainer, "evaluate", checked_evaluate), patch.object(sys, "argv", ["eval.py", "--config", str(path)]):
                evaluate.main()
            self.assertEqual(evaluated_splits[-1], set(split["test"]))
            test_manifest = json.loads((root / "result_dir/eval_fixed_test_samples.json").read_text())
            self.assertTrue(all(sample["trajectory"] in split["test"] for sample in test_manifest))
            metrics = json.loads((root / "result_dir/eval_metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["mse"]))
            self.assertIn("mse_at_3", metrics)
            config["training"].update(load_weights_only=True,
                                       checkpoint_path=str(checkpoint_dir / "last.pt"), epochs=1)
            path.write_text(yaml.safe_dump(config))
            with patch.object(sys, "argv", ["train.py", "--config", str(path)]):
                train.main()
            frozen = torch.load(checkpoint_dir / "frozen_epoch_003.pt", map_location="cpu")
            for key in final["model"]:
                torch.testing.assert_close(final["model"][key], frozen["model"][key], rtol=0, atol=0)
            self.assertEqual(manifest, (root / "result_dir/fixed_train_samples.json").read_text())


if __name__ == "__main__":
    unittest.main()
