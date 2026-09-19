import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.datasets.kth import KTHDataset, discover_kth_videos, split_kth_videos
from src.method.loss import kth_velocity_loss
from src.models.kth_predictor import KTHVelocityPredictor, SpatialTransformerLayer
from src.models.vqvae_taming import VQModelInterface
from src.trainers.kth_pipeline import fixed_kth_samples
from src.trainers.kth_trainer import KTHTrainer
from src.utils.config import load_config
from src.utils.metrics import kth_frame_metrics
from src.utils.visualization import save_video_rollout_comparison


ROOT = Path(__file__).resolve().parents[1]
TINY_VQ = dict(embed_dim=2, n_embed=16, double_z=False, z_channels=2, resolution=8,
               in_channels=3, out_ch=3, ch=32, ch_mult=[1, 1], num_res_blocks=1,
               attn_resolutions=[], dropout=0.0)


def create_shard(root):
    path = Path(root) / "shard.hdf5"
    with h5py.File(path, "w") as handle:
        lengths = handle.create_group("len")
        for i in range(5):
            frames = np.stack([np.full((8, 8, 3), t*20+i, np.uint8) for t in range(9)])
            lengths[str(i)] = len(frames)
            if i == 0:
                handle[str(i)] = frames
            elif i == 1:
                handle.create_group(str(i))["frames"] = frames
            else:
                group = handle.create_group(str(i))
                for t, frame in enumerate(frames):
                    group[str(t)] = frame
    return path


def tiny_config(root):
    config = load_config(ROOT / "configs/kth.yaml")
    config.dataset.update(dataset_path=str(root), image_size=8, condition_frames=2,
                          prediction_frames=3, batch_size=2, num_workers=0)
    config.model.update(state_size=2, state_res=[4, 4], inner_dim=16, heads=2,
                        depth=1, mid_depth=1, dropout=0.0)
    config.training.update(epochs=1, max_steps=10, warmup_steps=0, device="cpu")
    config.evaluation.horizons = [1, 3]
    config.vqvae.checkpoint_path = str(Path(root) / "vq.ckpt")
    config.visualization.update(num_fixed_test_samples=1, rollout_plot_frames=5)
    for key in config.paths:
        config.paths[key] = str(Path(root) / key)
    return config


class KTHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(12)

    def test_hdf5_layouts_time_and_disjoint_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = discover_kth_videos(create_shard(tmp))
            train, test = split_kth_videos(records, seed=4)
            self.assertEqual((len(train), len(test)), (4, 1))
            self.assertFalse(set(train) & set(test))
            self.assertEqual((train, test), split_kth_videos(records, seed=4))
            dataset = KTHDataset(records, frames_per_sample=3, frame_stride=2,
                                 image_size=8, frame_time_delta=0.25, time_origin=3)
            for i in range(5):
                frames, times = dataset.get_clip(i, 2)
                self.assertEqual(frames.shape, (3, 3, 8, 8))
                torch.testing.assert_close(times, torch.tensor([3.5, 4.0, 4.5]))
                torch.testing.assert_close(frames[0, 0, 0, 0], torch.tensor((40+i)/127.5-1))
                torch.testing.assert_close(dataset[i][0], dataset[i][0])
            a, b = fixed_kth_samples(dataset, 2, 1), fixed_kth_samples(dataset, 2, 1)
            for first, second in zip(a, b):
                self.assertEqual(first["start_frame"], second["start_frame"])
                torch.testing.assert_close(first["frames"], second["frames"])

    def test_attention_matches_source_and_second_derivatives(self):
        source = nn.TransformerEncoderLayer(16, 2, 64, 0.0, activation="gelu",
                                            norm_first=True, batch_first=True).eval()
        port = SpatialTransformerLayer(16, 2, 64, 0.0, activation="gelu",
                                        norm_first=True, batch_first=True).eval()
        port.load_state_dict(source.state_dict())
        value = torch.randn(2, 4, 16)
        torch.testing.assert_close(port(value), source(value), atol=1e-6, rtol=1e-5)
        value.requires_grad_(True)
        grad = torch.autograd.grad(port(value).square().sum(), value, create_graph=True)[0]
        grad.square().sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_time_changes_output_and_full_loss_backpropagates(self):
        model = KTHVelocityPredictor(2, (2, 2), 16, 1, 1, 2, dropout=0).eval()
        z, ref, target = (torch.randn(2, 2, 2, 2) for _ in range(3))
        self.assertFalse(torch.allclose(model(z, 0.0, ref), model(z, 2.0, ref)))
        loss = kth_velocity_loss(model, z, ref, target, torch.tensor([0.0, 2.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.time_embedding.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        with torch.no_grad():
            metric = kth_velocity_loss(model, z, ref, target, 1.0, create_graph=False)
        self.assertFalse(metric.requires_grad)

    def test_vjp_equals_full_off_diagonal_jacobian(self):
        class Field(nn.Module):
            def __init__(self):
                super().__init__()
                self.matrix = nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))

            def forward(self, z, time, ref):
                return (z.flatten(1) @ self.matrix.T + time[:, None]).reshape_as(z)

        model = Field()
        z = torch.tensor([[[[0.2, 0.3]]]])
        delta = torch.tensor([[1., 2.]])
        time = torch.tensor([4.])
        value = model(z, time, z).flatten(1)
        diffusion = torch.einsum("bi,ij,bj->b", delta, model.matrix, delta)
        expected = (value.square().sum(1) - 2*(value*delta).sum(1)/0.5 - diffusion/0.5).mean()
        actual = kth_velocity_loss(model, z, z, z+delta.reshape_as(z), time, time_delta=0.5)
        torch.testing.assert_close(actual, expected)
        g1 = torch.autograd.grad(actual, model.matrix)[0]
        g2 = torch.autograd.grad(expected, model.matrix)[0]
        torch.testing.assert_close(g1, g2)
        self.assertEqual(float(diffusion), 27.0)

    def test_rollout_time_progression(self):
        class TimeField(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))
                self.times = []

            def forward(self, z, time, ref):
                self.times.append(time.clone())
                return (self.weight * time[:, None, None, None]).expand_as(z)

        class IdentityCodec(nn.Module):
            def encode(self, x):
                return x

            def decode(self, x):
                return x

        with tempfile.TemporaryDirectory() as tmp:
            config = tiny_config(tmp)
            config.dataset.frame_time_delta = 0.1
            model = TimeField()
            trainer = KTHTrainer(model, IdentityCodec(), torch.optim.Adam(model.parameters()),
                                 torch.device("cpu"), config, tmp, logging.getLogger("kth_test"))
            context = torch.zeros(1, 2, 3, 8, 8)
            prediction = trainer.rollout(context, torch.tensor([[0., 0.1]]))
            torch.testing.assert_close(torch.cat(model.times), torch.tensor([0.1, 0.2, 0.3]))
            torch.testing.assert_close(prediction[0, :, 0, 0, 0], torch.tensor([0.01, 0.03, 0.06]))

    def test_vq_checkpoint_and_quantized_decode(self):
        from src.models.vqvae import VQVAE

        with tempfile.TemporaryDirectory() as tmp:
            original = VQModelInterface(TINY_VQ).eval()
            checkpoint = Path(tmp) / "codec.pt"
            torch.save({"state_dict": original.state_dict()}, checkpoint)
            with patch("src.models.vqvae.vq_f8_small_ddconfig", TINY_VQ):
                codec = VQVAE(checkpoint, chunk_size=1)
            codec.train()
            self.assertFalse(codec.training)
            self.assertTrue(all(not p.requires_grad for p in codec.parameters()))
            frames = torch.randn(1, 2, 3, 8, 8)
            latent = codec.encode(frames)
            self.assertEqual(latent.shape, (1, 2, 2, 4, 4))
            torch.testing.assert_close(latent[0], original.encode(frames[0]))
            torch.testing.assert_close(codec.decode(latent)[0], original.decode(latent[0]))
            self.assertFalse(latent.requires_grad)

    def test_metrics_and_multiframe_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = torch.rand(5, 3, 8, 8)*2-1
            values = kth_frame_metrics(truth[None], truth[None], [1, 5])
            self.assertEqual(values["mse"], 0)
            self.assertAlmostEqual(values["ssim"], 1, places=5)
            with self.assertRaises(ValueError):
                kth_frame_metrics(truth[None], truth[None], [6])
            path = Path(tmp) / "comparison"
            save_video_rollout_comparison(truth, truth, torch.arange(5), path, condition_frames=2)
            with Image.open(path.with_suffix(".gif")) as gif:
                self.assertEqual(gif.n_frames, 5)
                self.assertGreater(gif.info["duration"], 0)
            self.assertTrue(path.with_suffix(".png").is_file())

    def test_optional_fvd_preprocessing_and_statistics(self):
        from src.utils.fvd import fvd_features, frechet_distance

        class Detector(nn.Module):
            def forward(self, x, rescale, resize, return_features):
                self.assertions = (x.shape, rescale, resize, return_features)
                return x.mean((2, 3, 4))

        detector = Detector()
        video = torch.zeros(2, 5, 3, 8, 8)
        video[1] = 1
        features = fvd_features(video, detector)
        self.assertEqual(detector.assertions, (torch.Size([2, 3, 5, 224, 224]), False, False, True))
        torch.testing.assert_close(features, torch.tensor([[0., 0., 0.], [1., 1., 1.]]))
        embedding = torch.randn(20, 4)
        self.assertAlmostEqual(frechet_distance(embedding, embedding), 0, places=7)
        with self.assertRaises(ValueError):
            frechet_distance(embedding[:1], embedding[:1])

    def test_reject_legacy_checkpoint_and_changed_time_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = tiny_config(tmp)
            model = KTHVelocityPredictor(**dict(config.model))
            trainer = KTHTrainer(model, nn.Identity(), torch.optim.AdamW(model.parameters()),
                                 torch.device("cpu"), config, tmp, logging.getLogger("kth_test"))
            path = Path(tmp) / "old.pt"
            torch.save({"model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "Legacy KTH"):
                trainer.load(path)
            torch.save({"model_state_dict": model.state_dict(), "time_signature": {}}, path)
            with self.assertRaisesRegex(ValueError, "time grid"):
                trainer.load(path)

    def test_unified_train_eval_resume_and_frozen_mode(self):
        from scripts import train, eval as evaluate

        def plain(value):
            return {key: plain(item) for key, item in value.items()} if isinstance(value, dict) else value

        with tempfile.TemporaryDirectory() as tmp:
            create_shard(tmp)
            config = tiny_config(tmp)
            torch.save({"state_dict": VQModelInterface(TINY_VQ).state_dict()}, config.vqvae.checkpoint_path)
            config_path = Path(tmp) / "kth.yaml"
            config_path.write_text(yaml.safe_dump(plain(config)), encoding="utf-8")
            with patch("src.models.vqvae.vq_f8_small_ddconfig", TINY_VQ):
                with patch.object(sys, "argv", ["train.py", "--config", str(config_path), "--epochs", "1"]):
                    train.main()
                checkpoint = Path(config.paths.checkpoint_dir) / "epoch_001.pt"
                payload = torch.load(checkpoint, weights_only=False)
                self.assertEqual(payload["global_step"], 2)
                self.assertTrue((Path(config.paths.figure_dir)/"epoch_001/sample_00.gif").exists())
                with patch.object(sys, "argv", ["eval.py", "--config", str(config_path)]):
                    evaluate.main()
                metrics = json.loads((Path(config.paths.result_dir)/"eval_metrics.json").read_text())
                self.assertIn("mse_at_3", metrics)
                self.assertIn("evaluation_seconds", metrics)
                config.training.checkpoint_path = str(checkpoint)
                config.training.load_weights_only = True
                config_path.write_text(yaml.safe_dump(plain(config)), encoding="utf-8")
                captured = []
                original_fit = KTHTrainer.fit
                def checked_fit(instance, *args, **kwargs):
                    before = {k: v.clone() for k, v in instance.model.state_dict().items()}
                    ema = {k: v.clone() for k, v in instance.ema.state_dict().items()}
                    result = original_fit(instance, *args, **kwargs)
                    for key, value in before.items():
                        self.assertTrue(torch.equal(value, instance.model.state_dict()[key]))
                        self.assertTrue(torch.equal(ema[key], instance.ema.state_dict()[key]))
                    captured.append(instance.global_step)
                    return result
                with patch.object(KTHTrainer, "fit", checked_fit):
                    with patch.object(sys, "argv", ["train.py", "--config", str(config_path), "--epochs", "1"]):
                        train.main()
                self.assertEqual(captured, [2])
                self.assertEqual(len(list(Path(config.paths.checkpoint_dir).glob("epoch_*.pt"))), 1)
                config.training.load_weights_only = False
                config_path.write_text(yaml.safe_dump(plain(config)), encoding="utf-8")
                with patch.object(sys, "argv", ["train.py", "--config", str(config_path), "--epochs", "1"]):
                    train.main()
                resumed = torch.load(Path(config.paths.checkpoint_dir) / "epoch_002.pt", weights_only=False)
                self.assertEqual(resumed["global_step"], 4)
                self.assertEqual(len(resumed["history"]), 2)


if __name__ == "__main__":
    unittest.main()
