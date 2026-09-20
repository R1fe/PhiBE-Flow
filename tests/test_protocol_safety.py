import logging
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np
import torch
from torch import nn

from src.datasets.kth import build_kth_datasets, OFFICIAL_TRAIN, OFFICIAL_TEST
from src.method.loss import nse_velocity_loss, velocity_loss
from src.models.kth_predictor import KTHVelocityPredictor
from src.trainers.kth_trainer import KTHTrainer
from src.utils.config import load_config
from src.utils.metrics import drift_mse_from_rollout

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, official=True):
    path = Path(root) / "videos.hdf5"
    with h5py.File(path, "w") as handle:
        for i in range(25 if official else 10):
            length = 45 if i % 3 == 0 else 90
            store = handle.create_dataset(str(i), shape=(length, 8, 8, 3), dtype="u1")
            if official:
                store.attrs["source"] = f"person{i+1:02d}_walking_d1_uncomp.avi"
                store.attrs["subject_id"] = i + 1
    config = load_config(ROOT / "configs/kth.yaml")
    config.dataset.update(dataset_path=str(path), condition_frames=10, prediction_frames=30,
                          image_size=8, split_protocol="official_subjects" if official else "random_video")
    config.model.update(state_size=2, state_res=[2, 2], inner_dim=16, depth=1, mid_depth=1, heads=2)
    for key in config.paths:
        config.paths[key] = str(Path(root) / key)
    return path, config


class ProtocolSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_subject_and_random_splits_are_fixed_before_horizon_filter(self):
        for official in (True, False):
            with self.subTest(official=official), tempfile.TemporaryDirectory() as tmp:
                path, config = fixture(tmp, official)
                train, test = build_kth_datasets(config, tmp)
                train_ids, test_ids = {r.key for r in train.records}, {r.key for r in test.records}
                if official:
                    self.assertEqual({r.subject for r in train.records}, set(OFFICIAL_TRAIN))
                    self.assertEqual({r.subject for r in test.records}, set(OFFICIAL_TEST))
                config.dataset.prediction_frames = 70
                long_train, long_test = build_kth_datasets(config, tmp)
                self.assertTrue({r.key for r in long_train.records} <= train_ids)
                self.assertTrue({r.key for r in long_test.records} <= test_ids)
                self.assertFalse(train_ids & {r.key for r in long_test.records})
                self.assertEqual(train.data_identity, long_train.data_identity)
                self.assertEqual(test.data_identity, long_test.data_identity)

    def test_official_split_requires_subject_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config = fixture(tmp, official=False)
            config.dataset.split_protocol = "official_subjects"
            with self.assertRaisesRegex(ValueError, "subject metadata"):
                build_kth_datasets(config, tmp)

    def test_checkpoint_checks_data_content_split_and_missing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config = fixture(tmp)
            train, _ = build_kth_datasets(config, tmp)
            model = KTHVelocityPredictor(**dict(config.model))
            trainer = KTHTrainer(model, nn.Identity(), torch.optim.Adam(model.parameters()),
                                 torch.device("cpu"), config, tmp, logging.getLogger("protocol"),
                                 data_identity=train.data_identity)
            checkpoint = Path(tmp) / "predictor.pt"
            payload = dict(model_state_dict=model.state_dict(), time_signature=trainer.time_signature(),
                           data_identity=train.data_identity)
            torch.save(payload, checkpoint)
            trainer.load(checkpoint, resume=False)
            config.dataset.prediction_frames = 70
            longer, _ = build_kth_datasets(config, tmp)
            trainer.data_identity = longer.data_identity
            trainer.load(checkpoint, resume=False)
            config.dataset.merge_official_validation = True
            changed_split, _ = build_kth_datasets(config, tmp)
            trainer.data_identity = changed_split.data_identity
            with self.assertRaisesRegex(ValueError, "dataset/split"):
                trainer.load(checkpoint, resume=False)
            config.dataset.merge_official_validation = False
            with h5py.File(path, "r+") as handle:
                handle["1"][0, 0, 0, 0] = 1
            changed_data, _ = build_kth_datasets(config, tmp)
            self.assertNotEqual(changed_data.data_identity["dataset_sha256"], train.data_identity["dataset_sha256"])
            trainer.data_identity = changed_data.data_identity
            with self.assertRaisesRegex(ValueError, "dataset/split"):
                trainer.load(checkpoint, resume=False)
            trainer.data_identity = train.data_identity
            del payload["data_identity"]
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "dataset/split"):
                trainer.load(checkpoint, resume=False)

    def test_rollout_mse_excludes_context_and_rejects_empty_future(self):
        truth = np.zeros((6, 4))
        prediction = truth.copy()
        prediction[4:] = 3
        self.assertEqual(drift_mse_from_rollout(prediction, truth, condition_frames=4), 9)
        self.assertEqual(drift_mse_from_rollout(prediction[4:], truth[4:]), 9)
        with self.assertRaises(ValueError):
            drift_mse_from_rollout(prediction, truth, condition_frames=6)

    def test_nse_diffusion_includes_cross_pixel_terms_and_parameter_gradient(self):
        class Field(nn.Module):
            def __init__(self):
                super().__init__()
                self.matrix = nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))

            def forward(self, x, time, condition=None):
                return (x.flatten(1) @ self.matrix.T + time[:, None]).reshape_as(x)

        model = Field()
        x = torch.tensor([[[[0.2, 0.3]]]])
        delta = torch.tensor([[1., 2.]])
        time = torch.tensor([4.])
        value = model(x, time).flatten(1)
        expected = ((value-delta/0.5).square().sum(1) -
                    torch.einsum("bi,ij,bj->b", delta, model.matrix, delta)/0.5).mean()
        batch = (x, x, x+delta.reshape_as(x), time)
        actual = nse_velocity_loss(model, batch, torch.device("cpu"), 0.5)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, model.matrix)[0],
                                   torch.autograd.grad(expected, model.matrix)[0])
        with torch.no_grad():
            value = nse_velocity_loss(model, batch, torch.device("cpu"), 0.5, create_graph=False)
        self.assertFalse(value.requires_grad)

    def test_state_and_image_latent_diffusion_enabled_by_default(self):
        class Field(nn.Module):
            def __init__(self):
                super().__init__()
                self.matrix = nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))

            def forward(self, x, time):
                return x[:, -1] @ self.matrix.T

        for dataset in ("acrobot_angles", "acrobot_frames"):
            model = Field()
            x = torch.tensor([[[0.2, 0.3]]])
            delta = torch.tensor([[1., 2.]])
            batch = (x, x[:, -1]+delta, torch.tensor([0.]))
            full = velocity_loss(model, batch, torch.device("cpu"), 0.5, dataset)
            drift_only = velocity_loss(model, batch, torch.device("cpu"), 0.5, dataset, False)
            torch.testing.assert_close(drift_only-full, torch.tensor(54.))


if __name__ == "__main__":
    unittest.main()
