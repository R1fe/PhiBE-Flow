import logging
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.acrobot_angles import AcrobotAnglesDataset
from src.datasets.latent import LatentRolloutDataset
from src.datasets.nse import NSEForecastDataset
from src.method.loss import compute_full_jacobian, nse_velocity_loss, velocity_loss
from src.models.latent_dynamics import LatentResidualDynamics
from src.models.nse_predictor import NSEDriftModel
from src.models.predictor import build_predictor
from src.trainers.latent_trainer import LatentTrainer
from src.trainers.nse_trainer import NSETrainer
from src.trainers.state_trainer import StateTrainer
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def state_model():
    return build_predictor(seq_length=2, feature_dim=4, hidden_dims=[16],
                           num_res_blocks=1, res_block_dim=16, activation="silu")


class TimeField(torch.nn.Module):
    """Analytic dx/dt = t, used to detect wrong RK4 stage times."""
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, context, time, condition=None):
        self.times.append(time.detach().clone())
        shape = (time.shape[0],) + (1,) * (context.ndim - 1)
        if condition is not None:
            return time.reshape(shape).expand_as(context)
        return time[:, None].expand_as(context[:, -1])


class TimeConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)

    def test_acrobot_dataset_time_and_training(self):
        trajectory = {"t": np.arange(8) * 0.25 + 4.0,
                      "y": np.random.default_rng(9).normal(size=(4, 8))}
        dataset = AcrobotAnglesDataset([trajectory], seq_length=2, prediction_horizon=2)
        self.assertEqual(float(dataset[2][2]), 4.75)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        model = state_model()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = StateTrainer(model, torch.optim.Adam(model.parameters()), torch.device("cpu"),
                                   0.25, tmp, tmp, tmp, logging.getLogger("test"),
                                   rollout_loss_weight=1.0, integration_method="rk4")
            loss = trainer._rollout_aware_loss(*batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.input_layer[0].weight.grad[:, -1].abs().sum()), 0)
            metrics = trainer.evaluate_loader(DataLoader(dataset, batch_size=2))
            self.assertTrue(np.isfinite(metrics["drift_mse"]))
            trainer.writer.close()

    def test_same_state_different_times_and_jacobian(self):
        model = state_model()
        state = torch.randn(1, 2, 4).expand(2, -1, -1).clone().requires_grad_(True)
        time = torch.tensor([0.0, 3.0])
        prediction = model(state, time)
        self.assertFalse(torch.allclose(prediction[0], prediction[1]))
        jacobian = compute_full_jacobian(prediction, state, "acrobot_angles")
        self.assertEqual(tuple(jacobian.shape), (2, 4, 4))
        loss = velocity_loss(model, (state, torch.randn(2, 4), time), torch.device("cpu"),
                             include_diffusion=True, dataset_name="acrobot_angles")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(TypeError):
            model(state)

    def test_rk4_time_stages_and_rollout(self):
        trainer = StateTrainer.__new__(StateTrainer)
        trainer.model = TimeField()
        trainer.time_delta = 0.5
        trainer.device = torch.device("cpu")
        trainer.integration_method = "rk4"
        context = torch.zeros(1, 2, 4)
        result = trainer._integrate_next_state(context, torch.tensor([2.0]))
        torch.testing.assert_close(result, torch.full((1, 4), 1.125))
        self.assertEqual([float(t) for t in trainer.model.times], [2.0, 2.25, 2.25, 2.5])
        rollout = trainer.predict_rollout(context, 2, start_time=2.0)
        torch.testing.assert_close(rollout[-1], torch.full((4,), 2.5))

    def test_nse_dataset_loss_and_rollout_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.pt"
            torch.save((torch.randn(1, 12, 8, 8), None), path)
            dataset = NSEForecastDataset([path], 8, 8, 1.0, time_lag=2,
                                         time_delta=0.5, time_origin=3.0)
            self.assertEqual(float(dataset[0][-1]), 3.5)
            self.assertEqual(float(dataset[2][-1]), 4.0)
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            model = NSEDriftModel(dim=8, dim_mults=(1, 2), resnet_block_groups=4,
                                  learned_sinusoidal_dim=8, attention_dim_head=8, attention_heads=1)
            loss = nse_velocity_loss(model, batch, torch.device("cpu"), 0.5)
            loss.backward()
            self.assertGreater(float(model._arch.time_mlp[1].weight.grad.abs().sum()), 0)
            same = torch.randn(1, 1, 8, 8).expand(2, -1, -1, -1)
            output = model(same, torch.tensor([1.0, 2.0]), condition=same)
            self.assertFalse(torch.allclose(output[0], output[1]))
            trainer = NSETrainer(model, torch.optim.Adam(model.parameters()), torch.device("cpu"),
                                 tmp, tmp, tmp, logging.getLogger("test"),
                                 time_delta=0.5, lo_size=8, hi_size=8, time_lag=2, time_origin=3.0)
            trainer.model = TimeField()
            truth, pred = trainer.rollout(dataset.get_trajectory(0), 3)
            self.assertEqual([float(t) for t in trainer.model.times], [3.5, 4.0, 4.5])
            self.assertEqual(truth.shape, pred.shape)

    def test_latent_training_and_time_input(self):
        dataset = LatentRolloutDataset({"a": np.random.default_rng(0).normal(size=(9, 6)).astype("float32")}, 2, 3)
        self.assertEqual(float(dataset[2][-1]), 3.0)
        sequence, future, time = next(iter(DataLoader(dataset, batch_size=2)))
        model = LatentResidualDynamics(2, torch.zeros(6), torch.ones(6), torch.ones(3),
                                      width=16, num_blocks=1)
        # The default zero output head masks time dependence until the first update.
        torch.nn.init.normal_(model.output_head[-1].weight, std=0.1)
        trainer = LatentTrainer.__new__(LatentTrainer)
        trainer.model = model
        trainer.state_std = torch.ones(1, 6)
        trainer.loss_weights = torch.ones(1, 6)
        trainer.rollout_discount = 0.97
        loss = trainer._rollout_loss(sequence, future, time)
        loss.backward()
        self.assertGreater(float(model.input_projection[0].weight.grad[:, -1].abs().sum()), 0)

    def test_checkpoint_roundtrip_and_legacy_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model = state_model()
            save_checkpoint(path, model)
            load_checkpoint(path, state_model())
            state = model.state_dict()
            state["input_layer.0.weight"] = state["input_layer.0.weight"][:, :-1]
            torch.save({"model_state_dict": state}, path)
            with self.assertRaisesRegex(ValueError, "legacy v"):
                load_checkpoint(path, model)

    def test_unified_train_eval_entrypoints(self):
        from scripts import train, eval as evaluate
        root = Path(__file__).resolve().parents[1]
        for name in ("acrobot_angles", "nse"):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                config = yaml.safe_load((root / "configs" / f"{name}.yaml").read_text())
                config["training"].update(device="cpu", epochs=1)
                config["dataset"].update(batch_size=8, num_workers=0)
                config["visualization"].update(num_fixed_samples=1, rollout_steps=2)
                for key in config["paths"]:
                    config["paths"][key] = str(directory / key)
                if name == "acrobot_angles":
                    trajectory = {"t": np.arange(8) * 0.25 + 3.0,
                                  "y": np.random.default_rng(5).normal(size=(4, 8))}
                    data_path = directory / "data.pkl"
                    with data_path.open("wb") as handle:
                        pickle.dump([trajectory] * 5, handle)
                    config["dataset"].update(dataset_path=str(data_path), seq_length=2)
                    config["model"].update(hidden_dims=[16], res_block_dim=16, num_res_blocks=1)
                    config["training"]["time_delta"] = 0.25
                else:
                    data_path = directory / "data"
                    data_path.mkdir()
                    for index in range(5):
                        torch.save((torch.randn(1, 8, 8, 8), None), data_path / f"{index}.pt")
                    config["dataset"].update(dataset_path=str(data_path), lo_size=8, hi_size=8,
                                             time_origin=3.0)
                    config["model"].update(channels=8, dim_mults=[1, 2],
                                           resnet_block_groups=4, learned_sinusoidal_dim=8,
                                           attention_dim_head=8, attention_heads=1)
                    config["training"]["time_delta"] = 0.5
                config_path = directory / "config.yaml"
                config_path.write_text(yaml.safe_dump(config))
                try:
                    trainer_class = StateTrainer if name == "acrobot_angles" else NSETrainer
                    original_evaluate = trainer_class.evaluate_loader
                    evaluated_counts = []
                    def checked_evaluate(instance, loader):
                        count = (len(loader.dataset.rollout_trajectories) if name == "acrobot_angles"
                                 else len(loader.dataset.trajectories))
                        evaluated_counts.append(count)
                        return original_evaluate(instance, loader)
                    with patch.object(trainer_class, "evaluate_loader", checked_evaluate), patch.object(sys, "argv", ["train.py", "--config", str(config_path)]):
                        train.main()
                    with patch.object(trainer_class, "evaluate_loader", checked_evaluate), patch.object(sys, "argv", ["eval.py", "--config", str(config_path)]):
                        evaluate.main()
                    self.assertEqual(evaluated_counts, [4, 1])
                    history = json.loads((directory / "result_dir/train_history.json").read_text())
                    self.assertIn("train_rollout_mse", history)
                    self.assertFalse(any(key.startswith("test_") for key in history))
                    self.assertTrue((directory / "result_dir/fixed_train_samples.json").is_file())
                    self.assertFalse((directory / "checkpoint_dir" / "best.pt").exists())
                    self.assertTrue((directory / "checkpoint_dir" / "epoch_001.pt").exists())
                    self.assertTrue((directory / "result_dir" / "eval_metrics.json").exists())
                    metrics = json.loads((directory / "result_dir" / "eval_metrics.json").read_text())
                    self.assertIn("rollout_mse", metrics)
                    self.assertEqual(metrics["num_rollout_trajectories"], 1)
                    self.assertTrue(list((directory / "figure_dir").rglob("*rollout.png")))
                finally:
                    for logger_name in ("train", "eval", "train_nse", "eval_nse"):
                        logger = logging.getLogger(logger_name)
                        for handler in list(logger.handlers):
                            handler.close()
                            logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
