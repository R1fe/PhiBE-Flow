from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.datasets.acrobot_frames import AcrobotFramesDataset, build_acrobot_frame_datasets
from src.utils.config import load_config


class AcrobotFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for trajectory in range(5):
            folder = self.root / f"traj_{trajectory:03d}"
            folder.mkdir()
            for index in range(12):
                # Unpadded names detect accidental lexicographic frame ordering.
                Image.fromarray(np.full((12, 20, 3), index * 20, dtype=np.uint8)).save(
                    folder / f"frame_{index}.png"
                )

    def tearDown(self):
        self.temp.cleanup()

    def test_window_order_normalization_and_time(self):
        dataset = AcrobotFramesDataset(self.root, seq_length=2, prediction_horizon=2,
                                       time_lag=2, time_delta=0.1, time_origin=3.0)
        sequence, target, time = dataset[1]
        self.assertEqual(tuple(sequence.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(target.shape), (2, 1, 32, 32))
        self.assertAlmostEqual(float(time), 3.3, places=5)
        expected = torch.tensor([20, 60, 100, 140]) / 255 * 2 - 1
        actual = torch.cat((sequence, target))[:, 0, 0, 0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(dataset), 5 * 6)
        images = AcrobotFramesDataset(self.root, mode="image", normalize=False)
        self.assertAlmostEqual(float(images[10][0][0, 0, 0]), 200 / 255, places=6)

    def test_modes_and_preload(self):
        pairs = AcrobotFramesDataset(self.root, mode="pair", time_lag=2, max_frames=4)
        first, second, time = pairs[0]
        self.assertEqual(tuple(first.shape), (1, 32, 32))
        self.assertGreater(float(second.mean()), float(first.mean()))
        self.assertEqual(float(time), 0.0)
        self.assertEqual(len(pairs), 10)
        sequence = AcrobotFramesDataset(self.root, mode="trajectory", max_frames=3,
                                        preload=True, normalize=False)
        frames, times = sequence[0]
        self.assertEqual(tuple(frames.shape), (3, 1, 32, 32))
        torch.testing.assert_close(times, torch.arange(3, dtype=torch.float32))
        frames.fill_(100)
        self.assertLessEqual(float(sequence[0][0].max()), 1.0)

    def test_config_split_and_dataloader(self):
        project = Path(__file__).resolve().parents[1]
        config = load_config(project / "configs" / "acrobot_frames.yaml")
        # Resolve a relative dataset path against a different project root.
        config.dataset.dataset_path = "."
        train, test = build_acrobot_frame_datasets(config, project_root=self.root)
        train_again, test_again = build_acrobot_frame_datasets(config, project_root=self.root)
        self.assertEqual(len(train.trajectory_names), 4)
        self.assertEqual(len(test.trajectory_names), 1)
        self.assertFalse(set(train.trajectory_names) & set(test.trajectory_names))
        self.assertEqual(train.trajectory_names, train_again.trajectory_names)
        self.assertEqual(test.trajectory_names, test_again.trajectory_names)
        sequence, future, times = next(iter(DataLoader(train, batch_size=2)))
        self.assertEqual(tuple(sequence.shape), (2, 2, 1, 32, 32))
        self.assertEqual(tuple(future.shape), (2, 1, 1, 32, 32))
        self.assertEqual(tuple(times.shape), (2,))

    def test_invalid_windows_and_corrupt_frames_fail_explicitly(self):
        with self.assertRaises(ValueError):
            AcrobotFramesDataset(self.root, seq_length=20)
        with self.assertRaises(ValueError):
            AcrobotFramesDataset(self.root, trajectory_names=["missing"])
        path = self.root / "traj_000" / "frame_0.png"
        path.write_bytes(b"not an image")
        dataset = AcrobotFramesDataset(self.root, mode="image")
        with self.assertRaises(OSError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
