import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from scripts.generate_acrobot_benchmark import generate
from src.physics import AcrobotSystem


class OriginalGeneratorTests(unittest.TestCase):
    def test_seeded_recipe_matches_direct_legacy_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            previous = np.random.get_state()
            try:
                generate(path, trajectories=2, duration=0.02, fps=200, noise=0.001, seed=7)
                after = np.random.get_state()
                np.testing.assert_array_equal(after[1], previous[1])
                self.assertEqual(after[2:], previous[2:])
                np.random.seed(7)
                times = np.linspace(0, 0.02, 4)
                expected = []
                for _ in range(2):
                    initial = [np.random.uniform(-np.pi, np.pi),
                               np.random.uniform(-np.pi, np.pi),
                               np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
                    result = solve_ivp(AcrobotSystem(noise=0.001).dynamics,
                                       (0, 0.02), initial, t_eval=times)
                    expected.append(result.y)
                with np.load(path, allow_pickle=False) as data:
                    np.testing.assert_array_equal(data["t"], times)
                    np.testing.assert_allclose(data["y"], expected, rtol=1e-12, atol=1e-12)
                    meta = json.loads(str(data["metadata_json"]))
                    self.assertAlmostEqual(meta["saved_frame_dt"], 0.02 / 3)
                    self.assertEqual(meta["seed"], 7)
                    self.assertEqual(meta["solver"], "RK45")
            finally:
                np.random.set_state(previous)

    def test_solver_budget_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            with self.assertRaisesRegex(RuntimeError, "max_rhs_calls"):
                generate(path, trajectories=2, duration=0.02, fps=200,
                         noise=0.001, max_rhs_calls=1)
            self.assertFalse(path.exists())
            path.write_bytes(b"existing data")
            with self.assertRaises(FileExistsError):
                generate(path)
            self.assertEqual(path.read_bytes(), b"existing data")

    def test_zero_noise_still_reproduces_legacy_rng_consumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/"a.npz", Path(tmp)/"b.npz"
            generate(first, trajectories=2, duration=0.02, fps=200, noise=0)
            generate(second, trajectories=2, duration=0.02, fps=200, noise=0)
            with np.load(first) as a, np.load(second) as b:
                np.testing.assert_array_equal(a["y"], b["y"])


if __name__ == "__main__":
    unittest.main()
