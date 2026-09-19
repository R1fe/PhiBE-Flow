"""Reproduce the supplied Acrobot generator recipe, including its solver caveats."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from src.physics import AcrobotSystem


def generate(output, trajectories=10000, duration=8.0, fps=30.0, noise=0.5,
             seed=42, max_rhs_calls=2000000):
    """Preserve legacy RK45/linspace behavior, not an Euler-Maruyama replacement.

    The original did not set a seed. A seed now defines a reproducible new run;
    it cannot recover the exact unpublished trajectories used in a paper.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing Acrobot data.")
    if output.suffix.lower() != ".npz":
        raise ValueError("Use a .npz output path for numeric, pickle-free data.")
    if (trajectories < 2 or not np.isfinite([duration, fps, noise]).all()
            or duration <= 0 or fps <= 0 or noise < 0 or max_rhs_calls < 1):
        raise ValueError("Invalid sample count, time grid, noise or solver budget.")
    frames = int(duration * fps)
    if frames < 3:
        raise ValueError("The legacy time grid needs at least three frames.")
    times = np.linspace(0.0, duration, frames)
    system = AcrobotSystem(noise=0.0)
    states, initials, evaluations = [], [], []
    previous_rng = np.random.get_state()
    try:
        np.random.seed(seed)
        for _ in range(trajectories):
            initial = np.r_[np.random.uniform(-np.pi, np.pi, 2),
                            np.random.uniform(-0.1, 0.1, 2)]
            calls = 0

            def rhs(t, y):
                nonlocal calls
                calls += 1
                if calls > max_rhs_calls:
                    raise RuntimeError("Legacy noisy RK45 exceeded max_rhs_calls; "
                                       "no output was written. Do not silently change the solver.")
                # The source draws four independent normals on EVERY RHS call,
                # including when noise is zero; preserve that RNG consumption.
                return np.asarray(system.dynamics(t, y)) + noise * np.random.randn(4)

            solution = solve_ivp(rhs, (0.0, duration), initial, t_eval=times,
                                 method="RK45", rtol=1e-3, atol=1e-6)
            if (not solution.success or solution.y.shape != (4, frames)
                    or not np.isfinite(solution.y).all()):
                raise RuntimeError("Incomplete or non-finite Acrobot trajectory.")
            states.append(solution.y)
            initials.append(initial)
            evaluations.append(solution.nfev)
    finally:
        np.random.set_state(previous_rng)

    metadata = {
        "recipe": "supplied_cv_dataset_gen", "seed": seed, "noise": noise,
        "num_samples": trajectories, "duration": duration, "fps_argument": fps,
        "saved_frame_dt": float(times[1] - times[0]), "solver": "RK45",
        "rtol": 1e-3, "atol": 1e-6, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "rhs_evaluations": evaluations,
        "warning": "Random RHS evaluations are not a standard SDE discretization; "
                   "exact historical paper data have not been recovered.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, t=times, y=np.stack(states),
                            initial_state=np.stack(initials),
                            metadata_json=json.dumps(metadata, sort_keys=True))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/acrobot_angles/benchmark.npz")
    parser.add_argument("--trajectories", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rhs-calls", type=int, default=2000000)
    args = parser.parse_args()
    print("Legacy noisy-RHS RK45 recipe; potentially slow, not a standard SDE solver.", flush=True)
    report = generate(args.output, args.trajectories, args.duration, args.fps,
                      args.noise, args.seed, args.max_rhs_calls)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
