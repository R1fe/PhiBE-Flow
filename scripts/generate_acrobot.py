"""Generate deterministic Acrobot demonstration data, not a paper benchmark."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp
from src.physics import AcrobotSystem


def generate(output, trajectories=20, frames=60, fps=30.0, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing Acrobot data.")
    if trajectories < 2 or frames < 3 or fps <= 0:
        raise ValueError("Need trajectories>=2, frames>=3 and fps>0.")
    rng = np.random.default_rng(seed)
    system = AcrobotSystem(noise=0)
    # Exact fixed frame spacing, unlike linspace(0, duration, duration*fps).
    times = np.arange(frames, dtype=np.float64) / fps
    states = []
    for _ in range(trajectories):
        initial = np.r_[rng.uniform(-np.pi, np.pi, 2), rng.uniform(-0.1, 0.1, 2)]
        solution = solve_ivp(system.dynamics, (0, times[-1]), initial, t_eval=times,
                             rtol=1e-8, atol=1e-10)
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError("Acrobot integration failed.")
        states.append(solution.y)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, t=times, y=np.stack(states).astype(np.float32),
                            seed=seed, noise=0.0, purpose="deterministic_demo_not_benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/acrobot_angles/demo.npz")
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.trajectories, args.frames, seed=args.seed)
    print("Generated deterministic demonstration data. Do not report it as the original benchmark.")


if __name__ == "__main__":
    main()
