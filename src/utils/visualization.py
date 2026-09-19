from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def inverse_angle_transform(features: np.ndarray) -> np.ndarray:
    """Convert [cos, sin, cos, sin, vel, vel] features back to [theta1, theta2, vel, vel]."""
    angles = np.zeros((len(features), 4), dtype=np.float32)
    angles[:, 0] = np.arctan2(features[:, 1], features[:, 0])
    angles[:, 1] = np.arctan2(features[:, 3], features[:, 2])
    angles[:, 2:] = features[:, 4:]
    return angles


def visualize_acrobot_prediction(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    output_dir: str | Path,
    tag: str,
    transform_angles: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if transform_angles:
        true_traj = inverse_angle_transform(true_traj)
        pred_traj = inverse_angle_transform(pred_traj)

    length = min(len(true_traj), len(pred_traj))
    true_traj = true_traj[:length]
    pred_traj = pred_traj[:length]

    trajectory_path = plot_joint_trajectories(true_traj, pred_traj, output_dir, tag)
    state_paths = plot_individual_state_trajectories(
        true_traj,
        pred_traj,
        output_dir,
        tag,
    )
    phase_path = plot_phase_space(true_traj, pred_traj, output_dir, tag)
    rollout_path = save_rollout_comparison(true_traj, pred_traj, output_dir, tag)
    gif_path = save_acrobot_animation(true_traj, pred_traj, output_dir, tag)

    return {
        "trajectory_plot": trajectory_path,
        "state_plots": state_paths,
        "phase_plot": phase_path,
        "rollout_plot": rollout_path,
        "animation": gif_path,
    }


def plot_individual_state_trajectories(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    output_dir: str | Path,
    tag: str,
    context_steps: int = 16,
) -> list[Path]:
    """Save four publication-style state comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    length = min(len(true_traj), len(pred_traj))
    true_traj = np.asarray(true_traj[:length])
    pred_traj = np.asarray(pred_traj[:length])
    time = np.arange(length)

    states = [
        ("theta1", r"Joint 1 angle  $\theta_1$", "Angle (rad)"),
        ("theta2", r"Joint 2 angle  $\theta_2$", "Angle (rad)"),
        ("theta1_dot", r"Joint 1 angular velocity  $\dot{\theta}_1$", "Angular velocity (rad/s)"),
        ("theta2_dot", r"Joint 2 angular velocity  $\dot{\theta}_2$", "Angular velocity (rad/s)"),
    ]
    truth_color = "#2563EB"
    prediction_color = "#F04452"
    grid_color = "#D8DEE9"
    output_paths: list[Path] = []

    for index, (slug, title, ylabel) in enumerate(states):
        error = pred_traj[:, index] - true_traj[:, index]
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))

        fig, axis = plt.subplots(figsize=(11.5, 6.2), facecolor="#F6F8FC")
        axis.set_facecolor("#FFFFFF")
        axis.axvspan(
            0,
            max(context_steps - 1, 0),
            color="#E8EEF9",
            alpha=0.75,
            linewidth=0,
            label="Observed context",
        )
        if 0 < context_steps < length:
            axis.axvline(
                context_steps - 1,
                color="#7B8794",
                linewidth=1.3,
                linestyle=(0, (3, 4)),
                alpha=0.9,
            )
            axis.text(
                context_steps + 2,
                0.035,
                "free prediction →",
                transform=axis.get_xaxis_transform(),
                color="#667085",
                fontsize=10,
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )

        axis.plot(
            time,
            true_traj[:, index],
            color=truth_color,
            linewidth=2.8,
            label="Ground truth",
            zorder=3,
        )
        axis.plot(
            time,
            pred_traj[:, index],
            color=prediction_color,
            linewidth=2.35,
            linestyle=(0, (5, 3)),
            label="Prediction",
            zorder=4,
        )
        axis.fill_between(
            time,
            true_traj[:, index],
            pred_traj[:, index],
            color=prediction_color,
            alpha=0.075,
            linewidth=0,
            zorder=2,
        )

        axis.set_xlabel("Time step", fontsize=12, color="#344054", labelpad=9)
        axis.set_ylabel(ylabel, fontsize=12, color="#344054", labelpad=9)
        axis.grid(True, which="major", color=grid_color, linewidth=0.85, alpha=0.72)
        axis.set_axisbelow(True)
        axis.margins(x=0.012)
        axis.tick_params(colors="#596579", labelsize=10.5, length=0, pad=7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#CBD2DC")
            axis.spines[side].set_linewidth(1.0)

        handles, legend_labels = axis.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.975, 0.955),
            frameon=True,
            facecolor="white",
            edgecolor="#D7DCE4",
            framealpha=0.96,
            fontsize=10.5,
            ncol=2,
            borderpad=0.75,
            handlelength=2.6,
        )
        legend.get_frame().set_linewidth(0.8)
        fig.suptitle(
            title,
            x=0.095,
            y=0.94,
            ha="left",
            fontsize=19,
            fontweight="semibold",
            color="#172033",
        )
        axis.text(
            0.018,
            0.965,
            f"RMSE  {rmse:.4f}    MAE  {mae:.4f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#344054",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#F8FAFD",
                "edgecolor": "#D7DCE4",
                "linewidth": 0.8,
                "alpha": 0.97,
            },
        )
        fig.text(
            0.985,
            0.018,
            "Simple test trajectory 47 · 224-step rollout",
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="#7A8494",
        )
        fig.subplots_adjust(left=0.095, right=0.975, bottom=0.14, top=0.80)

        output_path = output_dir / f"{tag}_{slug}_polished.png"
        fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_joint_trajectories(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    output_dir: Path,
    tag: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), facecolor="white")
    time = np.arange(len(true_traj))
    states = [
        ("theta1", "theta1 (rad)"),
        ("theta2", "theta2 (rad)"),
        ("theta1_dot", "theta1_dot (rad/s)"),
        ("theta2_dot", "theta2_dot (rad/s)"),
    ]
    truth_color = "#2563EB"
    prediction_color = "#F04452"

    for index, axis in enumerate(axes.flat):
        axis.set_facecolor("#FFFFFF")
        axis.plot(
            time,
            true_traj[:, index],
            label=f"true_{states[index][0]}",
            color=truth_color,
            linewidth=2.15,
            zorder=3,
        )
        axis.plot(
            time,
            pred_traj[:, index],
            label=f"pred_{states[index][0]}",
            color=prediction_color,
            linewidth=1.95,
            linestyle=(0, (5, 3)),
            zorder=4,
        )
        values = np.concatenate((true_traj[:, index], pred_traj[:, index]))
        value_min = float(values.min())
        value_max = float(values.max())
        value_span = max(value_max - value_min, 1.0e-6)
        axis.set_ylim(value_min - 0.06 * value_span, value_max + 0.26 * value_span)

        axis.set_title(states[index][0], fontsize=14, color="#172033", pad=10)
        axis.set_xlabel("time_step", fontsize=10.5, color="#344054", labelpad=6)
        axis.set_ylabel(states[index][1], fontsize=10.5, color="#344054", labelpad=6)
        axis.grid(True, color="#DDE2EA", linewidth=0.7, alpha=0.48)
        axis.set_axisbelow(True)
        axis.margins(x=0.012)
        axis.tick_params(colors="#596579", labelsize=9.5, length=3, pad=5)
        for spine in axis.spines.values():
            spine.set_color("#AEB7C4")
            spine.set_linewidth(0.85)
        legend = axis.legend(
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=True,
            facecolor="white",
            edgecolor="#C8CFD9",
            framealpha=0.94,
            fontsize=8.2,
            borderpad=0.38,
            labelspacing=0.28,
            handlelength=2.15,
            handletextpad=0.55,
        )
        legend.get_frame().set_linewidth(0.7)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.965, wspace=0.16, hspace=0.22)
    output_path = output_dir / f"{tag}_trajectory.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_phase_space(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    output_dir: Path,
    tag: str,
) -> Path:
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(true_traj[:, 0], true_traj[:, 2], label="true", color="tab:blue")
    axis.plot(pred_traj[:, 0], pred_traj[:, 2], label="pred", color="tab:red", linestyle="--")
    axis.scatter(true_traj[0, 0], true_traj[0, 2], label="start", color="tab:green")
    axis.set_xlabel("theta1")
    axis.set_ylabel("theta1_dot")
    axis.set_title("Acrobot phase space")
    axis.grid(True)
    axis.legend()

    output_path = output_dir / f"{tag}_phase.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_acrobot_animation(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    output_dir: Path,
    tag: str,
    fps: int = 15,
) -> Path:
    frames = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for index in range(0, len(true_traj), 2):
        for axis in axes:
            axis.clear()

        draw_acrobot_state(
            axes[0],
            true_traj[index],
            color="tab:blue",
            line_style="-",
            legend_label="Original",
            panel_title="Original",
        )
        draw_acrobot_state(
            axes[1],
            pred_traj[index],
            color="tab:red",
            line_style="--",
            legend_label="Generated",
            panel_title="Generated",
        )
        fig.suptitle(f"Acrobot Comparison | Frame {index}", fontsize=14)

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba(), copy=True)
        frames.append(frame)

    output_path = output_dir / f"{tag}.gif"
    imageio.mimsave(output_path, frames, duration=1000/fps, loop=0)
    plt.close(fig)
    return output_path


def save_rollout_comparison(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    output_dir: Path,
    tag: str,
    num_snapshots: int = 6,
) -> Path:
    snapshot_indices = np.linspace(
        0,
        len(true_traj) - 1,
        num=min(num_snapshots, len(true_traj)),
        dtype=int,
    )
    fig, axes = plt.subplots(2, len(snapshot_indices), figsize=(3 * len(snapshot_indices), 6))

    if len(snapshot_indices) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, frame_index in enumerate(snapshot_indices):
        draw_acrobot_state(
            axes[0, column],
            true_traj[frame_index],
            color="tab:blue",
            line_style="-",
            legend_label="Original",
            panel_title=f"t = {frame_index}",
            show_legend=False,
        )
        draw_acrobot_state(
            axes[1, column],
            pred_traj[frame_index],
            color="tab:red",
            line_style="--",
            legend_label="Generated",
            panel_title="",
            show_legend=False,
        )
        if column == 0:
            axes[0, column].set_ylabel("Original", fontsize=12)
            axes[1, column].set_ylabel("Generated", fontsize=12)

    legend_handles = [
        Line2D([0], [0], color="tab:blue", linewidth=2, linestyle="-", marker="o", label="Original"),
        Line2D([0], [0], color="tab:red", linewidth=2, linestyle="--", marker="o", label="Generated"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Rollout Comparison", fontsize=15, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = output_dir / f"{tag}_rollout.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def draw_acrobot_state(
    axis,
    state: np.ndarray,
    *,
    color: str,
    line_style: str,
    legend_label: str,
    panel_title: str,
    show_legend: bool = True,
) -> None:
    x1, y1, x2, y2 = compute_joint_positions(state)
    axis.plot([0, x1, x2], [0, y1, y2], marker="o", linestyle=line_style, color=color, linewidth=2, label=legend_label)
    axis.set_xlim(-2.5, 2.5)
    axis.set_ylim(-2.5, 2.5)
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.3)
    axis.set_xticks([])
    axis.set_yticks([])
    if panel_title:
        axis.set_title(panel_title)
    if show_legend:
        axis.legend(loc="upper right", frameon=False)


def compute_joint_positions(state: np.ndarray) -> tuple[float, float, float, float]:
    theta1 = float(state[0])
    theta2 = float(state[1])
    x1 = np.sin(theta1)
    y1 = -np.cos(theta1)
    x2 = x1 + np.sin(theta1 + theta2)
    y2 = y1 - np.cos(theta1 + theta2)
    return x1, y1, x2, y2


def save_video_rollout_comparison(ground_truth, generated, times, output_stem,
                                  condition_frames=10, max_frames=10, fps=10.0,
                                  title="KTH Rollout Comparison"):
    """Save an animated side-by-side GIF and a two-row rollout PNG."""
    from PIL import Image, ImageDraw

    if len(ground_truth) != len(generated) or len(times) != len(ground_truth):
        raise ValueError("Video comparison needs aligned truth, predictions and times.")
    if fps <= 0 or max_frames < 1:
        raise ValueError("Visualization fps and max_frames must be positive.")
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if ground_truth.shape[1] == 1:
        ground_truth = ground_truth.repeat(1, 3, 1, 1)
        generated = generated.repeat(1, 3, 1, 1)
    truth = ((ground_truth.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    pred = ((generated.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    panels = []
    width, height = 256, 256
    for index, (real, fake) in enumerate(zip(truth, pred)):
        panel = Image.new("RGB", (2*width, height+48), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((8, 5), "Ground Truth", fill="black")
        draw.text((width+8, 5), "Generated", fill="black")
        phase = "Conditioning" if index < condition_frames else "Prediction"
        draw.text((8, 23), f"Frame {index} | t={float(times[index]):.4g} | {phase}", fill="black")
        panel.paste(Image.fromarray(real).resize((width, height), Image.Resampling.NEAREST), (0, 48))
        panel.paste(Image.fromarray(fake).resize((width, height), Image.Resampling.NEAREST), (width, 48))
        panels.append(panel)
    panels[0].save(output_stem.with_suffix(".gif"), save_all=True, append_images=panels[1:],
                   duration=max(10, round(1000/fps)), loop=0, disposal=2, optimize=False)
    # Include both the final conditioning frame and the first predicted frame.
    mandatory = {0, condition_frames-1, condition_frames, len(truth)-1}
    indices = sorted(set(np.linspace(0, len(truth)-1, min(max_frames, len(truth)), dtype=int)) | mandatory)
    indices = [i for i in indices if 0 <= i < len(truth)]
    fig, axes = plt.subplots(2, len(indices), figsize=(2*len(indices), 4.7), squeeze=False)
    for column, index in enumerate(indices):
        axes[0, column].imshow(truth[index])
        axes[1, column].imshow(pred[index])
        phase = "Context" if index < condition_frames else f"Pred +{index-condition_frames+1}"
        axes[0, column].set_title(f"{phase}\nt={float(times[index]):.4g}", fontsize=9)
        for axis in axes[:, column]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("Ground Truth")
    axes[1, 0].set_ylabel("Generated")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=150)
    plt.close(fig)


def save_nse_rollout_comparison(
    ground_truth: np.ndarray,
    generated: np.ndarray,
    output_path: str | Path,
    max_frames: int = 8,
) -> Path:
    """Save ground truth, generated NSE fields, and absolute error in one panel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = min(len(ground_truth), len(generated), max_frames)
    indices = np.linspace(0, min(len(ground_truth), len(generated)) - 1, frame_count, dtype=int)
    truth = np.asarray(ground_truth)[indices]
    prediction = np.asarray(generated)[indices]
    error = np.abs(truth - prediction)
    value_limit = max(float(np.abs(truth).max()), float(np.abs(prediction).max()), 1e-8)

    fig, axes = plt.subplots(3, frame_count, figsize=(2.2 * frame_count, 6.6), squeeze=False)
    field_image = None
    error_image = None
    for column, frame_index in enumerate(indices):
        field_image = axes[0, column].imshow(
            truth[column], cmap="coolwarm", vmin=-value_limit, vmax=value_limit
        )
        axes[0, column].set_title(f"t = {frame_index}")
        axes[1, column].imshow(
            prediction[column], cmap="coolwarm", vmin=-value_limit, vmax=value_limit
        )
        error_image = axes[2, column].imshow(error[column], cmap="inferno", vmin=0)
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    axes[0, 0].set_ylabel("Ground Truth", fontsize=11)
    axes[1, 0].set_ylabel("Generated", fontsize=11)
    axes[2, 0].set_ylabel("Absolute Error", fontsize=11)
    fig.suptitle("NSE Rollout Comparison", fontsize=15)
    if field_image is not None:
        fig.colorbar(field_image, ax=axes[:2, :], fraction=0.012, pad=0.015, label="Field value")
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[2, :], fraction=0.012, pad=0.015, label="Absolute error")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.04, wspace=0.08, hspace=0.12)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
