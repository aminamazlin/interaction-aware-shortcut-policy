from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


def _repeat_first_obs(batch: Dict, num_samples: int, device) -> Dict[str, torch.Tensor]:
    obs = batch["obs"]
    if isinstance(obs, dict):
        return {
            key: value[:1].repeat(num_samples, *([1] * (value.ndim - 1))).to(device)
            for key, value in obs.items()
        }
    return {
        "obs": obs[:1].repeat(num_samples, *([1] * (obs.ndim - 1))).to(device)
    }


@torch.no_grad()
def _sample_actions(policy, batch: Dict, num_samples: int, sample_batch_size: int) -> np.ndarray:
    was_training = policy.training
    policy.eval()

    actions = []
    remaining = int(num_samples)
    while remaining > 0:
        this_batch = min(int(sample_batch_size), remaining)
        obs_dict = _repeat_first_obs(batch, this_batch, policy.device)
        policy.reset()
        action = policy.predict_action(obs_dict)["action"]
        actions.append(action.detach().cpu().numpy())
        remaining -= this_batch

    if was_training:
        policy.train()

    return np.concatenate(actions, axis=0)


def _pca_2d(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[: min(2, vh.shape[0])].T
    coords = centered @ components
    if coords.shape[1] == 1:
        coords = np.pad(coords, ((0, 0), (0, 1)))
    return coords


def _set_shared_limits(axes, points):
    xy = np.concatenate(points, axis=0)
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    pad = 0.08 * np.maximum(maxs - mins, 1e-6)
    for ax in axes:
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])


def plot_action_distribution(
    teacher_policy,
    student_policy,
    batch: Dict,
    num_samples: int = 256,
    sample_batch_size: int = 64,
):
    teacher_actions = _sample_actions(teacher_policy, batch, num_samples, sample_batch_size)
    student_actions = _sample_actions(student_policy, batch, num_samples, sample_batch_size)

    teacher_flat = teacher_actions.reshape(teacher_actions.shape[0], -1)
    student_flat = student_actions.reshape(student_actions.shape[0], -1)
    all_flat = np.concatenate([teacher_flat, student_flat], axis=0)
    all_2d = _pca_2d(all_flat)

    teacher_2d = all_2d[: teacher_flat.shape[0]]
    student_2d = all_2d[teacher_flat.shape[0] :]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    _set_shared_limits(axes, [teacher_2d, student_2d])

    axes[0].hexbin(teacher_2d[:, 0], teacher_2d[:, 1], gridsize=35, cmap="Blues", mincnt=1)
    axes[0].scatter(teacher_2d[:, 0], teacher_2d[:, 1], s=8, alpha=0.25, color="#1f77b4")
    axes[0].set_title("Teacher")

    axes[1].hexbin(student_2d[:, 0], student_2d[:, 1], gridsize=35, cmap="Oranges", mincnt=1)
    axes[1].scatter(student_2d[:, 0], student_2d[:, 1], s=8, alpha=0.25, color="#ff7f0e")
    axes[1].set_title("Student")

    axes[2].scatter(teacher_2d[:, 0], teacher_2d[:, 1], s=10, alpha=0.35, label="Teacher")
    axes[2].scatter(student_2d[:, 0], student_2d[:, 1], s=10, alpha=0.35, label="Student")
    axes[2].set_title("Overlay")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle("Action Chunk Distribution for One Fixed Observation")
    fig.tight_layout()
    return fig
