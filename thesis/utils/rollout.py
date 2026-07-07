"""
Generic rollout utility for evaluating any model wrapper on any MimicGen env.

This is a standalone helper (not tied to any baseline) that the trainers
and the eval.py script can call directly.

Example
-------
    from thesis.utils.rollout import run_rollouts
    from thesis.models import DiffusionPolicyWrapper
    from thesis.env.mimicgen_env import MimicGenEnvWrapper

    wrapper = DiffusionPolicyWrapper()
    wrapper.load_checkpoint("outputs/dp_stack/checkpoints/latest.ckpt")

    env = MimicGenEnvWrapper.from_dataset("data/mimicgen/stack_d0.hdf5")

    metrics = run_rollouts(wrapper, env, n_episodes=50, max_steps=400)
    print(metrics)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from env.base_env import BaseEnvWrapper
from models.base_model import BaseModelWrapper


def run_rollouts(
    model: BaseModelWrapper,
    env: BaseEnvWrapper,
    n_episodes: int = 50,
    max_steps: int = 400,
    n_action_steps: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run *n_episodes* rollouts of *model* in *env* and return aggregated metrics.

    Args:
        model:          A loaded BaseModelWrapper (predict() must work).
        env:            A BaseEnvWrapper instance.
        n_episodes:     Number of evaluation episodes.
        max_steps:      Maximum environment steps per episode.
        n_action_steps: How many actions from each prediction to execute.
                        If None, executes the full predicted action chunk.
        seed:           Optional RNG seed for reproducibility.
        verbose:        Print per-episode results.

    Returns:
        {
            "success_rate":  float  [0, 1]
            "mean_reward":   float
            "std_reward":    float
            "num_episodes":  int
        }
    """
    if seed is not None:
        np.random.seed(seed)

    successes: list[float] = []
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        model.reset()
        obs = env.reset()
        episode_reward = 0.0
        done = False

        step = 0
        while not done and step < max_steps:
            # Run policy inference.
            actions = model.predict(obs)  # (horizon, action_dim) or (n_action_steps, action_dim)

            # Clip to n_action_steps if specified.
            if n_action_steps is not None:
                actions = actions[:n_action_steps]

            # Execute each action in the chunk.
            for action in actions:
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
                if done or step >= max_steps:
                    break

        success = float(env.is_success())
        successes.append(success)
        episode_rewards.append(episode_reward)

        if verbose:
            print(f"  Episode {ep+1:3d}/{n_episodes} | "
                  f"steps={step:3d} | reward={episode_reward:.2f} | "
                  f"success={bool(success)}")

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "num_episodes": n_episodes,
    }
