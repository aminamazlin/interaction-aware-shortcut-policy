"""
Thesis-facing evaluation runner for the Franka Kitchen benchmark.

Franka Kitchen support is provided by the local thesis kitchen package.
This subclass keeps the thesis task YAML independent from upstream module paths
while reusing the battle-tested vectorized rollout and logging implementation.
"""
from __future__ import annotations

import collections
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb

from thesis.utils.pytorch_util import dict_apply
from thesis.env.kitchen.base import KitchenBase
from thesis.env.runners.kitchen_lowdim_runner import (
    KitchenLowdimRunner as _KitchenLowdimRunner,
)
from thesis.policy.base_lowdim_policy import BaseLowdimPolicy


class FrankaKitchenLowdimRunner(_KitchenLowdimRunner):
    """Adapter around diffusion-policy's Kitchen low-dimensional runner."""

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        
        # init kitchen environment
        env = self.env

        # number of parallel envs
        n_envs = len(self.env_fns)
        
        # number of total initializations to run (some may be repeated if n_envs > n_inits)
        n_inits = len(self.env_init_fn_dills)
        
        # if runner is configured to run more initializations than parallel envs,  it splits evaluation into chunks
        n_chunks = math.ceil(n_inits / n_envs)

        # one video path per episode initialization
        all_video_paths = [None] * n_inits
        
        # one reward list per episode initialization, each containing the rewards at each step of the episode
        all_rewards = [None] * n_inits
        
        # one info dict per episode init, which is later used to extract things like completed tasks.
        last_info = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # reset policy and env state for new trajectories
            obs = env.reset()
            past_action = None
            policy.reset()
            
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval KitcenRunner {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            while not done:
                # obs input for policy
                np_obs_dict = {
                    "obs": obs.astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1):
                    ].astype(np.float32)

                # convert every np array in the obs dict to a torch tensor
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    # run policy
                    action_dict = policy.predict_action(obs_dict)

                # convert every tensor in the action dict to a numpy array on CPU
                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                # extract action sequence for env step
                action = np_action_dict["action"]

                obs, reward, done_array, info = env.step(action)
                done_array = np.asarray(done_array, dtype=bool)
                
                # check if all envs are done to know when to stop the loop
                done = np.all(done_array)
                past_action = action

                pbar.update(action.shape[1])
            pbar.close()

            # record video, rewards, and last info dict for this chunk of trajectories
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[this_local_slice]
            last_info[this_global_slice] = [
                dict((k, v[-1]) for k, v in x.items()) for x in info
            ][this_local_slice]

        # holds final metrics and vids returned by the runner
        log_data = dict()
        # stores rewards per split (train, val)
        prefix_total_reward_map = collections.defaultdict(list)
        # stores how many sub-tasks were completed per split 
        prefix_n_completed_map = collections.defaultdict(list)
        # stores per-subtask success indicators per split
        prefix_task_success_map = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

        # loops through each rollout
        for i in range(n_inits):
            # get seed used for that rollout
            seed = self.env_seeds[i]
            # get split label for that rollout
            prefix = self.env_prefixs[i]
            # get rewards
            this_rewards = all_rewards[i]
            # normalize total reward
            total_reward = np.sum(this_rewards) / 7
            # store total reward
            prefix_total_reward_map[prefix].append(total_reward)

            # get number of completed tasks 
            completed_tasks = set(last_info[i]["completed_tasks"])
            n_completed_tasks = len(completed_tasks)
            # store number of completed tasks per split
            prefix_n_completed_map[prefix].append(n_completed_tasks)
            for task_name in KitchenBase.ALL_TASKS:
                prefix_task_success_map[prefix][task_name].append(
                    float(task_name in completed_tasks)
                )

            # get saved video
            video_path = all_video_paths[i]
            # add video to final log dict
            if video_path is not None:
                log_data[prefix + f"sim_video_{seed}"] = wandb.Video(video_path)

        for prefix, value in prefix_total_reward_map.items():
            # store total normalized reward per split in final log dict
            log_data[prefix + "mean_score"] = np.mean(value)
            
        n_total_tasks = len(KitchenBase.ALL_TASKS)
        n_success_tasks = 4
        for prefix, value in prefix_n_completed_map.items():
            # get num of completed tasks per split
            n_completed = np.array(value)
            log_data[prefix + "success_rate"] = float(
                np.mean(n_completed >= n_success_tasks)
            )
            log_data[prefix + "completion_fraction"] = float(
                np.mean(n_completed / max(n_total_tasks, 1))
            )
            # loop through each possible number of completed tasks 
            for i in range(n_total_tasks):
                n = i + 1
                # store percentage of episodes that completed at least n tasks in final log dict
                log_data[prefix + f"p_{n}"] = np.mean(n_completed >= n)
        
        if prefix_task_success_map:
            task_names = list(KitchenBase.ALL_TASKS)
            print(f"prefix_task_success_map: {prefix_task_success_map}")
            print(f"task_names: {task_names}")
            
            split_prefixes = [
                prefix for prefix in ("train/", "test/") if prefix in prefix_task_success_map
            ]
            if not split_prefixes:
                split_prefixes = sorted(prefix_task_success_map.keys())

            x = np.arange(len(task_names))
            for prefix in split_prefixes:
                success_rates = [
                    float(np.mean(prefix_task_success_map[prefix][task_name]))
                    if prefix_task_success_map[prefix][task_name] else 0.0
                    for task_name in task_names
                ]
                for task_name, success_rate in zip(task_names, success_rates):
                    log_data[prefix + f"subgoal/{task_name}"] = success_rate
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.bar(x, success_rates, width=0.6)
                ax.set_xticks(x)
                ax.set_xticklabels(task_names, rotation=25, ha="right")
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("Success Rate")
                ax.set_xlabel("Kitchen Subtask")
                ax.set_title(f"{prefix.rstrip('/').title()} Kitchen Subtask Success Rate")
                fig.tight_layout()
                log_data[prefix + "subtask_success_rate"] = wandb.Image(fig)
                plt.close(fig)

        return log_data
