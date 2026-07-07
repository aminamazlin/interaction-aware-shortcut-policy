"""
Custom evaluation runner for MimicGen environments using MimicGenEnvWrapper.
Compatible with diffusion policy training workflows.
"""
from __future__ import annotations

import os
import pathlib
import inspect
import json
import time
import numpy as np
import torch
import dill
import math
import tqdm
import wandb
import wandb.sdk.data_types.video as wv
from collections import defaultdict

from utils.video_recoder import VideoRecorder
from utils.video_recording_wrapper import VideoRecordingWrapper
from utils.multistep_wrapper import MultiStepWrapper
from utils.async_vector_env import AsyncVectorEnv


from thesis.policy.base_lowdim_policy import BaseLowdimPolicy
from thesis.policy.base_image_policy import BaseImagePolicy
from thesis.utils.pytorch_util import dict_apply
from thesis.env.runners.base_image_runner import BaseImageRunner
from thesis.env.runners.base_lowdim_runner import BaseLowdimRunner

from env.wrappers.mimicgen_wrapper import MimicGenEnvWrapper


class MimicGenLowdimRunner(BaseLowdimRunner):
    """
    Evaluation runner for MimicGen environments using MimicGenEnvWrapper.

    Creates MimicGen simulation environments and runs policy rollouts for evaluation
    during training.

    Args:
        output_dir: Where to save results
        dataset_path: Path to robomimic HDF5 dataset (for metadata only)
        task_name: MimicGen task name (e.g., "Coffee_D0", "Stack_D0")
        obs_keys: List of observation keys to extract from env
        n_train: Number of training rollouts
        n_train_vis: Number of training rollouts to visualize (with video)
        train_start_seed: Random seed for training init
        n_test: Number of test rollouts
        n_test_vis: Number of test rollouts to visualize
        test_start_seed: Random seed for test init
        max_steps: Maximum steps per episode
        n_obs_steps: Number of observation steps for history
        n_action_steps: Number of action steps per policy step
        n_latency_steps: Simulated latency in observation steps
        render_hw: (H, W) for rendering
        fps: Video frames per second
        crf: CRF quality for video encoding
        past_action: Whether to include past actions in obs
        abs_action: Whether actions are absolute
        n_envs: Number of parallel environments (None = n_train + n_test)
        robosuite_kwargs: Task-specific arguments forwarded to robosuite.make().
            Rendering and camera arguments are controlled by this runner.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_path: str,
        task_name: str,
        obs_keys: list[str],
        n_train: int = 10,
        n_train_vis: int = 2,
        train_start_seed: int = 0,
        n_test: int = 22,
        n_test_vis: int = 4,
        test_start_seed: int = 10000,
        max_steps: int = 400,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        n_latency_steps: int = 0,
        render_hw: tuple = (256, 256),
        fps: int = 10,
        crf: int = 22,
        past_action: bool = False,
        abs_action: bool = False,
        n_envs: int = None,
        render_gpu_device_id: int | None = None,
        image_size: tuple = (84, 84),
        robosuite_kwargs: dict | None = None,
    ):
        super().__init__(output_dir)

        # number of envs
        n_inits = n_train + n_test
        if n_envs is None:
            n_envs = n_inits
        else:
            n_envs = min(int(n_envs), n_inits)

        # Handle latency
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        self.task_name = task_name
        self.obs_keys = obs_keys
        self.dataset_path = dataset_path
        self.robosuite_kwargs = dict(robosuite_kwargs or {})
        self.render_gpu_device_id = self._resolve_render_gpu_device_id(
            render_gpu_device_id
        )
        self.image_size = tuple(image_size)
        self.image_keys = [key for key in obs_keys if "image" in key]
        self.lowdim_keys = [key for key in obs_keys if "image" not in key]

        env_seeds = []
        env_prefixs = []
        rollout_specs = []
        rollout_render_flags = []

        # Training rollouts
        for i in range(n_train):
            enable_render = i < n_train_vis
            seed = train_start_seed + i

            env_seeds.append(seed)
            env_prefixs.append("train/")
            rollout_specs.append({
                "seed": seed,
                "enable_render": enable_render,
            })
            rollout_render_flags.append(enable_render)

        # Test rollouts
        for i in range(n_test):
            enable_render = i < n_test_vis
            seed = test_start_seed + i

            env_seeds.append(seed)
            env_prefixs.append("test/")
            rollout_specs.append({
                "seed": seed,
                "enable_render": enable_render,
            })
            rollout_render_flags.append(enable_render)

        def make_init_fn_dill(seed: int, enable_render: bool):
            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.seed(seed)

            return dill.dumps(init_fn)

        slot_requires_render = [False] * n_envs
        for rollout_idx, enable_render in enumerate(rollout_render_flags):
            if enable_render:
                slot_requires_render[rollout_idx % n_envs] = True

        def make_env_fn(enable_render: bool):
            def env_fn():
                use_camera_obs = len(self.image_keys) > 0
                camera_names = [
                    key[:-6] if key.endswith("_image") else key
                    for key in self.image_keys
                ] or ["agentview"]
                camera_h, camera_w = self.image_size
                env_robosuite_kwargs = dict(self.robosuite_kwargs)
                env_robosuite_kwargs.update({
                    "has_renderer": False,
                    # Only allocate robosuite's offscreen context for env slots
                    # that are ever asked to record video across the chunk schedule.
                    "has_offscreen_renderer": enable_render or use_camera_obs,
                    "use_camera_obs": use_camera_obs,
                    "use_object_obs": True,
                    "control_freq": 20,
                })
                if use_camera_obs:
                    env_robosuite_kwargs.update(
                        {
                            "camera_names": camera_names,
                            "camera_heights": camera_h,
                            "camera_widths": camera_w,
                        }
                    )
                if enable_render or use_camera_obs:
                    # Slurm / EGL jobs are much more reliable when robosuite
                    # receives an explicit render GPU id instead of the default
                    # -1 auto-selection.
                    env_robosuite_kwargs["render_gpu_device_id"] = self.render_gpu_device_id

                env = MimicGenEnvWrapper(
                    task_name=task_name,
                    obs_keys=obs_keys,
                    robosuite_kwargs=env_robosuite_kwargs,
                )
                return MultiStepWrapper(
                    VideoRecordingWrapper(
                        env,
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec="h264",
                            input_pix_fmt="rgb24",
                            crf=crf,
                            thread_type="FRAME",
                            thread_count=1,
                        ),
                        file_path=None,
                        steps_per_render=max(20 // fps, 1),  # robosuite runs at 20 Hz
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps,
                )

            return env_fn

        # number of envs to evaluate on in parallel
        env_fns = [make_env_fn(enable_render) for enable_render in slot_requires_render]

        # spin up multiple workers to evaluate policy against various envs in parallel
        # Use shared_memory=False for Dict observation spaces
        env_init_fn_dills = [
            make_init_fn_dill(spec["seed"], spec["enable_render"])
            for spec in rollout_specs
        ]
        padding_init_fn_dill = make_init_fn_dill(env_seeds[0], False)
        videos_disabled = False
        try:
            # Important: the dummy env is constructed in the parent process
            # before AsyncVectorEnv forks/spawns workers. If it creates an
            # offscreen MuJoCo context, child workers inherit broken GL/EGL
            # state and can fail with "Offscreen framebuffer is not complete".
            # Keep the dummy env non-rendering; real worker slots still render.
            env = AsyncVectorEnv(
                env_fns,
                dummy_env_fn=make_env_fn(False),
                context="spawn",
                shared_memory=False,
            )
        except Exception as exc:
            if not any(slot_requires_render):
                raise
            print(
                "[mimicgen_runner] Offscreen rendering failed during env startup; "
                "retrying evaluation without video capture."
            )
            print(f"[mimicgen_runner] Original error: {type(exc).__name__}: {exc}")
            env_fns = [make_env_fn(False) for _ in range(n_envs)]
            env_init_fn_dills = [
                make_init_fn_dill(spec["seed"], False)
                for spec in rollout_specs
            ]
            padding_init_fn_dill = make_init_fn_dill(env_seeds[0], False)
            env = AsyncVectorEnv(
                env_fns,
                dummy_env_fn=make_env_fn(False),
                context="spawn",
                shared_memory=False,
            )
            videos_disabled = True

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.padding_init_fn_dill = padding_init_fn_dill
        self.videos_disabled = videos_disabled
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.abs_action = abs_action

    def _build_policy_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        policy_obs = {}
        if self.lowdim_keys:
            policy_obs["obs"] = obs["obs"][:, : self.n_obs_steps].astype(np.float32)
        for key in self.image_keys:
            image = obs[key][:, : self.n_obs_steps]
            policy_obs[key] = np.moveaxis(image, -1, -3).astype(np.float32) / 255.0
        return policy_obs

    def _resolve_render_gpu_device_id(self, render_gpu_device_id: int | None) -> int:
        if render_gpu_device_id is not None:
            return int(render_gpu_device_id)

        for env_name in ("ROBOSUITE_RENDER_GPU_DEVICE_ID", "MUJOCO_EGL_DEVICE_ID"):
            value = os.environ.get(env_name)
            if value is not None and value.strip().isdigit():
                return int(value)

        return 0

    def run(self, policy: BaseLowdimPolicy):
        """
        Run policy evaluation rollouts.

        Args:
            policy: Trained policy to evaluate

        Returns:
            Dictionary with evaluation metrics and video paths
        """
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # Plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # Allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_episode_metrics = [None] * n_inits
        inference_times = []
        inference_action_counts = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.padding_init_fn_dill] * n_diff)
            assert len(this_init_fns) == n_envs

            # Init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # Start rollout
            obs = env.reset()
            past_action = None
            latent_state = None
            policy.reset()
            predict_accepts_state = "state" in inspect.signature(
                policy.predict_action
            ).parameters

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {self.task_name} chunk {chunk_idx+1}/{n_chunks}",
                leave=False,
            )

            done = False
            while not done:
                # Create obs dict
                np_obs_dict = self._build_policy_obs(obs)
                if self.past_action and (past_action is not None):
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)

                # Device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # Run policy
                self._sync_policy_device(device)
                inference_start = time.perf_counter()
                with torch.no_grad():
                    if predict_accepts_state:
                        action_dict = policy.predict_action(
                            obs_dict,
                            state=latent_state,
                        )
                    else:
                        action_dict = policy.predict_action(obs_dict)
                self._sync_policy_device(device)
                inference_elapsed = time.perf_counter() - inference_start

                latent_state = action_dict.get("state")
                if latent_state is not None:
                    latent_state = latent_state.detach()

                # Device transfer
                action_only_dict = {
                    key: value
                    for key, value in action_dict.items()
                    if key != "state"
                }
                np_action_dict = dict_apply(
                    action_only_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )

                # Handle latency_steps
                action = np_action_dict["action"][:, self.n_latency_steps :]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                inference_times.append(inference_elapsed)
                inference_action_counts.append(int(action.shape[0] * action.shape[1]))

                # Step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # Update pbar
                pbar.update(action.shape[1])

            pbar.close()

            # Collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
            all_episode_metrics[this_global_slice] = env.call("get_episode_metrics")[
                this_local_slice
            ]

        # Log
        completion_scores = defaultdict(list)
        total_rewards = defaultdict(list)
        max_rewards = defaultdict(list)
        episode_records = defaultdict(list)
        debug_records = []
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            episode_record = self._build_episode_record(
                episode_metrics=all_episode_metrics[i],
                max_reward=max_reward,
                seed=seed,
            )
            n_subgoals = max(len(episode_record["stage_names"]), 1)
            rewards = np.asarray(all_rewards[i], dtype=np.float32)
            reward_sum = float(np.sum(rewards))
            reward_max = float(np.max(rewards)) if rewards.size > 0 else 0.0
            nonzero_reward_steps = np.flatnonzero(rewards > 0.0).astype(int).tolist()
            env_reward_score = float(reward_sum / n_subgoals)
            completion_score = float(episode_record["n_completed_tasks"] / n_subgoals)
            total_rewards[prefix].append(env_reward_score)
            completion_scores[prefix].append(completion_score)
            episode_records[prefix].append(episode_record)

            # Visualize sim
            video_path = all_video_paths[i]
            video_key = None
            if video_path is not None:
                video_key = prefix + f"sim_video_{seed}"
                sim_video = wandb.Video(video_path)
                log_data[video_key] = sim_video

            stage_completion_steps = dict(
                all_episode_metrics[i].get("stage_completion_steps", {})
            )
            current_stage_signals = dict(
                all_episode_metrics[i].get("current_stage_signals", {})
            )
            for stage_name in episode_record["stage_names"]:
                stage_completion_steps.setdefault(stage_name, None)
                current_stage_signals.setdefault(
                    stage_name,
                    bool(episode_record["completed_stages"].get(stage_name, False)),
                )
            episode_length = int(all_episode_metrics[i].get("episode_length", 0))
            success = bool(episode_record["success"])
            ended_at_horizon = bool(episode_length >= self.max_steps)
            if success:
                termination_reason = "success"
            elif ended_at_horizon:
                termination_reason = "horizon"
            else:
                termination_reason = "env_done_before_success"

            debug_record = {
                "prefix": prefix.rstrip("/"),
                "seed": int(seed),
                "video_key": video_key,
                "video_path": None if video_path is None else str(video_path),
                "reward_sum": reward_sum,
                "reward_max": reward_max,
                "mean_score_contribution": completion_score,
                "env_reward_score_contribution": env_reward_score,
                "nonzero_reward_steps": nonzero_reward_steps,
                "success": success,
                "n_completed_tasks": int(episode_record["n_completed_tasks"]),
                "episode_length": episode_length,
                "max_steps": int(self.max_steps),
                "ended_at_horizon": ended_at_horizon,
                "termination_reason": termination_reason,
                "stage_names": list(episode_record["stage_names"]),
                "completed_stages": dict(episode_record["completed_stages"]),
                "stage_completion_steps": stage_completion_steps,
                "current_stage_signals": current_stage_signals,
            }
            debug_records.append(debug_record)
            print(
                "[mimicgen_eval_debug] "
                f"{debug_record['prefix']} seed={seed} video_key={video_key} "
                f"reward_sum={reward_sum:.4f} reward_max={reward_max:.4f} "
                f"score_contrib={completion_score:.4f} "
                f"env_reward_score={env_reward_score:.4f} "
                f"success={debug_record['success']} "
                f"episode_length={debug_record['episode_length']} "
                f"ended_at_horizon={debug_record['ended_at_horizon']} "
                f"termination_reason={debug_record['termination_reason']} "
                f"completed={debug_record['completed_stages']} "
                f"signals={debug_record['current_stage_signals']} "
                f"video_path={debug_record['video_path']}"
            )

        debug_dir = pathlib.Path(self.output_dir).joinpath("eval_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir.joinpath("latest_rollouts.json")
        with debug_path.open("w") as f:
            json.dump(debug_records, f, indent=2)
        timestamped_debug_path = debug_dir.joinpath(
            f"rollouts_{int(time.time())}.json"
        )
        with timestamped_debug_path.open("w") as f:
            json.dump(debug_records, f, indent=2)
        print(f"[mimicgen_eval_debug] wrote {debug_path}")

        for table_prefix in sorted({record["prefix"] for record in debug_records}):
            table = wandb.Table(
                columns=[
                    "seed",
                    "video_key",
                    "video_path",
                    "reward_sum",
                    "reward_max",
                    "mean_score_contribution",
                    "success",
                    "n_completed_tasks",
                    "completed_stages",
                    "nonzero_reward_steps",
                ]
            )
            for record in debug_records:
                if record["prefix"] != table_prefix:
                    continue
                table.add_data(
                    record["seed"],
                    record["video_key"],
                    record["video_path"],
                    record["reward_sum"],
                    record["reward_max"],
                    record["mean_score_contribution"],
                    record["success"],
                    record["n_completed_tasks"],
                    json.dumps(record["completed_stages"], sort_keys=True),
                    json.dumps(record["nonzero_reward_steps"]),
                )
            log_data[f"{table_prefix}/rollout_debug"] = table

        # Kitchen reward encodes completed subtasks. CoffeePreparation reward is
        # sparse, so use detector completion for the Kitchen-equivalent score and
        # keep raw env reward as a separate diagnostic.
        for prefix, value in completion_scores.items():
            log_data[prefix + "mean_score"] = float(np.mean(value))
            log_data[prefix + "mean_env_reward_score"] = float(
                np.mean(total_rewards[prefix])
            )
            log_data[prefix + "mean_max_reward"] = float(np.mean(max_rewards[prefix]))
            prefix_records = episode_records[prefix]
            log_data.update(self._summarize_episode_records(prefix, prefix_records))
        log_data.update(
            self._summarize_inference_times(
                inference_times=inference_times,
                inference_action_counts=inference_action_counts,
            )
        )

        return log_data

    def _sync_policy_device(self, device) -> None:
        try:
            torch_device = torch.device(device)
        except (TypeError, RuntimeError):
            return
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(torch_device)

    def _summarize_inference_times(
        self,
        inference_times: list[float],
        inference_action_counts: list[int],
    ) -> dict:
        if not inference_times:
            return {}

        times = np.asarray(inference_times, dtype=np.float64)
        total_time = float(np.sum(times))
        total_actions = float(np.sum(inference_action_counts))
        return {
            "inference/mean_policy_call_ms": float(np.mean(times) * 1000.0),
            "inference/actions_per_sec": float(total_actions / total_time)
            if total_time > 0.0
            else 0.0,
        }

    def _build_episode_record(self, episode_metrics: dict, max_reward: float, seed: int):
        stage_names = list(episode_metrics.get("stage_names", []))
        completed_stages = dict(episode_metrics.get("completed_stages", {}))
        for stage_name in stage_names:
            completed_stages.setdefault(stage_name, False)
        success = bool(episode_metrics.get("success", False))
        if success and "task" in stage_names:
            completed_stages["task"] = True
        n_completed_tasks = sum(
            bool(completed_stages.get(stage_name, False))
            for stage_name in stage_names
        )
        if not stage_names:
            n_completed_tasks = int(success)
        return {
            "seed": seed,
            "success": success,
            "max_reward": float(max_reward),
            "stage_names": stage_names,
            "completed_stages": completed_stages,
            "n_completed_tasks": n_completed_tasks,
        }

    def _summarize_episode_records(self, prefix: str, records: list[dict]):
        log_data = dict()
        if not records:
            return log_data

        n_completed = np.array(
            [record["n_completed_tasks"] for record in records],
            dtype=np.float32,
        )
        successes = np.array(
            [record["success"] for record in records],
            dtype=np.float32,
        )
        n_subgoals = max(len(record["stage_names"]) for record in records)
        if n_subgoals == 0:
            n_subgoals = 1

        log_data[prefix + "success_rate"] = float(np.mean(successes))
        log_data[prefix + "completion_fraction"] = float(
            np.mean(n_completed / max(n_subgoals, 1))
        )

        for i in range(n_subgoals):
            n = i + 1
            log_data[prefix + f"p_{n}"] = float(np.mean(n_completed >= n))

        stage_names = []
        for record in records:
            for stage_name in record["stage_names"]:
                if stage_name not in stage_names:
                    stage_names.append(stage_name)

        if stage_names:
            table = wandb.Table(columns=["subgoal", "success_rate"])
            for stage_name in stage_names:
                success_rate = float(
                    np.mean(
                        [
                            bool(record["completed_stages"].get(stage_name, False))
                            for record in records
                        ]
                    )
                )
                log_data[prefix + f"subgoal/{stage_name}"] = success_rate
                table.add_data(stage_name, success_rate)

            prefix_name = prefix.rstrip("/") or "eval"
            log_data[prefix + "subgoal_success_rates"] = wandb.plot.bar(
                table,
                "subgoal",
                "success_rate",
                title=f"{prefix_name} subgoal success rates",
            )

        return log_data


class MimicGenImageRunner(MimicGenLowdimRunner, BaseImageRunner):
    """Image-capable MimicGen runner for Diffusion Policy hybrid workspaces."""

    def _build_policy_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        np_obs_dict: dict[str, np.ndarray] = {}
        for key in self.lowdim_keys:
            if key not in obs:
                raise RuntimeError(
                    f"Expected MimicGen low-dimensional observation key '{key}', "
                    f"got {sorted(obs.keys())}."
                )
            np_obs_dict[key] = obs[key][:, : self.n_obs_steps].astype(np.float32)

        for key in self.image_keys:
            if key not in obs:
                raise RuntimeError(
                    f"Expected MimicGen image observation key '{key}', got {sorted(obs.keys())}."
                )
            image = obs[key][:, : self.n_obs_steps]
            if image.ndim != 5:
                raise RuntimeError(
                    f"Expected image obs '{key}' as (B,T,H,W,C), got {image.shape}."
                )
            np_obs_dict[key] = np.moveaxis(image, -1, 2).astype(np.float32) / 255.0
        return np_obs_dict
