"""
Wrapper for MimicGen environments that exposes a Gym-compatible interface.

MimicGen tasks are Robosuite environments registered under names like
"Stack_D0", "Coffee_D0", etc.  We create them via robosuite.make() and
expose the unified BaseEnvWrapper interface.
"""
from __future__ import annotations
import os
import re
import sys
import pathlib
from typing import Any
import numpy as np


_ENV_DIR = pathlib.Path(__file__).resolve().parents[1]
_LOCAL_PACKAGE_ROOTS = {
    "robosuite": _ENV_DIR / "robosuite",
    "mimicgen": _ENV_DIR / "mimicgen",
    "robomimic": _ENV_DIR / "robomimic",

    # Kitchen_D0 before robosuite.make() is called.
    "robosuite_task_zoo": _ENV_DIR / "robosuite-task-zoo",
}


def _path_is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _purge_package_if_loaded_from_elsewhere(package: str, root: pathlib.Path) -> None:
    module = sys.modules.get(package)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if module_file is None:
        return
    if _path_is_relative_to(pathlib.Path(module_file), root):
        return

    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            del sys.modules[name]


for _pkg, _root in reversed(_LOCAL_PACKAGE_ROOTS.items()):
    _p = str(_root)
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

for _pkg, _root in _LOCAL_PACKAGE_ROOTS.items():
    _purge_package_if_loaded_from_elsewhere(_pkg, _root)


os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL", "").lower().strip() == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mimicgen  
import robosuite
from robosuite.environments.base import make  
from mimicgen.env_interfaces.base import make_interface  


PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robosuite.controllers import load_controller_config
import gym

from env.base_env import BaseEnvWrapper

_DEFAULT_ROBOSUITE_KWARGS: dict[str, Any] = {
    "robots": "Panda",
    "controller_configs": load_controller_config(default_controller="OSC_POSE"),
    "has_renderer": False,
    "has_offscreen_renderer": True,
    "use_camera_obs": True,
    "use_object_obs": True,
    "camera_names": ["agentview"],
    "camera_heights": 84,
    "camera_widths": 84,
    "reward_shaping": False,
    "control_freq": 20,
}


def _select_render_gpu_device_id(kwargs: dict[str, Any]) -> None:
    """Populate render_gpu_device_id for offscreen rendering when possible."""
    if not kwargs.get("has_offscreen_renderer", False):
        return
    if kwargs.get("render_gpu_device_id", None) not in (None, -1):
        return

    try:
        import egl_probe
    except ImportError:
        return

    valid_gpu_devices = egl_probe.get_available_devices()
    if valid_gpu_devices:
        kwargs["render_gpu_device_id"] = valid_gpu_devices[0]


class MimicGenEnvWrapper(BaseEnvWrapper):
    """
    Adapter for MimicGen simulation environments.

    Creates and owns a robosuite environment for the specified MimicGen task,
    then maps its output to the unified BaseEnvWrapper interface.

    Args:
        task_name:      Robosuite/MimicGen task name, e.g. "Stack_D0", "Coffee_D0".
        obs_keys:       Keys to extract from robosuite's obs dict and pack into
                        the unified obs dict.  Defaults to robot proprioception +
                        agentview image.
        robosuite_kwargs: Extra kwargs forwarded to robosuite.make().  These
                          override the defaults in _DEFAULT_ROBOSUITE_KWARGS.
    """

    def __init__(
        self,
        task_name: str,
        obs_keys: list[str] | None = None,
        robosuite_kwargs: dict[str, Any] | None = None,
    ):
        # Import mimicgen to trigger environment registration via mimicgen/__init__.py
        # noqa: F401  # type: ignore[import]

        self.task_name = task_name
        self._obs_keys = obs_keys or [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "agentview_image",
        ]

        kwargs = {**_DEFAULT_ROBOSUITE_KWARGS, **(robosuite_kwargs or {})}
        _select_render_gpu_device_id(kwargs)

        # initialize environment
        self.env = make(task_name, **kwargs)
        self.env_interface = self._create_env_interface()

        self._success = False
        self._episode_steps = 0
        self._subgoal_order = self._infer_subgoal_order()
        self._completed_stage_flags = {}
        self._stage_completion_steps = {}
        self._completion_fraction_history = []
        self._action_history = []
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}


    @property
    def action_space(self):
        """Create a gym-compatible action space from robosuite action_spec."""
        if not hasattr(self, '_action_space'):
            action_spec = self.env.action_spec
            if isinstance(action_spec, tuple) and len(action_spec) == 2:
                low, high = action_spec
                self._action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                raise ValueError(f"Unexpected action_spec format: {action_spec}")
        return self._action_space

    @property
    def observation_space(self):
        """Create a gym-compatible observation space based on actual observation structure."""
        if not hasattr(self, '_observation_space'):
            # Get a sample observation to determine shapes
            obs = self.reset()
            spaces = {}
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    if val.dtype in (np.uint8, np.int8):
                        low, high = 0, 255
                    else:
                        low, high = -np.inf, np.inf
                    spaces[key] = gym.spaces.Box(low=low, high=high, shape=val.shape, dtype=val.dtype)
            self._observation_space = gym.spaces.Dict(spaces)
        return self._observation_space

    def reset(self) -> dict[str, np.ndarray]:
        self._success = False
        self._episode_steps = 0
        self._completed_stage_flags = {name: False for name in self._subgoal_order}
        self._stage_completion_steps = {name: None for name in self._subgoal_order}
        self._completion_fraction_history = []
        self._action_history = []
        obs = self.env.reset()
        if self.task_name.startswith("PickPlace") and self.env.sim.model.stat.extent > 1e6:
            self.env.sim.model.stat.extent = 10.0
        self._record_stage_progress()
        return self.get_observation(obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        self._action_history.append(np.asarray(action, dtype=np.float32).copy())
        obs, reward, done, info = self.env.step(action)
        self._episode_steps += 1
        self._success = bool(self.env._check_success())
        stage_snapshot = self._record_stage_progress()
        done = done or self._success
        info = dict(info)
        info["success"] = self._success
        info["stage_completion_fraction"] = self._completion_fraction_history[-1]
        info["completed_stages"] = stage_snapshot
        return self.get_observation(obs), float(reward), done, info

    def render(
        self,
        mode: str = "rgb_array",
        height: int = 256,
        width: int = 256,
        camera_name: str = "agentview",
    ) -> np.ndarray | None:
        if mode == "rgb_array":
            # robosuite sim.render returns (H, W, 3) uint8 in BGR; flip to RGB
            frame = self.env.sim.render(
                height=height, width=width, camera_name=camera_name
            )[::-1]
            return frame
        elif mode == "human":
            if self.env.viewer is None:
                print("Warning: env created without renderer. Use has_renderer=True")
                return None
            return self.env.render()  # renders to window
        return None

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int | None = None) -> None:
        np.random.seed(seed)

  
    #  Metadata                                                         
    @property
    def action_dim(self) -> int:
        return int(np.prod(self.env.action_spec[0].shape))

    @property
    def obs_keys(self) -> list[str]:
        return self._obs_keys

    def is_success(self) -> bool:
        return self._success

    def get_episode_metrics(self) -> dict[str, Any]:
        if not self._subgoal_order:
            self._ensure_subgoal_tracking_initialized()
        return {
            "task_name": self.task_name,
            "success": bool(self._success),
            "episode_length": int(self._episode_steps),
            "stage_names": list(self._subgoal_order),
            "completed_stages": dict(self._completed_stage_flags),
            "stage_completion_steps": dict(self._stage_completion_steps),
            "current_stage_signals": self._get_stage_signals(),
            "completion_fraction_history": list(self._completion_fraction_history),
            "action_history": np.asarray(self._action_history, dtype=np.float32),
        }

    def reset_to(
        self,
        *,
        states: np.ndarray | None = None,
        model: str | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Restore a robosuite / MimicGen simulator state from a robomimic HDF5 demo.

        The generated coffee data stores MuJoCo XML per demo plus flattened
        simulator states per frame.  This mirrors robomimic's EnvRobosuite
        reset_to behavior so image datasets can render RGB observations from
        state-only demonstrations.
        """
        should_return_obs = False
        if model is not None:
            self.env.reset()
            xml = self.env.edit_model_xml(model)
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
        if states is not None:
            self.env.sim.set_state_from_flattened(states)
            self.env.sim.forward()
            should_return_obs = True
        if should_return_obs:
            raw_obs = self.env._get_observations(force_update=True)
            if all(key in raw_obs for key in self.obs_keys):
                return self.get_observation(raw_obs)
        return None

 
    #  Internal helpers                                                    
    def get_observation(self, raw_obs):
        lowdim_keys = [k for k in self.obs_keys if "image" not in k]
        image_keys  = [k for k in self.obs_keys if "image" in k]
        obs = {}
        if lowdim_keys:
            # Per-key tensors for image runners / hybrid policies that expect
            # each modality separately.
            for k in lowdim_keys:
                obs[k] = self._resolve_raw_obs_key(raw_obs, k)
            # Backward-compat concatenated view for the lowdim runner / policy.
            obs["obs"] = np.concatenate(
                [self._resolve_raw_obs_key(raw_obs, k).flatten() for k in lowdim_keys]
            )
        for k in image_keys:
            obs[k] = raw_obs[k]  # keep image as (H, W, C)
        return obs

    def _resolve_raw_obs_key(self, raw_obs: dict[str, np.ndarray], key: str) -> np.ndarray:
        if key in raw_obs:
            return raw_obs[key]
        if key == "object" and "object-state" in raw_obs:
            return raw_obs["object-state"]
        raise KeyError(key)

    def _create_env_interface(self):
        base_task_name = re.sub(r"_D\d+$", "", self.task_name)
        interface_name = f"MG_{base_task_name}"
        try:
            return make_interface(
                name=interface_name,
                interface_type="robosuite",
                env=self.env,
            )
        except Exception:
            return None

    def _get_stage_signals(self) -> dict[str, bool]:
        signals = {}

        if self.env_interface is not None:
            try:
                signals.update(
                    {
                        name: bool(value)
                        for name, value in self.env_interface.get_subtask_term_signals().items()
                    }
                )
            except Exception:
                pass

        if self.task_name.startswith("CoffeePreparation"):
            partial_metrics = self.env._get_partial_task_metrics()
            return {
                "mug_grasp": bool(
                    partial_metrics.get("mug_grasp", signals.get("mug_grasp", False))
                ),
                "mug_place": bool(
                    partial_metrics.get("mug_place", signals.get("mug_place", False))
                ),
                "drawer_open": bool(signals.get("drawer_open", False)),
                "pod_grasp": bool(
                    partial_metrics.get("grasp", signals.get("pod_grasp", False))
                ),
                "pod_insert": bool(partial_metrics.get("insertion", False)),
                # The lid starts closed, so the raw lid check would complete this
                # stage at reset. Task success also requires the mug and inserted
                # pod to be in place, making it a valid final lid-closure signal.
                "lid_closed": bool(
                    partial_metrics.get("task", self.env._check_success())
                ),
            }

        if self.task_name.startswith("MugCleanup"):
            object_rot = self.env.sim.data.body_xmat[
                self.env.obj_body_id["object"]
            ].reshape(3, 3)
            object_upright = bool((1.0 - object_rot[2, 2]) < 1e-3)
            object_in_drawer = bool(
                self.env.check_contact(
                    "DrawerObject_drawer_bottom",
                    self.env.cleanup_object,
                )
            )
            mug_placed = object_in_drawer and object_upright
            return {
                "drawer_open": bool(signals.get("open", False)),
                "mug_grasp": bool(signals.get("grasp", False)),
                "mug_place": mug_placed,
                # The drawer starts closed. Final task success gates closure on
                # the mug already being upright and inside the drawer.
                "drawer_closed": bool(self.env._check_success()),
            }

        if not signals and hasattr(self.env, "_get_partial_task_metrics"):
            partial_metrics = self.env._get_partial_task_metrics()
            signals.update(
                {
                    name: bool(value)
                    for name, value in partial_metrics.items()
                    if name != "task"
                }
            )

        signals["task"] = bool(self.env._check_success())
        return signals

    def _infer_subgoal_order(self) -> list[str]:
        stage_names = list(self._get_stage_signals().keys())
        if "task" in stage_names:
            stage_names = [name for name in stage_names if name != "task"] + ["task"]
        return stage_names

    def _ensure_subgoal_tracking_initialized(self) -> None:
        if not self._subgoal_order:
            self._subgoal_order = self._infer_subgoal_order()
        for stage_name in self._subgoal_order:
            self._completed_stage_flags.setdefault(stage_name, False)
            self._stage_completion_steps.setdefault(stage_name, None)

    def _record_stage_progress(self) -> dict[str, bool]:
        self._ensure_subgoal_tracking_initialized()
        stage_signals = self._get_stage_signals()
        for stage_name in self._subgoal_order:
            is_complete = bool(stage_signals.get(stage_name, False))
            if is_complete and not self._completed_stage_flags[stage_name]:
                self._completed_stage_flags[stage_name] = True
                self._stage_completion_steps[stage_name] = self._episode_steps

        completion_fraction = float(
            np.mean(list(self._completed_stage_flags.values()))
        ) if self._completed_stage_flags else 0.0
        self._completion_fraction_history.append(completion_fraction)
        return dict(self._completed_stage_flags)
    
def test():
    print("Creating MimicGenEnvWrapper for CoffeePreparation_D0...")
    env = MimicGenEnvWrapper(task_name="CoffeePreparation_D0", robosuite_kwargs={"has_renderer": True})

    print(f"Action dim: {env.action_dim}")
    print(f"Obs keys: {env.obs_keys}")

    # Reset and run a few steps
    print("\nResetting environment...")
    obs = env.reset()
    print(f"Low dim Obs shape: {obs['obs'].shape}")
    print(f"Image dim Obs shape: {obs['agentview_image'].shape}")

    # Run 10 random steps
    for i in range(10):
        action = np.random.uniform(-1, 1, env.action_dim)
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")
        if done:
            print("Episode finished!")
            break

    env.close()
    print("✓ Wrapper test passed!")


def generate_mimicgen_dataset(
    task_name: str,
    output_dir: str = "/tmp/core_datasets",
    num_traj: int = 100,
    guarantee: bool = True,
    collect_rgb: bool = True,
    camera_names: list[str] | None = None,
    camera_height: int = 84,
    camera_width: int = 84,
) -> str:
    """
    Generate a MimicGen dataset using the core data generation pipeline.

    Downloads source dataset → prepares it → generates diverse trajectories.

    Args:
        task_name: Task name (e.g., "coffee", "stack", "threading")
        output_dir: Where to save generated datasets
        num_traj: Number of trajectories to generate (if guarantee=False)
        guarantee: If True, generate until success_count is reached
        collect_rgb: If True, store rendered camera observations in the HDF5
        camera_names: Cameras to render when collect_rgb=True
        camera_height: Height for rendered camera observations
        camera_width: Width for rendered camera observations

    Returns:
        Path to generated dataset

    Example:
        dataset_path = generate_mimicgen_dataset("coffee_preparation")
        # Use in config: task.dataset.dataset_path=${dataset_path}
    """
    import subprocess
    import os

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ############# Download source dataset
    print(f"\n{'='*60}")
    print(f"Step 1: Downloading source {task_name} dataset")
    print('='*60)
    result = subprocess.run(
        [
            sys.executable, "-m", "mimicgen.scripts.download_datasets",
            "--dataset_type", "source",
            "--tasks", task_name,
        ],
        cwd=str(_ENV_DIR / "mimicgen"),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download source dataset for {task_name}")

    ############# Prepare source dataset
    print(f"\n{'='*60}")
    print(f"Step 2: Preparing source {task_name} dataset")
    print('='*60)
    src_dataset = f"datasets/source/{task_name}.hdf5"
    env_interface = f"MG_{task_name.title().replace('_', '')}"

    result = subprocess.run(
        [
            sys.executable, "-m", "mimicgen.scripts.prepare_src_dataset",
            "--dataset", src_dataset,
            "--env_interface", env_interface,
            "--env_interface_type", "robosuite",
        ],
        cwd=str(_ENV_DIR / "mimicgen"),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare source dataset for {task_name}")

    ############## Generate dataset
    print(f"\n{'='*60}")
    print(f"Step 3: Generating diverse {task_name} trajectories")
    print('='*60)
    print(f"num_traj={num_traj}, guarantee={guarantee}")
    print('This may take a while...')

    import json
    import tempfile

    mimicgen_dir = str(_ENV_DIR / "mimicgen")

    # Load the per-task template config
    template_path = (
        _ENV_DIR / "mimicgen" / "mimicgen" / "exps" / "templates" / "robosuite"
        / f"{task_name}.json"
    )
    with open(template_path) as _f:
        mg_cfg = json.load(_f)

    # Fill in required null fields
    mg_cfg["experiment"]["source"]["dataset_path"] = os.path.join(
        mimicgen_dir, "datasets", "source", f"{task_name}.hdf5"
    )
    mg_cfg["experiment"]["generation"]["path"] = output_dir
    mg_cfg["experiment"]["generation"]["num_trials"] = num_traj
    mg_cfg["experiment"]["generation"]["guarantee"] = guarantee
    if collect_rgb:
        camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
    else:
        camera_names = []
    mg_cfg["obs"]["collect_obs"] = True
    mg_cfg["obs"]["camera_names"] = camera_names
    mg_cfg["obs"]["camera_height"] = camera_height
    mg_cfg["obs"]["camera_width"] = camera_width

    # Write modified config to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as _cfg_f:
        json.dump(mg_cfg, _cfg_f)
        config_path = _cfg_f.name

    # Run generation script with the proper --config argument
    result = subprocess.run(
        [
            sys.executable, "-m", "mimicgen.scripts.generate_dataset",
            "--config", config_path,
            "--auto-remove-exp",
        ],
        cwd=mimicgen_dir,
        check=False,
    )

    if result.returncode != 0:
        print(f"⚠️  Data generation had issues, but may still have created dataset")

    # Return expected output path
    output_path = f"{output_dir}/{task_name}/demo_src_{task_name}_task_D0/demo.hdf5"
    return output_path


def list_available_tasks() -> dict[str, str]:
    """
    List available MimicGen tasks and their descriptions.

    Returns:
        Dict mapping task names to descriptions
    """
    tasks = {
        "coffee": "Simple coffee task: grasp pod, insert in machine",
        "coffee_preparation": "Complex 5-step coffee preparation task",
        "stack": "Stack 3 blocks on top of each other",
        "threading": "Thread a rope through a sequence of pegs",
        "pick_place": "Pick up and place objects",
        "nut_assembly": "Assemble nuts on pegs",
        "mug_cleanup": "Clean up mugs from table",
        "square": "Manipulate object into square shape",
        "three_piece_assembly": "Assemble 3-piece structure",
    }

    print("\nAvailable MimicGen tasks:")
    print("-" * 60)
    for task, desc in tasks.items():
        print(f"  {task:20s} → {desc}")
    print("-" * 60)

    return tasks


if __name__ == "__main__":
    generate_mimicgen_dataset(
        task_name="coffee_preparation",
        output_dir = "/Users/aminamazlin/Desktop/thesis-test-1/thesis/dataset",
        num_traj = 1000,
        guarantee=True,
    )
