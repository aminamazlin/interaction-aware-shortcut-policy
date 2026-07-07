from __future__ import annotations

from typing import Any, Dict
import copy
import pathlib

import numpy as np
import torch
from tqdm import tqdm

from thesis.dataset.base_dataset import BaseLowdimDataset
from thesis.env.kitchen.base import (
    BONUS_THRESH,
    OBS_ELEMENT_GOALS,
    OBS_ELEMENT_INDICES,
)
from thesis.env.kitchen.kitchen_util import parse_mjl_logs

from thesis.model.common.normalizer import LinearNormalizer
from thesis.utils.pytorch_util import dict_apply
from thesis.utils.replay_buffer import ReplayBuffer
from thesis.utils.sampler import SequenceSampler, get_val_mask


KITCHEN_INTERACTION_TASK_NAMES = (
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
)
KITCHEN_INTERACTION_FINAL_STAGE = len(KITCHEN_INTERACTION_TASK_NAMES)
KITCHEN_INTERACTION_TASK_TO_IDX = {
    name: idx for idx, name in enumerate(KITCHEN_INTERACTION_TASK_NAMES)
}
KITCHEN_FOLDER_TASK_ALIASES = {
    "bottomknob": "bottom burner",
    "topknob": "top burner",
    "switch": "light switch",
    "slide": "slide cabinet",
    "hinge": "hinge cabinet",
    "microwave": "microwave",
    "kettle": "kettle",
}
KITCHEN_FOLDER_PREFIXES = {"friday", "postcorl"}
KITCHEN_INTERACTION_STATE_KEYS = [
    "obs",
    "action",
    "interaction_stage",
    "interaction_progress",
    "interaction_object",
    "interaction_contact_delta",
    "interaction_place_delta",
    "interaction_motion_delta",
    "interaction_motion_mask",
]


def infer_kitchen_tasks_from_mjl_path(mjl_path: str | pathlib.Path) -> list[str]:
    """Infer the semantic Kitchen subtasks encoded in a demo folder name."""
    folder_name = pathlib.Path(mjl_path).parent.name
    tokens = folder_name.split("_")
    if tokens and tokens[0] in KITCHEN_FOLDER_PREFIXES:
        tokens = tokens[1:]

    tasks: list[str] = []
    for token in tokens:
        task_name = KITCHEN_FOLDER_TASK_ALIASES.get(token)
        if task_name is not None and task_name not in tasks:
            tasks.append(task_name)

    if not tasks:
        raise ValueError(
            f"Could not infer Kitchen tasks from folder {folder_name!r}. "
            f"Known tokens: {sorted(KITCHEN_FOLDER_TASK_ALIASES)}."
        )
    return tasks


def _pad_or_trim_to_three(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    out = np.zeros((values.shape[0], 3), dtype=np.float32)
    width = min(values.shape[1], 3)
    out[:, :width] = values[:, :width]
    return out


def _completion_boundary(qpos: np.ndarray, task_name: str, horizon: int) -> int:
    element_idx = OBS_ELEMENT_INDICES[task_name]
    element_goal = np.asarray(OBS_ELEMENT_GOALS[task_name], dtype=np.float32)
    distance = np.linalg.norm(
        qpos[:, element_idx].astype(np.float32) - element_goal[None],
        axis=-1,
    )
    complete = np.nonzero(distance < float(BONUS_THRESH))[0]
    if complete.size <= 0:
        return horizon
    return int(min(max(int(complete[0]) + 1, 1), horizon))


def compute_franka_kitchen_interaction_state_targets(
    qpos: np.ndarray,
    *,
    task_order: list[str] | tuple[str, ...],
    motion_delta_offset: int = 8,
) -> dict[str, np.ndarray]:
    """Derive interaction progress and affordance targets from Kitchen state.

    For Franka Kitchen lowdim data, the closest available interaction analogue
    is each active fixture's state relative to its goal state.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    horizon = int(qpos.shape[0])
    if horizon <= 0:
        raise ValueError("Cannot derive Kitchen interaction targets for an empty episode.")

    task_order = [str(task) for task in task_order]
    unknown = [task for task in task_order if task not in KITCHEN_INTERACTION_TASK_TO_IDX]
    if unknown:
        raise ValueError(f"Unknown Kitchen interaction tasks: {unknown}.")

    events: list[tuple[int, int, str]] = []
    for task_name in task_order:
        task_idx = KITCHEN_INTERACTION_TASK_TO_IDX[task_name]
        events.append(
            (
                _completion_boundary(qpos, task_name, horizon),
                int(task_idx),
                task_name,
            )
        )
    events.sort(key=lambda item: (item[0], item[1]))

    stage = np.full((horizon,), KITCHEN_INTERACTION_FINAL_STAGE, dtype=np.int64)
    progress = np.zeros((horizon,), dtype=np.float32)
    start = 0
    last_task_idx = int(events[-1][1]) if events else 0
    for boundary, task_idx, _task_name in events:
        if boundary <= start:
            continue
        end = int(min(max(boundary, start + 1), horizon))
        if start >= horizon:
            break
        length = max(end - start, 1)
        stage[start:end] = int(task_idx)
        if length == 1:
            progress[start:end] = 1.0
        else:
            progress[start:end] = np.linspace(
                0.0,
                1.0,
                num=length,
                endpoint=False,
                dtype=np.float32,
            )
        start = end
        last_task_idx = int(task_idx)

    if start < horizon:
        length = max(horizon - start, 1)
        stage[start:horizon] = KITCHEN_INTERACTION_FINAL_STAGE
        if length == 1:
            progress[start:horizon] = 1.0
        else:
            progress[start:horizon] = np.linspace(
                0.0,
                1.0,
                num=length,
                endpoint=False,
                dtype=np.float32,
            )

    object_labels = np.full((horizon,), last_task_idx, dtype=np.int64)
    for task_name, task_idx in KITCHEN_INTERACTION_TASK_TO_IDX.items():
        object_labels[stage == task_idx] = int(task_idx)
    object_labels[stage >= KITCHEN_INTERACTION_FINAL_STAGE] = last_task_idx

    current_by_task = []
    goal_by_task = []
    for task_name in KITCHEN_INTERACTION_TASK_NAMES:
        element_idx = OBS_ELEMENT_INDICES[task_name]
        current_by_task.append(_pad_or_trim_to_three(qpos[:, element_idx]))
        goal = np.asarray(OBS_ELEMENT_GOALS[task_name], dtype=np.float32)[None]
        goal_by_task.append(
            _pad_or_trim_to_three(np.repeat(goal, repeats=horizon, axis=0))
        )
    current_by_task = np.stack(current_by_task, axis=1)
    goal_by_task = np.stack(goal_by_task, axis=1)

    row_idx = np.arange(horizon)
    active_current = current_by_task[row_idx, object_labels]
    active_goal = goal_by_task[row_idx, object_labels]
    contact_delta = active_current - active_goal
    place_delta = active_goal - active_current

    offset = max(int(motion_delta_offset), 1)
    future_idx = np.arange(horizon) + offset
    motion_mask = future_idx < horizon
    future_idx = np.minimum(future_idx, horizon - 1)
    future_current = current_by_task[future_idx, object_labels]
    motion_delta = future_current - active_current

    return {
        "interaction_stage": stage.astype(np.int64),
        "interaction_progress": progress.astype(np.float32),
        "interaction_object": object_labels.astype(np.int64),
        "interaction_contact_delta": contact_delta.astype(np.float32),
        "interaction_place_delta": place_delta.astype(np.float32),
        "interaction_motion_delta": motion_delta.astype(np.float32),
        "interaction_motion_mask": motion_mask.astype(np.float32),
    }


def _move_to_device(data: Any, device: torch.device | str) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    return data


def _sampler_episode_ids(sampler: SequenceSampler) -> np.ndarray:
    return np.asarray(sampler.episode_ids, dtype=np.int64)


class _StatefulKitchenDatasetMixin:
    def get_sampler_episode_ids(self) -> np.ndarray:
        return _sampler_episode_ids(self.sampler)

    def postprocess(
        self,
        samples: dict[str, Any],
        device: torch.device | str,
    ) -> dict[str, Any]:
        return _move_to_device(samples, device)


class FrankaKitchenLowdimDataset(_StatefulKitchenDatasetMixin, BaseLowdimDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
    ):
        super().__init__()

        data_directory = pathlib.Path(dataset_dir)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in range(len(masks)):
            eps_len = int(masks[i].sum())
            obs = observations[i, :eps_len].astype(np.float32)
            action = actions[i, :eps_len].astype(np.float32)
            data = {
                "obs": obs,
                "action": action,
            }
            self.replay_buffer.add_episode(data)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=["obs", "action"],
            episode_mask=train_mask,
        )

        self.keys = ["obs", "action"]
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.keys,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "obs": self.replay_buffer["obs"],
            "action": self.replay_buffer["action"],
        }
        if "range_eps" not in kwargs:
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(sample, torch.from_numpy)


class FrankaKitchenMjlLowdimDataset(_StatefulKitchenDatasetMixin, BaseLowdimDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=True,
        robot_noise_ratio=0.0,
        seed=42,
        val_ratio=0.0,
    ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed=seed)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob("*/*.mjl")))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data["qpos"].astype(np.float32)
                obs = np.concatenate(
                    [
                        qpos[:, :9],
                        qpos[:, -21:],
                        np.zeros((len(qpos), 30), dtype=np.float32),
                    ],
                    axis=-1,
                )
                if robot_noise_ratio > 0:
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1.0,
                        high=1.0,
                        size=(obs.shape[0], 30),
                    )
                    obs[:, :30] += noise
                episode = {
                    "obs": obs,
                    "action": data["ctrl"].astype(np.float32),
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=["obs", "action"],
            episode_mask=train_mask,
        )

        self.keys = ["obs", "action"]
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.keys,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "obs": self.replay_buffer["obs"],
            "action": self.replay_buffer["action"],
        }
        if "range_eps" not in kwargs:
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(sample, torch.from_numpy)


class FrankaKitchenMjlInteractionStateLowdimDataset(
    _StatefulKitchenDatasetMixin,
    BaseLowdimDataset,
):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=True,
        robot_noise_ratio=0.0,
        motion_delta_offset=8,
        seed=42,
        val_ratio=0.0,
    ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed=seed)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob("*/*.mjl")))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data["qpos"].astype(np.float32)
                obs = np.concatenate(
                    [
                        qpos[:, :9],
                        qpos[:, -21:],
                        np.zeros((len(qpos), 30), dtype=np.float32),
                    ],
                    axis=-1,
                )
                if robot_noise_ratio > 0:
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1.0,
                        high=1.0,
                        size=(obs.shape[0], 30),
                    )
                    obs[:, :30] += noise
                interaction_targets = compute_franka_kitchen_interaction_state_targets(
                    qpos,
                    task_order=infer_kitchen_tasks_from_mjl_path(mjl_path),
                    motion_delta_offset=motion_delta_offset,
                )
                episode = {
                    "obs": obs,
                    "action": data["ctrl"].astype(np.float32),
                    **interaction_targets,
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=KITCHEN_INTERACTION_STATE_KEYS,
            episode_mask=train_mask,
        )

        self.keys = KITCHEN_INTERACTION_STATE_KEYS
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.keys,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "obs": self.replay_buffer["obs"],
            "action": self.replay_buffer["action"],
        }
        if "range_eps" not in kwargs:
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return {
            "obs": torch.from_numpy(sample["obs"].astype(np.float32)),
            "action": torch.from_numpy(sample["action"].astype(np.float32)),
            "interaction_stage": torch.from_numpy(
                sample["interaction_stage"].astype(np.int64)
            ),
            "interaction_progress": torch.from_numpy(
                sample["interaction_progress"].astype(np.float32)
            ),
            "interaction_object": torch.from_numpy(
                sample["interaction_object"].astype(np.int64)
            ),
            "interaction_contact_delta": torch.from_numpy(
                sample["interaction_contact_delta"].astype(np.float32)
            ),
            "interaction_place_delta": torch.from_numpy(
                sample["interaction_place_delta"].astype(np.float32)
            ),
            "interaction_motion_delta": torch.from_numpy(
                sample["interaction_motion_delta"].astype(np.float32)
            ),
            "interaction_motion_mask": torch.from_numpy(
                sample["interaction_motion_mask"].astype(np.float32)
            ),
        }
