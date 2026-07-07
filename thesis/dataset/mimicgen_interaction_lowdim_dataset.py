from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .mimicgen_lowdim_dataset import (
    _array_to_stats,
    _chunk_for_array,
    _default_cache_path,
    _limits_normalizer_from_stat,
    _looks_like_zarr_path,
    _move_to_device,
    _sorted_demo_keys,
    _validate_demo,
)
from .mimicgen_subgoal_lowdim_dataset import (
    PICK_PLACE_SUBGOAL_SIGNALS,
    compute_subgoal_stage_progress,
    resolve_subgoal_signals,
)
from ..model.common.normalizer import LinearNormalizer
from ..utils.replay_buffer import ReplayBuffer
from ..utils.sampler import SequenceSampler, downsample_mask, get_val_mask


DEFAULT_INTERACTION_OBJECTS = ("milk", "bread", "cereal", "can")

SIGNAL_OBJECT_ALIASES = {
    "pod": "coffee_pod",
    "machine": "coffee_machine",
}

SIGNAL_OBJECT_EXACT = {
    "mug_grasp": "mug",
    "drawer_open": "drawer",
    "pod_grasp": "coffee_pod",
    "pod_insert": "coffee_machine",
    "lid_closed": "coffee_machine",
    "open": "drawer",
    "grasp": "object",
    "drawer_closed": "drawer",
}


def _signal_to_object(signal_name: str, object_names: list[str]) -> str:
    if signal_name == "mug_place":
        if "coffee_machine" in object_names:
            return "coffee_machine"
        if "object" in object_names:
            return "object"
    exact = SIGNAL_OBJECT_EXACT.get(signal_name)
    if exact is not None and exact in object_names:
        return exact
    for object_name in object_names:
        if object_name in signal_name:
            return object_name
    for alias, object_name in SIGNAL_OBJECT_ALIASES.items():
        if alias in signal_name and object_name in object_names:
            return object_name
    raise ValueError(f"Cannot infer active object from signal {signal_name!r}.")


def _stage_to_object_labels(
    stage: np.ndarray,
    subgoal_signals: list[str],
    object_names: list[str],
    final_object_name: str | None = None,
) -> np.ndarray:
    object_to_idx = {name: idx for idx, name in enumerate(object_names)}
    stage_to_object: list[int] = []
    for signal in subgoal_signals:
        object_name = _signal_to_object(signal, object_names)
        if object_name not in object_to_idx:
            raise ValueError(
                f"Subgoal signal {signal!r} maps to object {object_name!r}, "
                f"but object_names={object_names}."
            )
        stage_to_object.append(object_to_idx[object_name])

    if final_object_name is not None:
        if final_object_name not in object_to_idx:
            raise ValueError(
                f"final_object_name={final_object_name!r} is not in "
                f"object_names={object_names}."
            )
        final_object = object_to_idx[final_object_name]
    else:
        final_object = stage_to_object[-1]
    labels = np.full_like(stage, final_object, dtype=np.int64)
    for stage_idx, object_idx in enumerate(stage_to_object):
        labels[stage == stage_idx] = object_idx
    labels[stage >= len(stage_to_object)] = final_object
    return labels.astype(np.int64)


def _read_positions_from_poses(dataset: h5py.Dataset, *, name: str) -> np.ndarray:
    poses = np.asarray(dataset[:], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected {name} shape (T, 4, 4), got {poses.shape}.")
    return poses[:, :3, 3].astype(np.float32)


def compute_interaction_state_targets(
    demo: h5py.Group,
    *,
    subgoal_signals: list[str] | tuple[str, ...] = PICK_PLACE_SUBGOAL_SIGNALS,
    object_names: list[str] | tuple[str, ...] = DEFAULT_INTERACTION_OBJECTS,
    final_object_name: str | None = None,
    motion_delta_offset: int = 8,
) -> dict[str, np.ndarray]:
    """Derive lowdim progress and interaction targets.

    The labels describe active object, contact/reach geometry, placement
    geometry, and future active-object motion.
    """
    stage, progress, _ = compute_subgoal_stage_progress(
        demo,
        subgoal_signals=list(subgoal_signals),
    )
    object_names = [str(name) for name in object_names]
    object_labels = _stage_to_object_labels(
        stage,
        subgoal_signals=list(subgoal_signals),
        object_names=object_names,
        final_object_name=final_object_name,
    )

    datagen = demo.get("datagen_info")
    if datagen is None:
        raise KeyError(f"Missing {demo.name}/datagen_info.")
    object_group = datagen.get("object_poses")
    if object_group is None:
        raise KeyError(f"Missing {demo.name}/datagen_info/object_poses.")
    if "eef_pose" not in datagen:
        raise KeyError(f"Missing {demo.name}/datagen_info/eef_pose.")
    if "target_pose" not in datagen:
        raise KeyError(f"Missing {demo.name}/datagen_info/target_pose.")

    object_positions = []
    for object_name in object_names:
        if object_name not in object_group:
            raise KeyError(
                f"Missing object pose {object_name!r} in {demo.name}; "
                f"available={sorted(object_group.keys())}."
            )
        object_positions.append(
            _read_positions_from_poses(
                object_group[object_name],
                name=f"{demo.name}/datagen_info/object_poses/{object_name}",
            )
        )
    object_positions = np.stack(object_positions, axis=1)

    horizon = int(stage.shape[0])
    if object_positions.shape[0] != horizon:
        raise ValueError(
            f"Object pose length {object_positions.shape[0]} does not match "
            f"stage length {horizon} in {demo.name}."
        )

    eef_pos = _read_positions_from_poses(datagen["eef_pose"], name=f"{demo.name}/datagen_info/eef_pose")
    target_pos = _read_positions_from_poses(
        datagen["target_pose"],
        name=f"{demo.name}/datagen_info/target_pose",
    )
    if eef_pos.shape[0] != horizon or target_pos.shape[0] != horizon:
        raise ValueError(f"EEF/target pose length mismatch in {demo.name}.")

    active_object_pos = object_positions[np.arange(horizon), object_labels]
    contact_delta = active_object_pos - eef_pos
    place_delta = target_pos - active_object_pos

    offset = max(int(motion_delta_offset), 1)
    future_idx = np.arange(horizon) + offset
    motion_mask = future_idx < horizon
    future_idx = np.minimum(future_idx, horizon - 1)
    future_active_pos = object_positions[future_idx, object_labels]
    motion_delta = future_active_pos - active_object_pos

    targets = {
        "interaction_stage": stage.astype(np.int64),
        "interaction_progress": progress.astype(np.float32),
        "interaction_object": object_labels.astype(np.int64),
        "interaction_contact_delta": contact_delta.astype(np.float32),
        "interaction_place_delta": place_delta.astype(np.float32),
        "interaction_motion_delta": motion_delta.astype(np.float32),
        "interaction_motion_mask": motion_mask.astype(np.float32),
    }
    return targets


def _convert_hdf5_to_interaction_state_replay(
    dataset_path: str,
    obs_keys: list[str],
    subgoal_signals: list[str],
    object_names: list[str],
    final_object_name: str | None,
    motion_delta_offset: int,
    root: zarr.Group,
    chunk_length: int,
    compressor: str,
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr(root=root)

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(
                f"Expected robomimic/MimicGen HDF5 with a 'data' group: {dataset_path}"
            )

        data_group = f["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if not demo_keys:
            raise ValueError(f"No demo_* groups found in {dataset_path}/data")

        for demo_key in demo_keys:
            demo = data_group[demo_key]
            _validate_demo(demo, obs_keys)

            actions = demo["actions"][:].astype(np.float32)
            obs = np.concatenate(
                [demo["obs"][key][:].astype(np.float32) for key in obs_keys],
                axis=-1,
            )
            targets = compute_interaction_state_targets(
                demo,
                subgoal_signals=subgoal_signals,
                object_names=object_names,
                final_object_name=final_object_name,
                motion_delta_offset=motion_delta_offset,
            )
            episode = {
                "obs": obs,
                "action": actions,
                **targets,
            }
            replay_buffer.add_episode(
                episode,
                chunks={
                    key: _chunk_for_array(value, chunk_length)
                    for key, value in episode.items()
                },
                compressors=compressor,
            )

    return replay_buffer


def prepare_mimicgen_interaction_state_lowdim_cache(
    *,
    dataset_path: str,
    obs_keys: list[str],
    subgoal_signals: str | list[str] | tuple[str, ...],
    object_names: list[str] | tuple[str, ...] = DEFAULT_INTERACTION_OBJECTS,
    final_object_name: str | None = None,
    motion_delta_offset: int = 8,
    cache_zarr_path: str,
    chunk_length: int = 1000,
    overwrite: bool = False,
) -> str:
    signals = resolve_subgoal_signals(subgoal_signals)
    object_names = [str(name) for name in object_names]
    cache_path = os.path.expanduser(cache_zarr_path)
    if os.path.exists(cache_path):
        if not overwrite:
            return cache_path
        shutil.rmtree(cache_path)

    tmp_path = cache_path + ".tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    parent_dir = os.path.dirname(cache_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    root = zarr.group(store=zarr.DirectoryStore(tmp_path), overwrite=True)
    _convert_hdf5_to_interaction_state_replay(
        dataset_path=os.path.expanduser(dataset_path),
        obs_keys=list(obs_keys),
        subgoal_signals=signals,
        object_names=object_names,
        final_object_name=final_object_name,
        motion_delta_offset=int(motion_delta_offset),
        root=root,
        chunk_length=int(chunk_length),
        compressor="disk",
    )
    os.replace(tmp_path, cache_path)
    return cache_path


INTERACTION_STATE_KEYS = [
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


def _load_interaction_state_replay_buffer(
    dataset_path: str,
    obs_keys: list[str],
    subgoal_signals: list[str],
    object_names: list[str],
    final_object_name: str | None,
    motion_delta_offset: int,
    use_cache: bool,
    cache_zarr_path: str | None,
    chunk_length: int,
) -> ReplayBuffer:
    replay_keys = list(INTERACTION_STATE_KEYS)
    if _looks_like_zarr_path(dataset_path):
        return ReplayBuffer.copy_from_path(dataset_path, keys=replay_keys)

    if not use_cache:
        root = zarr.group(store=zarr.MemoryStore())
        return _convert_hdf5_to_interaction_state_replay(
            dataset_path=dataset_path,
            obs_keys=obs_keys,
            subgoal_signals=subgoal_signals,
            object_names=object_names,
            final_object_name=final_object_name,
            motion_delta_offset=motion_delta_offset,
            root=root,
            chunk_length=chunk_length,
            compressor="default",
        )

    cache_path = cache_zarr_path or str(
        Path(_default_cache_path(dataset_path)).with_suffix(".interaction_state.zarr")
    )
    if os.path.exists(cache_path):
        print(f"Loading MimicGen interaction-state lowdim replay cache: {cache_path}")
        return ReplayBuffer.copy_from_path(cache_path, keys=replay_keys)

    print(f"Creating MimicGen interaction-state lowdim replay cache: {cache_path}")
    prepare_mimicgen_interaction_state_lowdim_cache(
        dataset_path=dataset_path,
        obs_keys=obs_keys,
        subgoal_signals=subgoal_signals,
        object_names=object_names,
        final_object_name=final_object_name,
        motion_delta_offset=motion_delta_offset,
        cache_zarr_path=cache_path,
        chunk_length=chunk_length,
        overwrite=False,
    )
    return ReplayBuffer.copy_from_path(cache_path, keys=replay_keys)


class MimicGenInteractionStateLowdimDataset(Dataset):
    """MimicGen lowdim dataset with progress and interaction labels."""

    def __init__(
        self,
        dataset_path: str,
        obs_keys: list[str],
        subgoal_signals: str | list[str] | tuple[str, ...],
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        object_names: list[str] | tuple[str, ...] = DEFAULT_INTERACTION_OBJECTS,
        final_object_name: str | None = None,
        motion_delta_offset: int = 8,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.02,
        train: bool = True,
        max_train_episodes: int | None = None,
        use_cache: bool = True,
        cache_zarr_path: str | None = None,
        chunk_length: int = 1000,
    ) -> None:
        self.dataset_path = dataset_path
        self.obs_keys = list(obs_keys)
        self.subgoal_signals = resolve_subgoal_signals(subgoal_signals)
        self.object_names = [str(name) for name in object_names]
        self.final_object_name = final_object_name
        self.motion_delta_offset = int(motion_delta_offset)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.train = bool(train)
        self.max_train_episodes = max_train_episodes
        self.use_cache = bool(use_cache)
        self.cache_zarr_path = cache_zarr_path
        self.chunk_length = int(chunk_length)
        self.keys = list(INTERACTION_STATE_KEYS)

        self.replay_buffer = _load_interaction_state_replay_buffer(
            dataset_path=self.dataset_path,
            obs_keys=self.obs_keys,
            subgoal_signals=self.subgoal_signals,
            object_names=self.object_names,
            final_object_name=self.final_object_name,
            motion_delta_offset=self.motion_delta_offset,
            use_cache=self.use_cache,
            cache_zarr_path=self.cache_zarr_path,
            chunk_length=self.chunk_length,
        )
        self.val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        self.train_mask = downsample_mask(
            mask=~self.val_mask,
            max_n=max_train_episodes,
            seed=self.seed,
        )
        episode_mask = self.train_mask if self.train else self.val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.keys,
            episode_mask=episode_mask,
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int | np.ndarray) -> dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        if isinstance(idx, np.ndarray):
            return default_collate([self[int(i)] for i in idx])
        if not isinstance(idx, (int, np.integer)):
            raise ValueError(idx)

        sample = self.sampler.sample_sequence(int(idx))
        result = {
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
        return result

    def get_sampler_episode_ids(self) -> np.ndarray:
        return self.sampler.episode_ids

    def postprocess(
        self,
        samples: dict[str, Any],
        device: torch.device | str,
    ) -> dict[str, Any]:
        return _move_to_device(samples, device)

    def get_validation_dataset(self) -> "MimicGenInteractionStateLowdimDataset":
        val_set = self.__class__.__new__(self.__class__)
        val_set.__dict__ = self.__dict__.copy()
        val_set.train = False
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.keys,
            episode_mask=self.val_mask,
        )
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["obs"] = _limits_normalizer_from_stat(
            _array_to_stats(self.replay_buffer["obs"][:])
        )
        normalizer["action"] = _limits_normalizer_from_stat(
            _array_to_stats(self.replay_buffer["action"][:])
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a MimicGen lowdim zarr cache with interaction-state labels."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--cache-zarr-path", required=True)
    parser.add_argument("--obs-keys", nargs="+", required=True)
    parser.add_argument(
        "--subgoal-signals",
        nargs="+",
        default=["pick_place"],
        help="Signal names or one preset: pick_place, coffee_prep.",
    )
    parser.add_argument(
        "--object-names",
        nargs="+",
        default=list(DEFAULT_INTERACTION_OBJECTS),
    )
    parser.add_argument("--final-object-name", default=None)
    parser.add_argument("--motion-delta-offset", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    subgoal_signals: str | list[str]
    if len(args.subgoal_signals) == 1:
        subgoal_signals = args.subgoal_signals[0]
    else:
        subgoal_signals = args.subgoal_signals
    cache_path = prepare_mimicgen_interaction_state_lowdim_cache(
        dataset_path=args.dataset_path,
        obs_keys=args.obs_keys,
        subgoal_signals=subgoal_signals,
        object_names=args.object_names,
        final_object_name=args.final_object_name,
        motion_delta_offset=int(args.motion_delta_offset),
        cache_zarr_path=args.cache_zarr_path,
        chunk_length=args.chunk_length,
        overwrite=bool(args.overwrite),
    )
    print(f"Wrote MimicGen interaction-state lowdim cache: {cache_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()
