from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ..model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ..utils.replay_buffer import ReplayBuffer
from ..utils.sampler import SequenceSampler, downsample_mask, get_val_mask


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def key_fn(name: str) -> tuple[int, str]:
        if name.startswith("demo_"):
            try:
                return (int(name.split("_")[-1]), name)
            except ValueError:
                pass
        return (10**18, name)

    return sorted(
        [name for name in data_group.keys() if name.startswith("demo_")],
        key=key_fn,
    )


def _move_to_device(data: Any, device: torch.device | str) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    return data


def _array_to_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "min": array.min(axis=0).astype(np.float32),
        "max": array.max(axis=0).astype(np.float32),
        "mean": array.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": array.std(axis=0, dtype=np.float64).astype(np.float32),
    }


def _identity_normalizer_from_stat(
    stat: dict[str, np.ndarray],
) -> SingleFieldLinearNormalizer:
    scale = np.ones_like(stat["min"], dtype=np.float32)
    offset = np.zeros_like(stat["min"], dtype=np.float32)
    return SingleFieldLinearNormalizer.create_manual(scale, offset, stat)


def _limits_normalizer_from_stat(
    stat: dict[str, np.ndarray],
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-7,
) -> SingleFieldLinearNormalizer:
    input_min = stat["min"]
    input_max = stat["max"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range = input_range.copy()
    input_range[ignore_dim] = output_max - output_min

    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return SingleFieldLinearNormalizer.create_manual(
        scale.astype(np.float32),
        offset.astype(np.float32),
        stat,
    )


def _default_cache_path(dataset_path: str) -> str:
    suffix = Path(dataset_path).suffix
    return str(Path(dataset_path).with_suffix(suffix + ".zarr"))


def _looks_like_zarr_path(path: str) -> bool:
    return path.endswith(".zarr") or os.path.isdir(path)


def _chunk_for_array(array: np.ndarray, chunk_length: int) -> tuple[int, ...]:
    return (min(int(chunk_length), int(array.shape[0])), *array.shape[1:])


def _validate_demo(demo: h5py.Group, obs_keys: list[str]) -> None:
    if "actions" not in demo:
        raise KeyError(f"Missing actions in {demo.name}")
    if "obs" not in demo:
        raise KeyError(f"Missing obs group in {demo.name}")

    horizon = int(demo["actions"].shape[0])
    for key in obs_keys:
        if key not in demo["obs"]:
            raise KeyError(f"Missing observation key {demo.name}/obs/{key}")
        if int(demo["obs"][key].shape[0]) != horizon:
            raise ValueError(
                f"{demo.name}/obs/{key} length {demo['obs'][key].shape[0]} "
                f"does not match actions length {horizon}."
            )


def _convert_hdf5_to_replay(
    dataset_path: str,
    obs_keys: list[str],
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
            episode = {
                "obs": obs,
                "action": actions,
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


def _load_replay_buffer(
    dataset_path: str,
    obs_keys: list[str],
    use_cache: bool,
    cache_zarr_path: str | None,
    chunk_length: int,
) -> ReplayBuffer:
    keys = ["obs", "action"]
    if _looks_like_zarr_path(dataset_path):
        return ReplayBuffer.copy_from_path(dataset_path, keys=keys)

    if not use_cache:
        root = zarr.group(store=zarr.MemoryStore())
        return _convert_hdf5_to_replay(
            dataset_path=dataset_path,
            obs_keys=obs_keys,
            root=root,
            chunk_length=chunk_length,
            compressor="default",
        )

    cache_path = cache_zarr_path or _default_cache_path(dataset_path)
    if os.path.exists(cache_path):
        print(f"Loading MimicGen lowdim replay cache: {cache_path}")
        return ReplayBuffer.copy_from_path(cache_path, keys=keys)

    tmp_path = cache_path + ".tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    parent_dir = os.path.dirname(cache_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    print(f"Creating MimicGen lowdim replay cache: {cache_path}")
    root = zarr.group(store=zarr.DirectoryStore(tmp_path), overwrite=True)
    _convert_hdf5_to_replay(
        dataset_path=dataset_path,
        obs_keys=obs_keys,
        root=root,
        chunk_length=chunk_length,
        compressor="disk",
    )
    os.replace(tmp_path, cache_path)
    return ReplayBuffer.copy_from_path(cache_path, keys=keys)


class MimicGenLowdimDataset(Dataset):
    """MimicGen lowdim dataset backed by ReplayBuffer and SequenceSampler.

    HDF5 files are converted to a zarr replay-buffer cache on first use. The
    stateful long-horizon workspace uses get_sampler_episode_ids() to build
    temporal chains without crossing episode boundaries.
    """

    def __init__(
        self,
        dataset_path: str,
        obs_keys: list[str],
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
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
        self.keys = ["obs", "action"]

        self.replay_buffer = _load_replay_buffer(
            dataset_path=self.dataset_path,
            obs_keys=self.obs_keys,
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
        return {
            "obs": torch.from_numpy(sample["obs"].astype(np.float32)),
            "action": torch.from_numpy(sample["action"].astype(np.float32)),
        }

    def get_sampler_episode_ids(self) -> np.ndarray:
        return self.sampler.episode_ids

    def postprocess(
        self,
        samples: dict[str, Any],
        device: torch.device | str,
    ) -> dict[str, Any]:
        return _move_to_device(samples, device)

    def get_validation_dataset(self) -> "MimicGenLowdimDataset":
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
        # Identity action normalization is the robomimic-style default for
        # already normalized relative controller actions. Keep this here so
        # experiments can switch back easily.
        # normalizer["action"] = _identity_normalizer_from_stat(
        #     _array_to_stats(self.replay_buffer["action"][:])
        # )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])
