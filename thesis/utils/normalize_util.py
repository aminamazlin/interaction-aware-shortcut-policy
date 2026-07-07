from __future__ import annotations

import numpy as np
import torch

from thesis.model.common.normalizer import SingleFieldLinearNormalizer


def array_to_stats(array: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
    """Return min/max/mean/std statistics compatible with local normalizers."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    return {
        "min": array.min(axis=0).astype(np.float32),
        "max": array.max(axis=0).astype(np.float32),
        "mean": array.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": array.std(axis=0, dtype=np.float64).astype(np.float32),
    }


def _as_tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.as_tensor(value, dtype=torch.float32)


def get_range_normalizer_from_stat(
    stat: dict[str, np.ndarray | torch.Tensor],
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
) -> SingleFieldLinearNormalizer:
    """Create a limits normalizer from precomputed input statistics."""
    input_min = _as_tensor(stat["min"]).flatten()
    input_max = _as_tensor(stat["max"]).flatten()
    input_mean = _as_tensor(stat["mean"]).flatten()
    input_std = _as_tensor(stat["std"]).flatten()

    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min

    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    input_stats = {
        "min": input_min,
        "max": input_max,
        "mean": input_mean,
        "std": input_std,
    }
    return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)


def get_identity_normalizer_from_stat(
    stat: dict[str, np.ndarray | torch.Tensor],
) -> SingleFieldLinearNormalizer:
    """Create an identity normalizer while preserving input statistics."""
    first = _as_tensor(next(iter(stat.values()))).flatten()
    scale = torch.ones_like(first)
    offset = torch.zeros_like(first)
    input_stats = {key: _as_tensor(value).flatten() for key, value in stat.items()}
    return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)


def get_image_range_normalizer() -> SingleFieldLinearNormalizer:
    """Normalize uint8-style image observations from [0, 255] to [0, 1]."""
    stat = {
        "min": torch.tensor([0.0], dtype=torch.float32),
        "max": torch.tensor([255.0], dtype=torch.float32),
        "mean": torch.tensor([0.0], dtype=torch.float32),
        "std": torch.tensor([1.0], dtype=torch.float32),
    }
    scale = torch.tensor([1.0 / 255.0], dtype=torch.float32)
    offset = torch.tensor([0.0], dtype=torch.float32)
    return SingleFieldLinearNormalizer.create_manual(scale, offset, stat)


__all__ = [
    "array_to_stats",
    "get_identity_normalizer_from_stat",
    "get_image_range_normalizer",
    "get_range_normalizer_from_stat",
]
