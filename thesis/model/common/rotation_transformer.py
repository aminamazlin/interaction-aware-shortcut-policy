from __future__ import annotations

import numpy as np


class RotationTransformer:
    """Convert rotations between axis-angle and 6D matrix-column format."""

    def __init__(self, from_rep: str, to_rep: str):
        if (from_rep, to_rep) != ("axis_angle", "rotation_6d"):
            raise NotImplementedError(
                "Only axis_angle <-> rotation_6d is supported locally."
            )
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, value):
        return self._matrix_to_6d(self._axis_angle_to_matrix(np.asarray(value)))

    def inverse(self, value):
        return self._matrix_to_axis_angle(self._rotation_6d_to_matrix(np.asarray(value)))

    @staticmethod
    def _normalize(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        return value / np.maximum(norm, eps)

    @classmethod
    def _axis_angle_to_matrix(cls, value: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(value, axis=-1, keepdims=True)
        axis = value / np.maximum(angle, 1e-8)
        x, y, z = np.moveaxis(axis, -1, 0)
        zero = np.zeros_like(x)
        k = np.stack(
            [
                np.stack([zero, -z, y], axis=-1),
                np.stack([z, zero, -x], axis=-1),
                np.stack([-y, x, zero], axis=-1),
            ],
            axis=-2,
        )
        eye = np.broadcast_to(np.eye(3, dtype=value.dtype), k.shape)
        sin = np.sin(angle)[..., None]
        cos = np.cos(angle)[..., None]
        matrix = eye + sin * k + (1.0 - cos) * (k @ k)
        small = (angle[..., 0] < 1e-8)[..., None, None]
        return np.where(small, eye, matrix)

    @staticmethod
    def _matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
        return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)

    @classmethod
    def _rotation_6d_to_matrix(cls, value: np.ndarray) -> np.ndarray:
        a1 = value[..., 0:3]
        a2 = value[..., 3:6]
        b1 = cls._normalize(a1)
        b2 = cls._normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)

    @staticmethod
    def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
        trace = np.trace(matrix, axis1=-2, axis2=-1)
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        axis = np.stack(
            [
                matrix[..., 2, 1] - matrix[..., 1, 2],
                matrix[..., 0, 2] - matrix[..., 2, 0],
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ],
            axis=-1,
        )
        denom = 2.0 * np.sin(angle)[..., None]
        axis = axis / np.maximum(np.abs(denom), 1e-8) * np.sign(denom)
        result = axis * angle[..., None]
        return np.where((angle < 1e-8)[..., None], np.zeros_like(result), result)
