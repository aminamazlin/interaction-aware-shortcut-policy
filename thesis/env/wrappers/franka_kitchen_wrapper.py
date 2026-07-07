"""
Thesis-facing wrapper adapters for the Franka Kitchen benchmark.

The actual Gym/MuJoCo observation, action, reset, render, and step behavior is
provided by the local KitchenLowdimWrapper.  This module gives thesis
configs and code a local import path without copying that implementation.
"""
from __future__ import annotations

from thesis.env.kitchen.kitchen_lowdim_wrapper import (
    KitchenLowdimWrapper as _KitchenLowdimWrapper,
)


class FrankaKitchenLowdimWrapper(_KitchenLowdimWrapper):
    """Adapter around the local Kitchen low-dimensional wrapper."""
