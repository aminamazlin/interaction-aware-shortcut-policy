from __future__ import annotations

import pathlib


class BaseLowdimRunner:
    """Base class for low-dimensional evaluation runners."""

    def __init__(self, output_dir: str):
        self.output_dir = pathlib.Path(output_dir)
