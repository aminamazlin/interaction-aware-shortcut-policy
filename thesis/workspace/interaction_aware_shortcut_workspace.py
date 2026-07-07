from __future__ import annotations

from ..policy.interaction_aware_shortcut_policy import InteractionAwareShortcutPolicy
from .one_step_flow_unet_lowdim_workspace import TrainOneStepFlowUnetLowdimWorkspace


class TrainInteractionAwareShortcutWorkspace(TrainOneStepFlowUnetLowdimWorkspace):
    """Lowdim workspace isolated to InteractionAwareShortcutPolicy."""

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        if not isinstance(self.model, InteractionAwareShortcutPolicy):
            raise TypeError(
                "TrainInteractionAwareShortcutWorkspace requires "
                "InteractionAwareShortcutPolicy, got "
                f"{self.model.__class__.__module__}.{self.model.__class__.__name__}."
            )


__all__ = ["TrainInteractionAwareShortcutWorkspace"]
