from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from .base_lowdim_policy import BaseLowdimPolicy
from ..model.common.normalizer import LinearNormalizer
from ..model.diffusion.cond_unet1d import ConditionalUNet1D
from ..model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    """Lowdim Diffusion Policy UNet without latent state updater."""

    def __init__(
        self,
        model: ConditionalUNet1D | None = None,
        noise_scheduler: DDPMScheduler | None = None,
        horizon: int = 16,
        obs_dim: int = 0,
        action_dim: int = 0,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        num_inference_steps: int | None = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = True,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if noise_scheduler is None:
            raise ValueError("noise_scheduler must be provided.")
        if obs_as_local_cond and obs_as_global_cond:
            raise ValueError("Only one of obs_as_local_cond and obs_as_global_cond can be true.")
        if pred_action_steps_only and not obs_as_global_cond:
            raise ValueError("pred_action_steps_only requires obs_as_global_cond=True.")

        input_dim = (
            action_dim
            if (obs_as_local_cond or obs_as_global_cond)
            else action_dim + obs_dim
        )
        local_cond_dim = obs_dim if obs_as_local_cond else None
        global_cond_dim = obs_dim * n_obs_steps if obs_as_global_cond else None
        if model is None:
            model = ConditionalUNet1D(
                input_dim=input_dim,
                local_cond_dim=local_cond_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_local_cond = bool(obs_as_local_cond)
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.pred_action_steps_only = bool(pred_action_steps_only)
        self.oa_step_convention = bool(oa_step_convention)
        for unused_key in (
            "state_dim",
            "num_state_patches",
            "state_updater_depth",
            "state_updater_heads",
            "state_updater_dropout",
        ):
            kwargs.pop(unused_key, None)
        self.kwargs = kwargs
        self.num_inference_steps = (
            int(noise_scheduler.config.num_train_timesteps)
            if num_inference_steps is None
            else int(num_inference_steps)
        )

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor] | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(obs_dict, torch.Tensor):
            obs = obs_dict
        else:
            assert "obs" in obs_dict
            assert "past_action" not in obs_dict
            obs = obs_dict["obs"]

        nobs = self.normalizer["obs"].normalize(obs)
        batch_size, _, obs_dim = nobs.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs_dim}.")

        local_cond = None
        global_cond = None
        horizon = self.horizon
        action_dim = self.action_dim
        n_obs_steps = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        if self.obs_as_local_cond:
            local_cond = torch.zeros(
                size=(batch_size, horizon, obs_dim),
                device=device,
                dtype=dtype,
            )
            local_cond[:, :n_obs_steps] = nobs[:, :n_obs_steps]
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :n_obs_steps].reshape(batch_size, -1)
            shape = (batch_size, horizon, action_dim)
            if self.pred_action_steps_only:
                shape = (batch_size, self.n_action_steps, action_dim)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim + obs_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :n_obs_steps, action_dim:] = nobs[:, :n_obs_steps]
            cond_mask[:, :n_obs_steps, action_dim:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = n_obs_steps - 1 if self.oa_step_convention else n_obs_steps
            action = action_pred[:, start : start + self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            obs_pred = self.normalizer["obs"].unnormalize(nsample[..., action_dim:])
            result["obs_pred"] = obs_pred
            result["action_obs_pred"] = obs_pred[:, start : start + self.n_action_steps]
        return result

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert "valid_mask" not in batch
        nobs = self.normalizer["obs"].normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        local_cond = None
        global_cond = None
        trajectory = nactions

        if self.obs_as_local_cond:
            local_cond = nobs.clone()
            local_cond[:, self.n_obs_steps :] = 0
        elif self.obs_as_global_cond:
            global_cond = nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)
            if self.pred_action_steps_only:
                start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
                trajectory = nactions[:, start : start + self.n_action_steps]
        else:
            trajectory = torch.cat([nactions, nobs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn_like(trajectory)
        batch_size = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory,
            noise,
            timesteps,
        )
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).type(loss.dtype)
        return reduce(loss, "b ... -> b (...)", "mean").mean()


__all__ = ["DiffusionUnetLowdimPolicy"]
