from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from .base_lowdim_policy import BaseLowdimPolicy
from .shortcut_unet_lowdim_policy import _cfg_get
from ..model.common.normalizer import LinearNormalizer
from ..model.diffusion.cond_unet1d import ConditionalUNet1D
from ..model.diffusion.mask_generator import LowdimMaskGenerator
from ..model.one_step_flow.self_consistency_targets import (
    make_one_step_flow_training_batch,
    time_contract_rho,
)


def _broadcast_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], *([1] * (target.ndim - 1)))


class OneStepFlowUnetLowdimPolicy(BaseLowdimPolicy):
    """Lowdim one-step flow policy with self-consistency and self-guidance."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int | None = None,
        denoise_timesteps: int = 128,
        bootstrap_every: int = 4,
        obs_as_global_cond: bool = True,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = True,
        diffusion_step_embed_dim: int = 256,
        shortcut_step_embed_dim: int | None = None,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        clip_sample: float | None = 4.0,
        null_cond_init_std: float = 0.0,
        use_warm_start: bool = False,
        warm_start_t: float = 0.15,
        inference_cfg_scale: float = 1.0,
        extra_global_cond_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        if not obs_as_global_cond:
            raise NotImplementedError(
                "OneStepFlowUnetLowdimPolicy currently assumes obs_as_global_cond=True."
            )

        obs_global_cond_dim = int(obs_dim) * int(n_obs_steps)
        extra_global_cond_dim = int(extra_global_cond_dim)
        if extra_global_cond_dim < 0:
            raise ValueError(
                f"extra_global_cond_dim must be non-negative, got {extra_global_cond_dim}."
            )
        global_cond_dim = obs_global_cond_dim + extra_global_cond_dim
        self.model = ConditionalUNet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            shortcut_step_embed_dim=shortcut_step_embed_dim,
            condition_on_shortcut_step=True,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
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
        self.obs_global_cond_dim = obs_global_cond_dim
        self.extra_global_cond_dim = extra_global_cond_dim
        self.global_cond_dim = global_cond_dim
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.pred_action_steps_only = bool(pred_action_steps_only)
        self.oa_step_convention = bool(oa_step_convention)
        self.kwargs = kwargs
        self.num_inference_steps = 1 if num_inference_steps is None else int(num_inference_steps)
        self.denoise_timesteps = int(denoise_timesteps)
        self.bootstrap_every = int(bootstrap_every)
        self.clip_sample = clip_sample
        self.use_warm_start = bool(use_warm_start)
        self.warm_start_t = min(max(float(warm_start_t), 0.0), 1.0)
        self.inference_cfg_scale = float(inference_cfg_scale)
        self._warm_start_action_pred: torch.Tensor | None = None

        self.null_global_cond = nn.Parameter(torch.zeros(1, global_cond_dim))
        if null_cond_init_std > 0.0:
            nn.init.normal_(self.null_global_cond, std=float(null_cond_init_std))

    def reset(self) -> None:
        self._warm_start_action_pred = None

    def _global_conditioning(self, nobs: torch.Tensor) -> torch.Tensor:
        """Build the model-specific observation conditioning tensor."""
        if nobs.shape[1] < self.n_obs_steps:
            raise ValueError(
                f"Expected at least {self.n_obs_steps} observation steps, "
                f"got {nobs.shape[1]}."
            )
        return nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)

    def apply_null_conditioning(
        self,
        global_cond: torch.Tensor,
        dropout_prob: float = 0.0,
        force_null: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.full(
            (global_cond.shape[0],),
            bool(force_null),
            device=global_cond.device,
            dtype=torch.bool,
        )

        if not force_null:
            if dropout_prob <= 0.0 or not self.training or not torch.is_grad_enabled():
                return global_cond, mask
            mask = torch.rand(global_cond.shape[0], device=global_cond.device) < float(
                dropout_prob
            )

        null_cond = self.null_global_cond.to(
            device=global_cond.device,
            dtype=global_cond.dtype,
        )
        if null_cond.shape[1:] != global_cond.shape[1:]:
            raise ValueError(
                "Null conditioning shape does not match global conditioning: "
                f"{tuple(null_cond.shape[1:])} != {tuple(global_cond.shape[1:])}."
            )
        null_cond = null_cond.expand(global_cond.shape[0], *global_cond.shape[1:])
        mask_shape = (global_cond.shape[0],) + (1,) * (global_cond.ndim - 1)
        return torch.where(mask.view(mask_shape), null_cond, global_cond), mask

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        generator=None,
        warm_start: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        start_t = 0.0
        if warm_start is not None:
            if warm_start.shape != condition_data.shape:
                raise ValueError(
                    "warm_start must match condition_data shape, got "
                    f"{tuple(warm_start.shape)} and {tuple(condition_data.shape)}."
                )
            start_t = self.warm_start_t
            trajectory = (1.0 - start_t) * noise + start_t * warm_start
        else:
            trajectory = noise

        num_steps = max(int(self.num_inference_steps), 1)
        dt = (1.0 - start_t) / float(num_steps)
        for step_idx in range(num_steps):
            t_value = start_t + step_idx * dt
            t = torch.full(
                (trajectory.shape[0],),
                t_value,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            self_consistency_step = torch.full_like(t, dt)
            trajectory = torch.where(condition_mask, condition_data, trajectory)
            pred = self.model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
                shortcut_step=self_consistency_step,
            )
            cfg_scale = float(getattr(self, "inference_cfg_scale", 1.0))
            if cfg_scale != 1.0 and global_cond is not None:
                null_global_cond, _ = self.apply_null_conditioning(
                    global_cond,
                    force_null=True,
                )
                pred_null = self.model(
                    trajectory,
                    t,
                    local_cond=local_cond,
                    global_cond=null_global_cond,
                    shortcut_step=self_consistency_step,
                )
                pred = pred_null + cfg_scale * (pred - pred_null)
            trajectory = (
                trajectory + _broadcast_like(self_consistency_step, trajectory) * pred
            )
            if self.clip_sample is not None:
                trajectory = torch.clamp(
                    trajectory,
                    -float(self.clip_sample),
                    float(self.clip_sample),
                )

        return torch.where(condition_mask, condition_data, trajectory)

    def _make_warm_start(
        self,
        *,
        batch_size: int,
        shape: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.use_warm_start or self._warm_start_action_pred is None:
            return None
        previous = self._warm_start_action_pred
        if previous.shape != shape:
            return None
        previous = previous.to(device=device, dtype=dtype)
        if previous.shape[0] != batch_size:
            return None

        shift = max(int(self.n_action_steps), 1)
        horizon = previous.shape[1]
        terminal = previous[:, -1:, :]
        if shift >= horizon:
            return terminal.expand(batch_size, horizon, self.action_dim).clone()
        pad = terminal.expand(batch_size, shift, self.action_dim)
        return torch.cat([previous[:, shift:, :], pad], dim=1).detach()

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

        global_cond = self._global_conditioning(nobs)
        shape = (batch_size, self.horizon, self.action_dim)
        if self.pred_action_steps_only:
            shape = (batch_size, self.n_action_steps, self.action_dim)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        warm_start = self._make_warm_start(
            batch_size=batch_size,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            warm_start=warm_start,
            **self.kwargs,
        )
        self._warm_start_action_pred = nsample.detach()
        action_pred = self.normalizer["action"].unnormalize(nsample)
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            action = action_pred[:, start : start + self.n_action_steps]
        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _compute_self_guidance_loss(
        self,
        trajectory: torch.Tensor,   # clean action sequences
        *,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: torch.Tensor,
        teacher_policy: "OneStepFlowUnetLowdimPolicy",
        loss_cfg: Any,
        sample_offset: int = 0,
        forbidden_end: int = 0,
        guidance_size: int | None = None,
        return_metadata: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute self-guidance loss by generating one-step predictions and using the teacher ema model to compute a guidance signal.
        
        Args:
            trajectory: Clean action sequences of shape (batch_size, horizon, action_dim).
            condition_data: Condition data for the trajectory, same shape as trajectory.
            condition_mask: Boolean mask indicating which elements of condition_data are valid conditions, same shape as trajectory.
            global_cond: Global conditioning data (e.g. observations) of shape (batch_size, global_cond_dim).
            teacher_policy: Teacher policy with an EMA model used to compute the guidance signal.
            loss_cfg: Configuration for computing the self-guidance loss, should contain "guidance_every" and optionally "clip_sample".
            sample_offset: start index self guidance
            forbidden_end: End index of the forbidden prefix used for self-consistency.
            return_metadata: Whether to return self-guidance slice metadata.
        """
        batch_size = trajectory.shape[0]
        
        if guidance_size is not None:
            requested_guidance_size = int(guidance_size)
        else:
            guidance_ratio = _cfg_get(loss_cfg, "self_guidance_sample_ratio", None)
            if guidance_ratio is not None:
                guidance_ratio = float(guidance_ratio)
                if guidance_ratio <= 0.0 or guidance_ratio >= 1.0:
                    raise ValueError(
                        "self_guidance_sample_ratio must be in (0, 1), "
                        f"got {guidance_ratio}."
                    )
                requested_guidance_size = int(round(float(batch_size) * guidance_ratio))
            else:
                guidance_every = int(_cfg_get(loss_cfg, "guidance_every", 10))
                if guidance_every <= 0:
                    raise ValueError(
                        f"guidance_every must be positive, got {guidance_every}."
                    )
                requested_guidance_size = batch_size // guidance_every
        
        start_idx = min(max(int(sample_offset), 0), batch_size)
        
        guidance_size = min(requested_guidance_size, batch_size - start_idx)
        if guidance_size <= 0:
            raise ValueError(
                f"Invalid guidance_size={guidance_size} computed from "
                f"batch_size={batch_size}, requested_guidance_size="
                f"{requested_guidance_size}, and sample_offset={sample_offset}."
            )
        
        end_idx = start_idx + guidance_size
        
        overlap_count = max(0, min(end_idx, int(forbidden_end)) - start_idx)

        # clean action sequences, using only the indexes for self-guidance targets
        x1 = trajectory[start_idx:end_idx]
        
        cond_data = condition_data[start_idx:end_idx]
        cond_mask = condition_mask[start_idx:end_idx]
        
        # global conditioning (e.g. obs) for the guided samples
        cond_global = global_cond[start_idx:end_idx]

        # sample first noise
        noise0 = torch.randn_like(x1)
        
        # sample first timestep
        t = torch.rand(guidance_size, device=x1.device, dtype=x1.dtype)
        
        # interpolate noisy action sequence at timestep t
        x_t = (1.0 - _broadcast_like(t, x1)) * noise0 + _broadcast_like(t, x1) * x1
        
        x_t = torch.where(cond_mask, cond_data, x_t)
        
        self_consistency_step = 1.0 - t

        # predict velocity with the current model
        pred = self.model(
            x_t,
            t,
            local_cond=None,
            global_cond=cond_global,
            shortcut_step=self_consistency_step,
        )
        
        # generate one-step action sequence
        x_hat = x_t + _broadcast_like(self_consistency_step, x_t) * pred
        
        # Self-guidance can disable clipping independently to match OFP Eq. 12-13.
        clip_sample = _cfg_get(loss_cfg, "clip_sample", self.clip_sample)
        if loss_cfg is not None:
            marker = object()
            if isinstance(loss_cfg, dict):
                self_guidance_clip_sample = loss_cfg.get(
                    "self_guidance_clip_sample",
                    marker,
                )
            else:
                try:
                    self_guidance_clip_sample = loss_cfg.get(
                        "self_guidance_clip_sample",
                        marker,
                    )
                except AttributeError:
                    self_guidance_clip_sample = getattr(
                        loss_cfg,
                        "self_guidance_clip_sample",
                        marker,
                    )
            if self_guidance_clip_sample is not marker:
                clip_sample = self_guidance_clip_sample
        if clip_sample is not None:
            x_hat = torch.clamp(x_hat, -float(clip_sample), float(clip_sample))
        x_hat = torch.where(cond_mask, cond_data, x_hat)

        # sample 2nd noise for teacher prediction
        noise1 = torch.randn_like(x1)
        
        # sample 2nd timestep for teacher prediction
        t_prime = torch.rand(guidance_size, device=x1.device, dtype=x1.dtype)
        
        # interpolate noisy action seq from predicted action sequence at t_prime
        z_prime = (
            (1.0 - _broadcast_like(t_prime, x_hat)) * noise1 + _broadcast_like(t_prime, x_hat) * x_hat.detach()
        )
        z_prime = torch.where(cond_mask, cond_data, z_prime)
        
        # diagonal flow estimate u(z', t', t' | o)
        teacher_step = torch.zeros_like(t_prime)

        null_global, _ = teacher_policy.apply_null_conditioning(
            cond_global,
            force_null=True,
        )
        with torch.no_grad():
            # conditional velocity with ema 
            teacher_cond = teacher_policy.model(
                z_prime,
                t_prime,
                local_cond=None,
                global_cond=cond_global,
                shortcut_step=teacher_step,
            )
            # unconditional velocity with ema (null global conditioning)
            teacher_null = teacher_policy.model(
                z_prime,
                t_prime,
                local_cond=None,
                global_cond=null_global,
                shortcut_step=teacher_step,
            )
            guidance_delta = teacher_null - teacher_cond
            guided_target = pred.detach() - guidance_delta

        loss = F.mse_loss(pred, guided_target, reduction="none")
        loss = loss * (~cond_mask).type(loss.dtype)
        loss = reduce(loss, "b ... -> b", "mean").mean()
        ratio = trajectory.new_tensor(float(guidance_size) / float(batch_size))
        pred_norm = reduce(pred.detach().pow(2), "b ... -> b", "mean").sqrt().mean()
        guidance_delta_norm = (
            reduce(guidance_delta.detach().pow(2), "b ... -> b", "mean")
            .sqrt()
            .mean()
        )
        
        if return_metadata:
            start = trajectory.new_tensor(float(start_idx))
            end = trajectory.new_tensor(float(end_idx))
            overlap_ratio = trajectory.new_tensor(float(overlap_count) / float(guidance_size))
            return (
                loss,
                ratio,
                start,
                end,
                overlap_ratio,
                pred_norm,
                guidance_delta_norm,
            )
        
        return loss, ratio

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_policy: Optional["OneStepFlowUnetLowdimPolicy"] = None,
        loss_cfg: Any = None,
        training_progress: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        
        assert "valid_mask" not in batch
        
        # normalize obs and action
        nobs = self.normalizer["obs"].normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        batch_size, _, obs_dim = nobs.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs_dim}.")

        # normalized action seqeunces
        trajectory = nactions
        
        if self.pred_action_steps_only:
            start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            trajectory = nactions[:, start : start + self.n_action_steps]
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)
        condition_data = trajectory

        # get observations as global conditioning
        global_cond = self._global_conditioning(nobs)
        
        condition_dropout = float(_cfg_get(loss_cfg, "condition_dropout", 0.0))
        training_global_cond, dropout_mask = self.apply_null_conditioning(
            global_cond,
            dropout_prob=condition_dropout,
        )

        denoise_timesteps = int(
            _cfg_get(loss_cfg, "denoise_timesteps", self.denoise_timesteps)
        )
        self_consistency_every = int(
            _cfg_get(loss_cfg, "bootstrap_every", self.bootstrap_every)
        )
        discrete_time = bool(_cfg_get(loss_cfg, "discrete_time", True))
        clip_sample = _cfg_get(loss_cfg, "clip_sample", self.clip_sample)
        lambda_flow = float(_cfg_get(loss_cfg, "lambda_flow", 1.0))
        lambda_self_consistency = float(
            _cfg_get(loss_cfg, "lambda_self_consistency", 1.0)
        )
        lambda_self_guidance = float(_cfg_get(loss_cfg, "lambda_self_guidance", 0.0))
        scale_self_guidance_by_ratio = bool(
            _cfg_get(loss_cfg, "scale_self_guidance_by_ratio", True)
        )
        use_ema_teacher = bool(_cfg_get(loss_cfg, "bootstrap_ema", True))
        time_contract_start = float(_cfg_get(loss_cfg, "time_contract_start", 1.0))
        time_contract_end = float(_cfg_get(loss_cfg, "time_contract_end", 0.0))
        time_contract_schedule = str(_cfg_get(loss_cfg, "time_contract_schedule", "linear"))
        time_contract_power = float(_cfg_get(loss_cfg, "time_contract_power", 2.0))
        min_middle_fraction = float(_cfg_get(loss_cfg, "min_middle_fraction", 0.0))
        t_sampler = str(_cfg_get(loss_cfg, "t_sampler", "beta"))
        t_beta_alpha = float(_cfg_get(loss_cfg, "t_beta_alpha", 1.0))
        t_beta_beta = float(_cfg_get(loss_cfg, "t_beta_beta", 1.5))
        interval_sampler = str(_cfg_get(loss_cfg, "interval_sampler", "logit_normal"))
        interval_mu = float(_cfg_get(loss_cfg, "interval_mu", -0.2))
        interval_sigma = float(_cfg_get(loss_cfg, "interval_sigma", 1.0))
        min_self_consistency_step = _cfg_get(
            loss_cfg,
            "min_self_consistency_step",
            None,
        )
        if min_self_consistency_step is not None:
            min_self_consistency_step = float(min_self_consistency_step)

        target_policy = teacher_policy if (use_ema_teacher and teacher_policy is not None) else self

        # create flow matching and OFP self-consistency targets for each sample in the batch
        targets = make_one_step_flow_training_batch(
            trajectory,
            condition_data=condition_data,
            condition_mask=condition_mask,
            global_cond=training_global_cond,
            teacher_model_fn=target_policy.model,
            denoise_timesteps=denoise_timesteps,
            self_consistency_every=self_consistency_every,
            discrete_time=discrete_time,
            clip_sample=clip_sample,
            training_progress=float(training_progress),
            time_contract_start=time_contract_start,
            time_contract_end=time_contract_end,
            time_contract_schedule=time_contract_schedule,
            time_contract_power=time_contract_power,
            # min_middle_fraction=min_middle_fraction,
            t_sampler=t_sampler,
            t_beta_alpha=t_beta_alpha,
            t_beta_beta=t_beta_beta,
            interval_sampler=interval_sampler,
            interval_mu=interval_mu,
            interval_sigma=interval_sigma,
            min_self_consistency_step=min_self_consistency_step,
        )

        # predict velocity with the current model for both flow matching and self-consistency targets
        pred = self.model(
            targets["x_t"],
            targets["t"],
            local_cond=None,
            global_cond=targets["global_cond"],
            shortcut_step=targets["self_consistency_step"],
        )
        # compute MSE loss
        loss = F.mse_loss(pred, targets["target"], reduction="none")
        loss = loss * (~targets["condition_mask"]).type(loss.dtype)
        per_example_loss = reduce(loss, "b ... -> b", "mean")

        is_self_consistency = targets["is_self_consistency"]
        is_flow = ~is_self_consistency
        if not is_flow.any():
            raise ValueError("No flow matching samples in the batch, cannot compute flow loss.")
        if not is_self_consistency.any():
            raise ValueError(
                "No self-consistency samples in the batch, cannot compute "
                "self-consistency loss."
            )
        loss_flow = per_example_loss[is_flow].mean()
        loss_self_consistency = per_example_loss[is_self_consistency].mean()
            
        # add lambda weighting to flow matching and self-consistency losses and combine them
        example_weights = torch.where(
            is_self_consistency,
            per_example_loss.new_tensor(lambda_self_consistency),
            per_example_loss.new_tensor(lambda_flow),
        )
        loss_main = (per_example_loss * example_weights).mean()

        if lambda_self_guidance > 0.0:
            # Start SGR after the self-consistency prefix so guidance is drawn
            # from the flow-matching portion of the batch.
            self_guidance_start_idx = trajectory.shape[0] // self_consistency_every
            (
                loss_self_guidance,
                self_guidance_ratio,
                self_guidance_start_idx_tensor,
                self_guidance_end_idx_tensor,
                self_guidance_overlap_ratio,
                self_guidance_pred_norm,
                self_guidance_delta_norm,
            ) = self._compute_self_guidance_loss(
                trajectory,
                condition_data=condition_data,
                condition_mask=condition_mask,
                global_cond=global_cond,
                teacher_policy=target_policy,
                loss_cfg=loss_cfg,
                sample_offset=self_guidance_start_idx,
                forbidden_end=self_guidance_start_idx,
                return_metadata=True,
            )
        else:
            loss_self_guidance = loss_main.new_tensor(0.0)
            self_guidance_ratio = loss_main.new_tensor(0.0)
            self_guidance_start_idx_tensor = loss_main.new_tensor(-1.0)
            self_guidance_end_idx_tensor = loss_main.new_tensor(-1.0)
            self_guidance_overlap_ratio = loss_main.new_tensor(0.0)
            self_guidance_pred_norm = loss_main.new_tensor(0.0)
            self_guidance_delta_norm = loss_main.new_tensor(0.0)

        self_guidance_weight = loss_main.new_tensor(lambda_self_guidance)
        if scale_self_guidance_by_ratio:
            self_guidance_weight = self_guidance_weight * self_guidance_ratio
        loss_total = loss_main + self_guidance_weight * loss_self_guidance
        
        # p(s)
        rho = time_contract_rho(
            training_progress=float(training_progress),
            start=time_contract_start,
            end=time_contract_end,
            schedule=time_contract_schedule,
            power=time_contract_power,
        )
        return {
            "loss_total": loss_total,
            "loss_main": loss_main.detach(),
            "loss_flow": loss_flow.detach(),
            "loss_self_consistency": loss_self_consistency.detach(),
            "loss_self_guidance": loss_self_guidance.detach(),
            "self_consistency_ratio": targets["self_consistency_ratio"].detach(),
            "self_guidance_ratio": self_guidance_ratio.detach(),
            "self_guidance_weight": self_guidance_weight.detach(),
            # "self_consistency_start_idx": targets["self_consistency_start_idx"].detach(),
            # "self_consistency_end_idx": targets["self_consistency_end_idx"].detach(),
            # "flow_start_idx": targets["flow_start_idx"].detach(),
            # "flow_end_idx": targets["flow_end_idx"].detach(),
            # "self_guidance_start_idx": self_guidance_start_idx_tensor.detach(),
            # "self_guidance_end_idx": self_guidance_end_idx_tensor.detach(),
            # "self_guidance_overlaps_self_consistency": self_guidance_overlap_ratio.detach(),
            "self_guidance_pred_norm": self_guidance_pred_norm.detach(),
            "self_guidance_delta_norm": self_guidance_delta_norm.detach(),
            "condition_dropout_ratio": dropout_mask.to(dtype=loss_total.dtype).mean().detach(),
            "time_contract_rho": loss_total.new_tensor(rho).detach(),
        }


__all__ = ["OneStepFlowUnetLowdimPolicy"]
