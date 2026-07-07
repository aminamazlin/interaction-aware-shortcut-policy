from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from .one_step_flow_unet_lowdim_policy import (
    OneStepFlowUnetLowdimPolicy,
)
from .shortcut_unet_lowdim_policy import _cfg_get
from ..model.one_step_flow.self_consistency_targets import (
    make_one_step_flow_training_batch,
    time_contract_rho,
)


INTERACTION_QUERY_NAMES = (
    "stage",
    "progress",
    "object",
    "contact",
    "place",
    "motion",
)


def interaction_aux_schedule_scale(
    *,
    enabled: bool,
    schedule: str,
    training_progress: float,
    start_scale: float,
    end_scale: float,
    decay_start: float,
    decay_end: float,
    power: float = 1.0,
) -> float:
    if isinstance(enabled, str):
        enabled_value = enabled.strip().lower()
        enabled = enabled_value in {"1", "true", "yes", "on", "enable", "enabled"}
    else:
        enabled = bool(enabled)
    if not enabled:
        return 1.0

    schedule = str(schedule)
    progress = min(max(float(training_progress), 0.0), 1.0)
    start_scale = float(start_scale)
    end_scale = float(end_scale)
    decay_start = min(max(float(decay_start), 0.0), 1.0)
    decay_end = min(max(float(decay_end), 0.0), 1.0)
    if decay_end <= decay_start:
        alpha = 1.0 if progress >= decay_end else 0.0
    else:
        alpha = (progress - decay_start) / (decay_end - decay_start)
        alpha = min(max(alpha, 0.0), 1.0)

    if schedule == "constant":
        return start_scale
    if schedule == "linear_decay":
        shaped = alpha
    elif schedule == "cosine_decay":
        shaped = 0.5 - 0.5 * math.cos(alpha * math.pi)
    elif schedule == "power_decay":
        shaped = alpha ** max(float(power), 0.0)
    elif schedule == "step_decay":
        shaped = 1.0 if alpha >= 1.0 else 0.0
    else:
        raise ValueError(
            "interaction_aux_schedule must be one of 'constant', 'linear_decay', "
            f"'cosine_decay', 'power_decay', or 'step_decay', got {schedule!r}."
        )
    return start_scale + (end_scale - start_scale) * shaped


class InteractionStateQueryEncoder(nn.Module):
    """Lowdim interaction query module."""

    def __init__(
        self,
        *,
        obs_dim: int,
        n_obs_steps: int,
        num_stages: int,
        num_objects: int,
        query_dim: int = 128,
        context_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_obs_steps = int(n_obs_steps)
        self.num_stages = int(num_stages)
        self.num_objects = int(num_objects)
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)
        self.num_queries = len(INTERACTION_QUERY_NAMES)

        if self.query_dim % int(num_heads) != 0:
            raise ValueError(
                f"query_dim={self.query_dim} must be divisible by num_heads={num_heads}."
            )

        self.obs_token = nn.Linear(self.obs_dim, self.query_dim)
        self.obs_pos = nn.Parameter(torch.zeros(1, self.n_obs_steps, self.query_dim))
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_queries, self.query_dim)
        )
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.query_dim,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.query_norms = nn.ModuleList(
            [nn.LayerNorm(self.query_dim) for _ in range(int(num_layers))]
        )
        self.obs_norms = nn.ModuleList(
            [nn.LayerNorm(self.query_dim) for _ in range(int(num_layers))]
        )
        self.ffn_norms = nn.ModuleList(
            [nn.LayerNorm(self.query_dim) for _ in range(int(num_layers))]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.query_dim, self.query_dim * 4),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.query_dim * 4, self.query_dim),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.query_dim)
        self.context_projector = nn.Sequential(
            nn.Linear(self.num_queries * self.query_dim, self.context_dim),
            nn.SiLU(),
            nn.Linear(self.context_dim, self.context_dim),
        )

        def head(out_dim: int, final: nn.Module | None = None) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.LayerNorm(self.query_dim),
                nn.Linear(self.query_dim, self.query_dim),
                nn.SiLU(),
                nn.Linear(self.query_dim, int(out_dim)),
            ]
            if final is not None:
                layers.append(final)
            return nn.Sequential(*layers)

        self.stage_head = head(self.num_stages)
        self.progress_head = head(1, nn.Sigmoid())
        self.object_head = head(self.num_objects)
        self.contact_head = head(3)
        self.place_head = head(3)
        self.motion_head = head(3)

        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.obs_pos, std=0.02)

    def forward(self, nobs: torch.Tensor) -> dict[str, torch.Tensor]:
        if nobs.shape[1] < self.n_obs_steps:
            raise ValueError(
                f"Expected at least {self.n_obs_steps} obs steps, got {nobs.shape[1]}."
            )
        obs_tokens = self.obs_token(nobs[:, : self.n_obs_steps]) + self.obs_pos
        queries = self.query_tokens.expand(nobs.shape[0], -1, -1)
        for attn, query_norm, obs_norm, ffn_norm, ffn in zip(
            self.cross_attn,
            self.query_norms,
            self.obs_norms,
            self.ffn_norms,
            self.ffns,
        ):
            attended, _ = attn(
                query_norm(queries),
                obs_norm(obs_tokens),
                obs_norm(obs_tokens),
                need_weights=False,
            )
            queries = queries + attended
            queries = queries + ffn(ffn_norm(queries))

        queries = self.output_norm(queries)
        query_by_name = {
            name: queries[:, idx]
            for idx, name in enumerate(INTERACTION_QUERY_NAMES)
        }
        context = self.context_projector(queries.reshape(queries.shape[0], -1))
        out = {
            "query_tokens": queries,
            "context": context,
            "stage_logits": self.stage_head(query_by_name["stage"]),
            "progress": self.progress_head(query_by_name["progress"]).squeeze(-1),
            "object_logits": self.object_head(query_by_name["object"]),
            "contact_delta": self.contact_head(query_by_name["contact"]),
            "place_delta": self.place_head(query_by_name["place"]),
            "motion_delta": self.motion_head(query_by_name["motion"]),
        }
        return out


class InteractionAwareShortcutPolicy(OneStepFlowUnetLowdimPolicy):
    """Interaction-aware lowdim OFP with progress and interaction query context."""

    def __init__(
        self,
        *args,
        num_interaction_stages: int,
        num_interaction_objects: int = 4,
        interaction_query_dim: int = 128,
        interaction_context_dim: int = 128,
        interaction_num_heads: int = 4,
        interaction_num_layers: int = 2,
        interaction_dropout: float = 0.0,
        interaction_uncertainty_init: float = 0.0,
        predict_action_progress: bool = True,
        action_dim: int,
        obs_dim: int,
        n_obs_steps: int,
        **kwargs,
    ) -> None:
        self.real_action_dim = int(action_dim)
        self.predict_action_progress = bool(predict_action_progress)
        self.joint_action_dim = self.real_action_dim + (
            1 if self.predict_action_progress else 0
        )
        self.num_interaction_stages = int(num_interaction_stages)
        self.num_interaction_objects = int(num_interaction_objects)
        self.interaction_context_dim = int(interaction_context_dim)
        kwargs["extra_global_cond_dim"] = (
            int(kwargs.get("extra_global_cond_dim", 0)) + self.interaction_context_dim
        )
        super().__init__(
            *args,
            obs_dim=obs_dim,
            action_dim=self.joint_action_dim,
            n_obs_steps=n_obs_steps,
            **kwargs,
        )
        self.real_action_dim = int(action_dim)
        self.joint_action_dim = self.action_dim
        self.interaction_query_encoder = InteractionStateQueryEncoder(
            obs_dim=int(obs_dim),
            n_obs_steps=int(n_obs_steps),
            num_stages=self.num_interaction_stages,
            num_objects=self.num_interaction_objects,
            query_dim=int(interaction_query_dim),
            context_dim=self.interaction_context_dim,
            num_heads=int(interaction_num_heads),
            num_layers=int(interaction_num_layers),
            dropout=float(interaction_dropout),
        )
        self.interaction_aux_log_vars = nn.Parameter(
            torch.full(
                (len(INTERACTION_QUERY_NAMES),),
                float(interaction_uncertainty_init),
            )
        )

    def _interaction_predictions(
        self,
        nobs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_global_cond = self._global_conditioning(nobs)
        interaction = self.interaction_query_encoder(nobs)
        global_cond = torch.cat([obs_global_cond, interaction["context"]], dim=1)
        return global_cond, interaction

    def _current_interaction_targets(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        required = (
            "interaction_stage",
            "interaction_progress",
            "interaction_object",
            "interaction_contact_delta",
            "interaction_place_delta",
            "interaction_motion_delta",
            "interaction_motion_mask",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "InteractionAwareShortcutPolicy requires interaction-state "
                f"batch keys, missing={missing}. Use MimicGenInteractionStateLowdimDataset."
            )
        idx = self.n_obs_steps - 1
        targets = {
            "stage": torch.clamp(
                batch["interaction_stage"][:, idx].to(dtype=torch.long),
                0,
                self.num_interaction_stages - 1,
            ),
            "progress": torch.clamp(
                batch["interaction_progress"][:, idx].to(dtype=torch.float32),
                0.0,
                1.0,
            ),
            "object": torch.clamp(
                batch["interaction_object"][:, idx].to(dtype=torch.long),
                0,
                self.num_interaction_objects - 1,
            ),
            "contact_delta": batch["interaction_contact_delta"][:, idx].to(dtype=torch.float32),
            "place_delta": batch["interaction_place_delta"][:, idx].to(dtype=torch.float32),
            "motion_delta": batch["interaction_motion_delta"][:, idx].to(dtype=torch.float32),
            "motion_mask": batch["interaction_motion_mask"][:, idx].to(dtype=torch.float32),
        }
        return targets

    def _make_joint_trajectory(
        self,
        nactions: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.predict_action_progress:
            return nactions
        if "interaction_progress" not in batch:
            raise KeyError(
                "predict_action_progress=True requires batch['interaction_progress']."
            )
        progress = batch["interaction_progress"].to(device=nactions.device, dtype=nactions.dtype)
        if self.pred_action_steps_only:
            start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            progress = progress[:, start : start + self.n_action_steps]
        progress = torch.clamp(progress, 0.0, 1.0)
        if progress.shape[:2] != nactions.shape[:2]:
            raise ValueError(
                f"Progress shape {tuple(progress.shape)} does not align with "
                f"actions shape {tuple(nactions.shape)}."
            )
        return torch.cat([nactions, progress.unsqueeze(-1)], dim=-1)

    def _split_joint_sample(
        self,
        sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        action = sample[..., : self.real_action_dim]
        if not self.predict_action_progress:
            return action, None
        progress = sample[..., self.real_action_dim : self.real_action_dim + 1]
        return action, progress

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

        global_cond, interaction = self._interaction_predictions(nobs)
        shape = (batch_size, self.horizon, self.joint_action_dim)
        if self.pred_action_steps_only:
            shape = (batch_size, self.n_action_steps, self.joint_action_dim)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        warm_start = self._make_warm_start(
            batch_size=batch_size,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

        nsample_joint = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            warm_start=warm_start,
            **self.kwargs,
        )
        self._warm_start_action_pred = nsample_joint.detach()
        nsample_action, progress_joint = self._split_joint_sample(nsample_joint)
        action_pred = self.normalizer["action"].unnormalize(nsample_action)
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            action = action_pred[:, start : start + self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
            "interaction_stage_logits": interaction["stage_logits"],
            "interaction_stage": torch.argmax(interaction["stage_logits"], dim=1),
            "interaction_progress": interaction["progress"],
            "interaction_object_logits": interaction["object_logits"],
            "interaction_object": torch.argmax(interaction["object_logits"], dim=1),
            "interaction_contact_delta": interaction["contact_delta"],
            "interaction_place_delta": interaction["place_delta"],
            "interaction_motion_delta": interaction["motion_delta"],
            "interaction_action_progress_pred": progress_joint,
        }
        return result

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_policy: Optional["InteractionAwareShortcutPolicy"] = None,
        loss_cfg: Any = None,
        training_progress: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        assert "valid_mask" not in batch

        nobs = self.normalizer["obs"].normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        batch_size, _, obs_dim = nobs.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs_dim}.")

        trajectory_actions = nactions
        if self.pred_action_steps_only:
            start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            trajectory_actions = nactions[:, start : start + self.n_action_steps]
        trajectory = self._make_joint_trajectory(trajectory_actions, batch)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)
        condition_data = trajectory

        global_cond, interaction = self._interaction_predictions(nobs)
        interaction_targets = self._current_interaction_targets(batch)

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
        lambda_progress_flow = float(
            _cfg_get(loss_cfg, "lambda_action_progress_flow", 0.1)
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
            t_sampler=t_sampler,
            t_beta_alpha=t_beta_alpha,
            t_beta_beta=t_beta_beta,
            interval_sampler=interval_sampler,
            interval_mu=interval_mu,
            interval_sigma=interval_sigma,
            min_self_consistency_step=min_self_consistency_step,
        )

        pred = self.model(
            targets["x_t"],
            targets["t"],
            local_cond=None,
            global_cond=targets["global_cond"],
            shortcut_step=targets["self_consistency_step"],
        )
        squared_error = (pred - targets["target"]).pow(2)
        action_error = squared_error[..., : self.real_action_dim]
        if self.predict_action_progress:
            progress_error = squared_error[..., self.real_action_dim :]
            weighted_error = torch.cat(
                [
                    action_error,
                    progress_error * float(lambda_progress_flow),
                ],
                dim=-1,
            )
            loss_progress_flow = (
                progress_error
                * (~targets["condition_mask"][..., self.real_action_dim :]).type(
                    progress_error.dtype
                )
            )
            loss_progress_flow = reduce(loss_progress_flow, "b ... -> b", "mean").mean()
        else:
            weighted_error = action_error
            loss_progress_flow = pred.new_tensor(0.0)

        loss = weighted_error * (~targets["condition_mask"]).type(weighted_error.dtype)
        per_example_loss = reduce(loss, "b ... -> b", "mean")

        is_self_consistency = targets["is_self_consistency"]
        is_flow = ~is_self_consistency
        if not is_flow.any():
            raise ValueError("No flow matching samples in the batch.")
        if not is_self_consistency.any():
            raise ValueError("No self-consistency samples in the batch.")
        loss_flow = per_example_loss[is_flow].mean()
        loss_self_consistency = per_example_loss[is_self_consistency].mean()
        example_weights = torch.where(
            is_self_consistency,
            per_example_loss.new_tensor(lambda_self_consistency),
            per_example_loss.new_tensor(lambda_flow),
        )
        loss_main = (per_example_loss * example_weights).mean()

        if lambda_self_guidance > 0.0:
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

        loss_interaction_stage = F.cross_entropy(
            interaction["stage_logits"],
            interaction_targets["stage"].to(device=interaction["stage_logits"].device),
        )
        loss_interaction_object = F.cross_entropy(
            interaction["object_logits"],
            interaction_targets["object"].to(device=interaction["object_logits"].device),
        )
        loss_interaction_progress = F.smooth_l1_loss(
            interaction["progress"],
            interaction_targets["progress"].to(device=interaction["progress"].device),
        )
        loss_interaction_contact = F.smooth_l1_loss(
            interaction["contact_delta"],
            interaction_targets["contact_delta"].to(device=interaction["contact_delta"].device),
        )
        loss_interaction_place = F.smooth_l1_loss(
            interaction["place_delta"],
            interaction_targets["place_delta"].to(device=interaction["place_delta"].device),
        )
        motion_target = interaction_targets["motion_delta"].to(device=interaction["motion_delta"].device)
        motion_mask = interaction_targets["motion_mask"].to(
            device=interaction["motion_delta"].device,
            dtype=interaction["motion_delta"].dtype,
        )
        motion_loss = F.smooth_l1_loss(
            interaction["motion_delta"],
            motion_target,
            reduction="none",
        ).mean(dim=-1)
        denom = torch.clamp(motion_mask.sum(), min=1.0)
        loss_interaction_motion = (motion_loss * motion_mask).sum() / denom

        interaction_aux_losses = torch.stack(
            [
                loss_interaction_stage,
                loss_interaction_progress,
                loss_interaction_object,
                loss_interaction_contact,
                loss_interaction_place,
                loss_interaction_motion,
            ]
        )
        interaction_aux_weighting = str(
            _cfg_get(loss_cfg, "interaction_aux_weighting", "per_term")
        ).strip().lower()
        interaction_aux_schedule_scale_value = interaction_aux_schedule_scale(
            enabled=_cfg_get(loss_cfg, "interaction_aux_schedule_enabled", False),
            schedule=str(_cfg_get(loss_cfg, "interaction_aux_schedule", "linear_decay")),
            training_progress=float(training_progress),
            start_scale=float(_cfg_get(loss_cfg, "interaction_aux_schedule_start_scale", 1.0)),
            end_scale=float(_cfg_get(loss_cfg, "interaction_aux_schedule_end_scale", 1.0)),
            decay_start=float(_cfg_get(loss_cfg, "interaction_aux_schedule_decay_start", 0.0)),
            decay_end=float(_cfg_get(loss_cfg, "interaction_aux_schedule_decay_end", 1.0)),
            power=float(_cfg_get(loss_cfg, "interaction_aux_schedule_power", 1.0)),
        )

        if interaction_aux_weighting == "per_term":
            lambda_interaction_stage = float(_cfg_get(loss_cfg, "lambda_interaction_stage", 0.1))
            lambda_interaction_progress = float(_cfg_get(loss_cfg, "lambda_interaction_progress", 0.1))
            lambda_interaction_object = float(_cfg_get(loss_cfg, "lambda_interaction_object", 0.1))
            lambda_interaction_contact = float(_cfg_get(loss_cfg, "lambda_interaction_contact", 1.0))
            lambda_interaction_place = float(_cfg_get(loss_cfg, "lambda_interaction_place", 1.0))
            lambda_interaction_motion = float(_cfg_get(loss_cfg, "lambda_interaction_motion", 1.0))

            loss_interaction_aux = (
                loss_main.new_tensor(lambda_interaction_stage) * loss_interaction_stage
                + loss_main.new_tensor(lambda_interaction_progress) * loss_interaction_progress
                + loss_main.new_tensor(lambda_interaction_object) * loss_interaction_object
                + loss_main.new_tensor(lambda_interaction_contact) * loss_interaction_contact
                + loss_main.new_tensor(lambda_interaction_place) * loss_interaction_place
                + loss_main.new_tensor(lambda_interaction_motion) * loss_interaction_motion
            )
            interaction_aux_weight = float(interaction_aux_schedule_scale_value)
            lambda_interaction_stage_eff = lambda_interaction_stage * interaction_aux_schedule_scale_value
            lambda_interaction_progress_eff = lambda_interaction_progress * interaction_aux_schedule_scale_value
            lambda_interaction_object_eff = lambda_interaction_object * interaction_aux_schedule_scale_value
            lambda_interaction_contact_eff = lambda_interaction_contact * interaction_aux_schedule_scale_value
            lambda_interaction_place_eff = lambda_interaction_place * interaction_aux_schedule_scale_value
            lambda_interaction_motion_eff = lambda_interaction_motion * interaction_aux_schedule_scale_value
        elif interaction_aux_weighting == "balanced_mean":
            lambda_interaction_aux = float(_cfg_get(loss_cfg, "lambda_interaction_aux", 1.0))
            interaction_aux_balance_min_scale = float(
                _cfg_get(loss_cfg, "interaction_aux_balance_min_scale", 1.0e-6)
            )
            if interaction_aux_balance_min_scale <= 0.0:
                raise ValueError(
                    "loss.interaction_aux_balance_min_scale must be positive for "
                    f"balanced_mean, got {interaction_aux_balance_min_scale}."
                )
            interaction_aux_terms = interaction_aux_losses / interaction_aux_losses.detach().clamp_min(
                interaction_aux_balance_min_scale
            )
            loss_interaction_aux = interaction_aux_terms.mean()
            interaction_aux_weight = lambda_interaction_aux * interaction_aux_schedule_scale_value
            lambda_interaction_stage_eff = 0.0
            lambda_interaction_progress_eff = 0.0
            lambda_interaction_object_eff = 0.0
            lambda_interaction_contact_eff = 0.0
            lambda_interaction_place_eff = 0.0
            lambda_interaction_motion_eff = 0.0
        elif interaction_aux_weighting == "learned_uncertainty":
            lambda_interaction_aux = float(_cfg_get(loss_cfg, "lambda_interaction_aux", 1.0))
            interaction_log_vars = self.interaction_aux_log_vars.to(
                device=interaction_aux_losses.device,
                dtype=interaction_aux_losses.dtype,
            )
            interaction_raw_weights = torch.exp(-interaction_log_vars)
            interaction_aux_terms = interaction_raw_weights * interaction_aux_losses + interaction_log_vars
            loss_interaction_aux = interaction_aux_terms.mean()
            interaction_aux_weight = lambda_interaction_aux * interaction_aux_schedule_scale_value
            interaction_effective_weights = (
                interaction_raw_weights
                * interaction_aux_losses.new_tensor(interaction_aux_weight / len(INTERACTION_QUERY_NAMES))
            )
            lambda_interaction_stage_eff = interaction_effective_weights[0]
            lambda_interaction_progress_eff = interaction_effective_weights[1]
            lambda_interaction_object_eff = interaction_effective_weights[2]
            lambda_interaction_contact_eff = interaction_effective_weights[3]
            lambda_interaction_place_eff = interaction_effective_weights[4]
            lambda_interaction_motion_eff = interaction_effective_weights[5]
        else:
            raise ValueError(
                "loss.interaction_aux_weighting must be one of 'per_term', "
                f"'balanced_mean', or 'learned_uncertainty', got {interaction_aux_weighting!r}."
            )

        self_guidance_weight = loss_main.new_tensor(lambda_self_guidance)
        if scale_self_guidance_by_ratio:
            self_guidance_weight = self_guidance_weight * self_guidance_ratio

        loss_total = (
            loss_main
            + self_guidance_weight * loss_self_guidance
            + loss_main.new_tensor(interaction_aux_weight) * loss_interaction_aux
        )

        with torch.no_grad():
            interaction_stage_acc = (
                torch.argmax(interaction["stage_logits"], dim=1)
                == interaction_targets["stage"].to(device=interaction["stage_logits"].device)
            ).to(dtype=loss_total.dtype).mean()
            interaction_object_acc = (
                torch.argmax(interaction["object_logits"], dim=1)
                == interaction_targets["object"].to(device=interaction["object_logits"].device)
            ).to(dtype=loss_total.dtype).mean()
            interaction_progress_mae = torch.mean(
                torch.abs(
                    interaction["progress"].detach()
                    - interaction_targets["progress"].to(device=interaction["progress"].device)
                )
            )
            interaction_context_norm = reduce(
                interaction["context"].detach().pow(2),
                "b ... -> b",
                "mean",
            ).sqrt().mean()

        rho = time_contract_rho(
            training_progress=float(training_progress),
            start=time_contract_start,
            end=time_contract_end,
            schedule=time_contract_schedule,
            power=time_contract_power,
        )

        def _metric_tensor(value: float | torch.Tensor) -> torch.Tensor:
            if torch.is_tensor(value):
                return value.detach()
            return loss_total.new_tensor(value).detach()

        return {
            # self distill loss
            "loss_total": loss_total,
            "loss_main": loss_main.detach(),
            "loss_flow": loss_flow.detach(),
            "loss_self_consistency": loss_self_consistency.detach(),
            "loss_self_guidance": loss_self_guidance.detach(),
            "loss_action_progress_flow": loss_progress_flow.detach(),
            "action_progress_flow_weight": loss_total.new_tensor(
                lambda_progress_flow
            ).detach(),
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
            "condition_dropout_ratio": dropout_mask.to(dtype=loss_total.dtype)
            .mean()
            .detach(),
            "time_contract_rho": loss_total.new_tensor(rho).detach(),
            # interaction cue loss
            "loss_interaction_stage": loss_interaction_stage.detach(),
            "loss_interaction_progress": loss_interaction_progress.detach(),
            "loss_interaction_object": loss_interaction_object.detach(),
            "loss_interaction_contact": loss_interaction_contact.detach(),
            "loss_interaction_place": loss_interaction_place.detach(),
            "loss_interaction_motion": loss_interaction_motion.detach(),
            "loss_interaction_aux": loss_interaction_aux.detach(),
            "interaction_aux_schedule_scale": loss_total.new_tensor(
                interaction_aux_schedule_scale_value
            ).detach(),
            # interaction cue lambdas
            "interaction_aux_weight": loss_total.new_tensor(interaction_aux_weight).detach(),
            "interaction_stage_weight": _metric_tensor(lambda_interaction_stage_eff),
            "interaction_progress_weight": _metric_tensor(lambda_interaction_progress_eff),
            "interaction_object_weight": _metric_tensor(lambda_interaction_object_eff),
            "interaction_contact_weight": _metric_tensor(lambda_interaction_contact_eff),
            "interaction_place_weight": _metric_tensor(lambda_interaction_place_eff),
            "interaction_motion_weight": _metric_tensor(lambda_interaction_motion_eff),
            
            "interaction_stage_log_var": self.interaction_aux_log_vars[0].detach(),
            "interaction_progress_log_var": self.interaction_aux_log_vars[1].detach(),
            "interaction_object_log_var": self.interaction_aux_log_vars[2].detach(),
            "interaction_contact_log_var": self.interaction_aux_log_vars[3].detach(),
            "interaction_place_log_var": self.interaction_aux_log_vars[4].detach(),
            "interaction_motion_log_var": self.interaction_aux_log_vars[5].detach(),
            "interaction_stage_acc": interaction_stage_acc.detach(),
            "interaction_object_acc": interaction_object_acc.detach(),
            "interaction_progress_mae": interaction_progress_mae.detach(),
            "interaction_context_norm": interaction_context_norm.detach(),
        }

__all__ = [
    "InteractionAwareShortcutPolicy",
]
