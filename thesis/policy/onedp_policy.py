from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from thesis.model.onedp.noise_schedule import make_noise_schedule


class OneStepDiffusionPolicy(nn.Module):
    """OneDP-S / OneDP-D wrapper around a frozen Diffusion Policy teacher."""

    def __init__(
        self,
        teacher_policy: nn.Module,
        variant: str = "stochastic",
        t_init: int = 65,
        t_min: int = 2,
        t_max: int = 95,
        noise_scheduler=None,
    ):
        super().__init__()
        if variant not in ("stochastic", "deterministic"):
            raise ValueError("variant must be 'stochastic' or 'deterministic'.")

        self.variant = variant
        self.teacher_policy = copy.deepcopy(teacher_policy).eval()
        
        # freeze teacher policy
        for param in self.teacher_policy.parameters():
            param.requires_grad_(False)

        # warm start action generator and score model from teacher policy
        self.generator_model = copy.deepcopy(teacher_policy.model)
        self.score_model = copy.deepcopy(teacher_policy.model) if variant == "stochastic" else None # only OneDP-S uses a score model
        
        # frozen teacher model
        self.teacher_model = self.teacher_policy.model
        self.normalizer = copy.deepcopy(teacher_policy.normalizer)
        if noise_scheduler is None:
            noise_scheduler = teacher_policy.noise_scheduler
        self.schedule = make_noise_schedule(noise_scheduler, t_min=t_min, t_max=t_max)
        
        # initial noise timestep
        self.t_init = int(t_init)

        self.generator_obs_encoder = None
        self.score_obs_encoder = None
        self.teacher_obs_encoder = None
        if hasattr(teacher_policy, "obs_encoder"):
            # if teacher has an obs encoder, warm start generator and score obs encoders from it as well, but keep a frozen copy for teacher conditioning
            self.generator_obs_encoder = copy.deepcopy(teacher_policy.obs_encoder)
            self.teacher_obs_encoder = copy.deepcopy(teacher_policy.obs_encoder).eval()
            # freeze teacher obs encoder
            for param in self.teacher_obs_encoder.parameters():
                param.requires_grad_(False)
            if variant == "stochastic":
                self.score_obs_encoder = copy.deepcopy(teacher_policy.obs_encoder)

        # make generator and score model and its obs encoder parameters trainable
        for param in self.generator_model.parameters():
            param.requires_grad_(True)
        if self.score_model is not None:
            for param in self.score_model.parameters():
                param.requires_grad_(True)
                
        if self.generator_obs_encoder is not None:
            for param in self.generator_obs_encoder.parameters():
                param.requires_grad_(True)
        if self.score_obs_encoder is not None:
            for param in self.score_obs_encoder.parameters():
                param.requires_grad_(True)

        self.policy_kind = self._detect_policy_kind(teacher_policy)
        self.horizon = int(teacher_policy.horizon)
        self.action_dim = int(teacher_policy.action_dim)
        self.n_obs_steps = int(teacher_policy.n_obs_steps)
        self.n_action_steps = int(teacher_policy.n_action_steps)
        self.pred_action_steps_only = bool(getattr(teacher_policy, "pred_action_steps_only", False))

    @staticmethod
    def _detect_policy_kind(policy: nn.Module) -> str:
        if hasattr(policy, "obs_encoder"):
            if not getattr(policy, "obs_as_global_cond", True):
                raise NotImplementedError("OneDP image support requires obs_as_global_cond=True.")
            return "image_unet"
        if hasattr(policy, "obs_as_cond"):
            return "lowdim_transformer"
        if hasattr(policy, "obs_as_global_cond") or hasattr(policy, "obs_as_local_cond"):
            if not (getattr(policy, "obs_as_global_cond", False) or getattr(policy, "obs_as_local_cond", False)):
                raise NotImplementedError("OneDP lowdim U-Net support requires obs_as_global_cond or obs_as_local_cond.")
            return "lowdim_unet"
        raise NotImplementedError(f"Unsupported teacher policy type: {type(policy)!r}")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def reset(self):
        pass

    def train(self, mode: bool = True):
        result = super().train(mode)
        # The teacher must stay frozen in eval mode so transformer dropout
        # does not perturb the distillation target during training.
        self.teacher_policy.eval()
        if self.teacher_obs_encoder is not None:
            self.teacher_obs_encoder.eval()
        return result

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.schedule.to(self.device)
        return result

    def generator_parameters(self):
        # model params
        params = list(self.generator_model.parameters())
        if self.generator_obs_encoder is not None:
            # obs encoder params for image-based policies
            params += list(self.generator_obs_encoder.parameters())
        return params

    def score_parameters(self):
        if self.score_model is None:
            return []
        params = list(self.score_model.parameters())
        if self.score_obs_encoder is not None:
            params += list(self.score_obs_encoder.parameters())
        return params

    def _action_shape(self):
        if self.pred_action_steps_only:
            return (self.n_action_steps, self.action_dim)
        return (self.horizon, self.action_dim)

    def _normalize_obs(self, obs):
        if self.policy_kind == "image_unet":
            return self.normalizer.normalize(obs)
        return self.normalizer["obs"].normalize(obs)

    def _condition_from_batch(self, batch: Dict, role: str) -> Dict:
        obs = batch["obs"]
        if self.policy_kind == "image_unet":
            return self._condition_image(obs, role)
        return self._condition_lowdim(obs)

    def _condition_from_obs_dict(self, obs_dict: Dict[str, torch.Tensor]) -> Dict:
        obs = obs_dict["obs"] if self.policy_kind != "image_unet" else obs_dict
        if self.policy_kind == "image_unet":
            return self._condition_image(obs, "generator")
        return self._condition_lowdim(obs)

    def _condition_image(self, obs_dict: Dict[str, torch.Tensor], role: str) -> Dict:
        # normalize raw obs dict and encode with the appropriate encoder depending on role
        nobs = self._normalize_obs(obs_dict)    # shape (batch, obs_time_steps, C, H, W) for image obs
        value = next(iter(nobs.values()))   # grab one of the obs tensors to get batch size
        batch_size = value.shape[0]
        encoder = self.generator_obs_encoder
        if role == "score" and self.score_obs_encoder is not None:
            encoder = self.score_obs_encoder
        if role == "teacher":
            encoder = self.teacher_obs_encoder
        # merge batch and time (obs time step) dimensions for encoding
        flat = {
            key: val[:, : self.n_obs_steps].reshape(-1, *val.shape[2:])
            for key, val in nobs.items()
        }
        # encode obs images and reshape to (batch, n_obs_steps * feature_dim)
        features = encoder(flat).reshape(batch_size, -1)
        # return conditioning input
        return {"global_cond": features}

    def _condition_lowdim(self, obs: torch.Tensor) -> Dict:
        # normalize raw obs dict 
        nobs = self._normalize_obs(obs) # shape (batch, obs_time_steps, obs_dim)
        # return obs until n_obs_steps as conditioning input, keeps timestep dim
        if self.policy_kind == "lowdim_transformer":
            return {"cond": nobs[:, : self.n_obs_steps]}
        teacher = self.teacher_policy
        # for unet, take the first n_obs_steps, then flattens time and feature dimensions together
        if getattr(teacher, "obs_as_global_cond", False):
            return {"global_cond": nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)}
        
        # for unet, if obs_as_local_cond is True, return the first n_obs_steps as a separate local conditioning tensor with the same shape as the model input, and the model can attend to it with cross attention at each timestep. This is more flexible than global cond, since the model can choose which obs time steps to attend to at each diffusion step, but also more expensive since it requires cross attention over the obs at each diffusion step.
        if getattr(teacher, "obs_as_local_cond", False):
            # creates a zero tensor over the whole prediction horizon, then inserts the observed timesteps at the beginning.
            local_cond = torch.zeros(
                (nobs.shape[0], self.horizon, nobs.shape[-1]),
                device=nobs.device,
                dtype=nobs.dtype,
            )
            local_cond[:, : self.n_obs_steps] = nobs[:, : self.n_obs_steps]
            return {"local_cond": local_cond}
        raise NotImplementedError("Unsupported lowdim conditioning mode.")

    def _model_forward(self, model: nn.Module, sample: torch.Tensor, timesteps: torch.Tensor, cond: Dict):
        if self.policy_kind == "lowdim_transformer":
            return model(sample, timesteps, cond.get("cond"))
        return model(
            sample=sample,
            timestep=timesteps,
            local_cond=cond.get("local_cond"),
            global_cond=cond.get("global_cond"),
        )

    def _generate(self, cond: Dict, batch_size: int, noise: Optional[torch.Tensor] = None):
        """
        Student generator forward pass to produce an initial noisy action prediction (if stochastic) or initial action prediction (if deterministic)
        """
        action_shape = self._action_shape()
        if noise is None:
            if self.variant == "stochastic":
                noise = torch.randn(batch_size, *action_shape, device=self.device, dtype=self.dtype)
            else:
                noise = torch.zeros(batch_size, *action_shape, device=self.device, dtype=self.dtype)
        timesteps = self.schedule.initial_timesteps(batch_size, self.device, self.t_init)
        model_fn = lambda sample, ts: self._model_forward(self.generator_model, sample, ts, cond)
        return self.schedule.generate_x0(model_fn, noise, timesteps)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # obs history conditioning 
        cond_g = self._condition_from_batch(batch, "generator")
        cond_t = self._condition_from_batch(batch, "teacher")
        cond_s = self._condition_from_batch(batch, "score") if self.variant == "stochastic" else None
        batch_size = next(iter(batch["obs"].values())).shape[0] if isinstance(batch["obs"], dict) else batch["obs"].shape[0]

        # Student action
        generated = self._generate(cond_g, batch_size) # model forward generator
        timesteps = self.schedule.sample_timesteps(batch_size, self.device, reference=generated)

        if self.variant == "stochastic":
            # Noisy sample & true noise
            noisy_detached, noise = self.schedule.q_sample(generated.detach(), timesteps)
            # compute pred noise from score network
            score_fn = lambda sample, ts: self._model_forward(self.score_model, sample, ts, cond_s)
            pred_noise = self.schedule.model_output(score_fn, noisy_detached, timesteps)
            
            loss_score = self.schedule.score_loss(pred_noise, noise, generated.detach(), timesteps) # eq. 6, loss function for score network
            
            # OneDP follows DDPM epsilon-prediction here, and the paper fixes
            # lambda(k)=1 in the discrete-time setting.

            # noisy student action with grad of noisy action wrt student model params
            noisy_with_grad, _ = self.schedule.q_sample(generated, timesteps, noise=noise)
            with torch.no_grad():
                teacher_fn = lambda sample, ts: self._model_forward(self.teacher_model, sample, ts, cond_t)
                pred_teacher = self.schedule.model_output(teacher_fn, noisy_detached, timesteps)
                # Eq. 5: fixed distillation direction for the generator update.
                weight = self.schedule.distillation_weight(timesteps)
                direction = weight * self.schedule.direction(
                    pred_teacher, pred_noise, noise, generated.detach(), timesteps
                )
            loss_generator = (direction * noisy_with_grad).mean()   # eq. 5, loss function for generator network
        else:
            noisy_with_grad, noise = self.schedule.q_sample(generated, timesteps)
            with torch.no_grad():
                # teacher score
                teacher_fn = lambda sample, ts: self._model_forward(self.teacher_model, sample, ts, cond_t)
                pred_teacher = self.schedule.model_output(teacher_fn, noisy_with_grad.detach(), timesteps)
                weight = self.schedule.distillation_weight(timesteps)
                direction = weight * self.schedule.direction(
                    pred_teacher, None, noise, generated.detach(), timesteps
                )
            loss_generator = (direction * noisy_with_grad).mean()
            loss_score = torch.zeros((), device=self.device, dtype=self.dtype)

        return {
            "loss_generator": loss_generator,
            "loss_score": loss_score,
            "loss_total": loss_generator + loss_score,
            "train_loss_score": loss_score,
            "train_loss_total": loss_generator + loss_score,
        }

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cond = self._condition_from_obs_dict(obs_dict)
        value = next(iter(obs_dict.values())) if self.policy_kind == "image_unet" else obs_dict["obs"]
        batch_size = value.shape[0]
        normalized_action_pred = self._generate(cond, batch_size)
        action_pred = self.normalizer["action"].unnormalize(normalized_action_pred)

        if self.pred_action_steps_only:
            action = action_pred
        elif self.policy_kind == "lowdim_unet":
            start = self.n_obs_steps
            if getattr(self.teacher_policy, "oa_step_convention", False):
                start = self.n_obs_steps - 1
            action = action_pred[:, start : start + self.n_action_steps]
        else:
            start = self.n_obs_steps - 1
            action = action_pred[:, start : start + self.n_action_steps]
        return {"action": action, "action_pred": action_pred}
