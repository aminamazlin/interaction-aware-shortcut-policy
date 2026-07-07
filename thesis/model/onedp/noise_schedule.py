from __future__ import annotations

from typing import Optional, Tuple

import torch


class TeacherDDPMSchedule:
    """DDPM helper built from a diffusers DDPM/DDIM-style scheduler."""

    name = "ddpm"

    def __init__(self, noise_scheduler, t_min: int = 2, t_max: int = 95):
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise TypeError("DDPM scheduler does not expose alphas_cumprod.")

        prediction_type = getattr(noise_scheduler.config, "prediction_type", None)
        if prediction_type != "epsilon":
            raise NotImplementedError(
                "OneDP DDPM distillation expects epsilon prediction, "
                f"got prediction_type={prediction_type!r}."
            )

        alphas_cumprod = noise_scheduler.alphas_cumprod.detach().clone().float()
        self.alpha = alphas_cumprod.sqrt()
        self.sigma = (1.0 - alphas_cumprod).clamp_min(0).sqrt()
        self.t_min = int(t_min)
        self.t_max = min(int(t_max), int(alphas_cumprod.shape[0]) - 1)
        if self.t_min > self.t_max:
            raise ValueError(f"Invalid distillation timestep range [{self.t_min}, {self.t_max}].")

    def to(self, device):
        self.alpha = self.alpha.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device, reference: Optional[torch.Tensor] = None):
        return torch.randint(self.t_min, self.t_max + 1, (batch_size,), device=device)

    def initial_timesteps(self, batch_size: int, device, t_init: int | float):
        return torch.full((batch_size,), int(t_init), device=device, dtype=torch.long)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha = self.alpha[timesteps].view(-1, 1, 1).to(x0)
        sigma = self.sigma[timesteps].view(-1, 1, 1).to(x0)
        return alpha * x0 + sigma * noise, noise

    def model_output(self, model_fn, sample: torch.Tensor, timesteps: torch.Tensor):
        return model_fn(sample, timesteps)

    def generate_x0(self, model_fn, sample: torch.Tensor, timesteps: torch.Tensor):
        pred_noise = self.model_output(model_fn, sample, timesteps)
        alpha = self.alpha[timesteps].view(-1, 1, 1).to(sample)
        sigma = self.sigma[timesteps].view(-1, 1, 1).to(sample)
        return (sample - sigma * pred_noise) / alpha.clamp_min(1e-8)

    def score_loss(self, pred_score, noise, clean, timesteps):
        return ((pred_score - noise) ** 2).mean()

    def distillation_weight(self, timesteps: torch.Tensor):
        return self.sigma[timesteps].view(-1, 1, 1)

    def direction(self, pred_teacher, pred_score, noise, clean, timesteps):
        return pred_teacher - (noise if pred_score is None else pred_score)


class KarrasScheduleAdapter:
    """Adapter around consistency_policy.diffusion.Karras_Scheduler."""

    name = "edm"

    def __init__(self, noise_scheduler):
        self.noise_scheduler = noise_scheduler

    def to(self, device):
        return self

    def sample_timesteps(self, batch_size: int, device, reference: Optional[torch.Tensor] = None):
        if reference is not None and hasattr(self.noise_scheduler, "sample_times"):
            return self.noise_scheduler.sample_times(reference)[0]
        return self.noise_scheduler.log_normal_sampler(batch_size, device)[0]

    def initial_timesteps(self, batch_size: int, device, t_init: int | float):
        time_max = getattr(self.noise_scheduler, "time_max", t_init)
        return torch.full((batch_size,), float(time_max), device=device, dtype=torch.float32)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        noisy = x0 + self.noise_scheduler.trajectory_time_product(noise, timesteps)
        return noisy, noise

    def model_output(self, model_fn, sample: torch.Tensor, timesteps: torch.Tensor):
        return self.noise_scheduler.calc_out(model_fn, sample, timesteps)

    def generate_x0(self, model_fn, sample: torch.Tensor, timesteps: torch.Tensor):
        return self.model_output(model_fn, sample, timesteps)

    def score_loss(self, pred_score, noise, clean, timesteps):
        return ((pred_score - clean.detach()) ** 2).mean()

    def distillation_weight(self, timesteps: torch.Tensor):
        return self.noise_scheduler.get_karras_weightings(timesteps).view(-1, 1, 1)

    def direction(self, pred_teacher, pred_score, noise, clean, timesteps):
        return pred_teacher - (clean if pred_score is None else pred_score)


def make_noise_schedule(noise_scheduler, t_min: int, t_max: int):
    target_name = f"{type(noise_scheduler).__module__}.{type(noise_scheduler).__name__}".lower()
    if hasattr(noise_scheduler, "alphas_cumprod"):
        return TeacherDDPMSchedule(noise_scheduler, t_min=t_min, t_max=t_max)
    if "karras" in target_name or hasattr(noise_scheduler, "calc_out"):
        return KarrasScheduleAdapter(noise_scheduler)
    raise TypeError(f"Unsupported OneDP noise scheduler: {type(noise_scheduler)!r}")
