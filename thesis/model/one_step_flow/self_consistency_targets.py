from __future__ import annotations

import math
from typing import Callable, Dict

import torch


def _broadcast_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], *([1] * (target.ndim - 1)))


def _require_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value}.")


def time_contract_rho(
    *,
    training_progress: float,
    start: float = 1.0,
    end: float = 0.0,
    schedule: str = "linear",
    power: float = 2.0,
) -> float:
    """Return the OFP time-contracting upper bound rho(s).
    
       args: 
         training_progress: A float in [0, 1] indicating the training progress.
         start: initial value of p, start of training (default 1.0 means no contraction at the start).
         end: final value of p, end of training (default 0.0 means full contraction at the end).
    
    """
    
    progress = min(max(float(training_progress), 0.0), 1.0)
    start = min(max(float(start), 0.0), 1.0)
    end = min(max(float(end), 0.0), 1.0)

    if schedule == "linear":
        value = start + (end - start) * progress
    elif schedule == "polynomial":
        #  1 - train progress ** power
        decay = (1.0 - progress) ** max(float(power), 0.0)
        # contraction value p(s)
        value = end + (start - end) * decay
    elif schedule == "constant":
        value = start
    else:
        raise ValueError(
            "time_contract_schedule must be 'linear', 'polynomial', or 'constant', "
            f"got {schedule!r}."
        )
    return min(max(float(value), 0.0), 1.0)


# def sample_contracted_middle_fraction(
#     *,
#     batch_size: int,
#     device: torch.device,
#     dtype: torch.dtype,
#     training_progress: float,
#     time_contract_start: float = 1.0,
#     time_contract_end: float = 0.0,
#     time_contract_schedule: str = "linear",
#     time_contract_power: float = 2.0,
#     min_middle_fraction: float = 0.0,
# ) -> torch.Tensor:
#     """Sample middle fraction from U[min_middle_fraction, rho(s)].
    
#         Args:
#             batch_size: number of samples to generate.
#             device: torch device for the output tensor.
#             dtype: torch dtype for the output tensor.
#             training_progress: A float in [0, 1] indicating the training progress.
#             time_contract_start: initial value of p, start of training (default 1.0 means no contraction at the start).
#             time_contract_end: final value of p, end of training (default 0.0 means full contraction at the end).
#             time_contract_schedule: schedule type for rho(s), one of 'linear', 'polynomial', or 'constant'.
#             time_contract_power: power for polynomial schedule (ignored for other schedules).
#             min_middle_fraction: minimum value for the middle fraction (default 0.0).
    
#     """
    
#     # current contraction value p(s) based on training progressi
#     p_s = time_contract_rho(
#         training_progress=training_progress,
#         start=time_contract_start,
#         end=time_contract_end,
#         schedule=time_contract_schedule,
#         power=time_contract_power,
#     )
#     # clamp p to [0,1] and ensure min_middle_fraction <= p
#     upper = min(max(p_s, 0.0), 1.0)
#     lower = min(max(float(min_middle_fraction), 0.0), upper)
#     fraction = torch.rand(batch_size, device=device, dtype=dtype)
#     return lower + (upper - lower) * fraction

def sample_contracted_middle_time(
    *,
    t: torch.Tensor,
    r: torch.Tensor,
    training_progress: float,
    time_contract_start: float = 1.0,
    time_contract_end: float = 0.0,
    time_contract_schedule: str = "polynomial",
    time_contract_power: float = 2.0,
) -> torch.Tensor:
    
    """
    Returns intermediate time m
    """
    
    p_s = time_contract_rho(
        training_progress=training_progress,
        start=time_contract_start,
        end=time_contract_end,
        schedule=time_contract_schedule,
        power=time_contract_power,
    )
    
    # clamp p(s) to [0,1]
    p_s = min(max(float(p_s), 0.0), 1.0)
    
    # compute upper bound of interval, t + (r-t) * p(s)
    m_upper = t + (r - t) * p_s
    
    # sample intermediate time uniformly from [t, upper bound]
    m = t + (m_upper - t) * torch.rand_like(t) # if rand value is 0, m=t; if rand value is 1, m=m_upper
    return m

def sample_self_consistency_times_and_interval(
    *,
    batch_size: int,
    denoise_timesteps: int,
    device: torch.device,
    dtype: torch.dtype,
    discrete_time: bool = False,
    t_sampler: str = "beta",
    t_beta_alpha: float = 1.0,
    t_beta_beta: float = 1.5,
    interval_sampler: str = "logit_normal",
    interval_mu: float = -0.2,
    interval_sigma: float = 1.0,
    min_self_consistency_step: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample OFP start time t and feasible interval d with t + d <= 1.
    
       Args:
            batch_size: number of samples to generate.
            denoise_timesteps: number of discrete timesteps in the diffusion process (used if discrete_time is True).
            device: torch device for the output tensors.
            dtype: torch dtype for the output tensors.
            discrete_time: whether to sample times on a discrete grid defined by denoise_timesteps.
            t_sampler: method to sample start time t, one of 'beta' or 'uniform'.
            t_beta_alpha: alpha parameter for beta distribution (if t_sampler is 'beta').
            t_beta_beta: beta parameter for beta distribution (if t_sampler is 'beta').
            interval_sampler: method to sample self-consistency interval, one of 'logit_normal', 'log_normal', 'uniform', or 'power_of_two'.
            interval_mu: mu parameter for logit normal or log normal distribution (if interval_sampler is 'logit_normal' or 'log_normal').
            interval_sigma: sigma parameter for logit normal or log normal distribution (if interval_sampler is 'logit_normal' or 'log_normal').
            min_self_consistency_step: minimum allowed interval length d to ensure numerical stability (if None, defaults to machine epsilon for the dtype).
            
        Returns:
            A tuple (t, r - t) 
    """
    
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    # minimum interval length to ensure t + d <= 1 and numerical stability
    min_step = (
        torch.finfo(dtype).eps
        if min_self_consistency_step is None
        else max(float(min_self_consistency_step), torch.finfo(dtype).eps)
    )
    
    # make sure t not sampled at exactly 1.0 to allow for a positive interval
    max_t = max(1.0 - min_step, 0.0)

    if t_sampler == "beta":
        # create beta dist params
        alpha = torch.tensor(float(t_beta_alpha), device=device, dtype=dtype)
        beta = torch.tensor(float(t_beta_beta), device=device, dtype=dtype)
        if alpha <= 0 or beta <= 0:
            raise ValueError("t_beta_alpha and t_beta_beta must be positive.")
        
        # sample t from beta distr
        t = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        t = t.to(device=device, dtype=dtype) * max_t
    elif t_sampler == "uniform":
        t = torch.rand(batch_size, device=device, dtype=dtype) * max_t
    else:
        raise ValueError(f"Unsupported t_sampler={t_sampler!r}.")

    # compute remaining interval after sampling t
    remaining = torch.clamp(1.0 - t, min=min_step)
    
    if interval_sampler == "logit_normal":
        # sample gaussian noise
        eps = torch.randn(batch_size, device=device, dtype=dtype)
        
        # transform gaussian noise to (0,1) interval using logit normal transformation
        fraction = torch.sigmoid(float(interval_mu) + float(interval_sigma) * eps)
        
        # scale fraction by remaining interval to get interval length, and clamp to minimum step
        self_consistency_step = torch.clamp(remaining * fraction, min=min_step) # r - t
    elif interval_sampler in {"log_normal", "lognormal"}:
        # sample gaussian noise
        eps = torch.randn(batch_size, device=device, dtype=dtype)
        
        # transform gaussian noise to positive values using log normal transformation
        raw = torch.exp(float(interval_mu) + float(interval_sigma) * eps)
        
        # transform to (0,1) interval using raw / (1 + raw) to ensure interval length is positive
        fraction = raw / (1.0 + raw)
        
        self_consistency_step = torch.clamp(remaining * fraction, min=min_step)
    elif interval_sampler == "uniform":
        # sample fraction uniformly from (0,1) and scale by remaining interval
        fraction = torch.rand(batch_size, device=device, dtype=dtype)
        
        # clamp to minimum step to ensure numerical stability and t + d <= 1
        self_consistency_step = torch.clamp(remaining * fraction, min=min_step)
    elif interval_sampler == "power_of_two":
        _require_power_of_two(int(denoise_timesteps), "denoise_timesteps")
        log2_sections = int(math.log2(int(denoise_timesteps)))
        interval_exp = torch.randint(
            low=0,
            high=log2_sections + 1,
            size=(batch_size,),
            device=device,
        )
        interval_sections = torch.pow(
            torch.tensor(2.0, device=device, dtype=dtype),
            interval_exp.to(dtype=dtype),
        )
        self_consistency_step = 1.0 / interval_sections
        if discrete_time:
            max_start = interval_sections.to(dtype=torch.long)
            t_idx = torch.floor(
                torch.rand(batch_size, device=device, dtype=dtype)
                * max_start.to(dtype=dtype)
            )
            t = t_idx / interval_sections
        else:
            t = torch.rand(batch_size, device=device, dtype=dtype) * (
                1.0 - self_consistency_step
            )
        return t, self_consistency_step
    else:
        raise ValueError(f"Unsupported interval_sampler={interval_sampler!r}.")

    if discrete_time:
        grid = float(denoise_timesteps)
        # get timestep index in denoise trajectory
        t = torch.floor(t * grid) / grid
        
        # get interval length in denoise trajectory
        self_consistency_step = torch.ceil(self_consistency_step * grid) / grid
        # ensure interval does not pass time 1.0 in denoise trajectory 
        self_consistency_step = torch.minimum(
            self_consistency_step,
            torch.clamp(1.0 - t, min=min_step),
        )
        # clamp to minimum step
        self_consistency_step = torch.clamp(self_consistency_step, min=min_step)

    self_consistency_step = torch.minimum(
        self_consistency_step,
        torch.clamp(1.0 - t, min=min_step),
    )
    
    # return start time t and interval length d = r - t
    return t, self_consistency_step


sample_self_consistency_times_and_steps = sample_self_consistency_times_and_interval


def make_flow_targets(
    x1: torch.Tensor,
    *,
    condition_data: torch.Tensor,
    condition_mask: torch.Tensor,
    denoise_timesteps: int,
    discrete_time: bool = True,
) -> Dict[str, torch.Tensor]:
    """Create OFP diagonal flow targets u(z_t, t, t | o).
    
        Args:
            x1: clean action sequence tensor of shape (b# The code you provided is a Python comment.
            # Comments in Python are lines of text that
            # are ignored by the Python interpreter and
            # are used to provide explanations or notes
            # within the code for better understanding.
            # In this case, the comment appears to be
            # describing a function or method that takes
            # in parameters such as `batch_size` and
            # others.
            atch_size, ...).
            condition_data: conditioning data tensor of shape (batch_size, ...).
            condition_mask: boolean mask tensor of shape (batch_size, ...) where True indicates positions to replace with condition_data.
            denoise_timesteps: number of discrete timesteps in the diffusion process (used if discrete_time is True).
            discrete_time: whether to sample times on a discrete grid defined by denoise_timesteps.
    """
    batch_size = x1.shape[0]
    device = x1.device
    dtype = x1.dtype

    # sample timestep t for each sample in the batch
    if discrete_time:
        t = torch.randint(
            low=0,
            high=int(denoise_timesteps),
            size=(batch_size,),
            device=device,
        ).to(dtype=dtype)
        t = t / float(denoise_timesteps)
    else:
        t = torch.rand(batch_size, device=device, dtype=dtype)

    # sample gaussian noise
    x0 = torch.randn_like(x1)
    
    # interpolate noisy action sequence 
    x_t = (1.0 - _broadcast_like(t, x1)) * x0 + _broadcast_like(t, x1) * x1
    x_t = torch.where(condition_mask, condition_data, x_t)

    return {
        "x_t": x_t,
        "target": x1 - x0,
        "t": t,
        "r": t,
        "self_consistency_step": torch.zeros_like(t),
    }


make_flow_anchor_targets = make_flow_targets


@torch.no_grad()
def make_self_consistency_targets(
    x1: torch.Tensor,
    *,
    condition_data: torch.Tensor,
    condition_mask: torch.Tensor,
    global_cond: torch.Tensor,
    teacher_model_fn: Callable[..., torch.Tensor],
    denoise_timesteps: int,
    discrete_time: bool = True,
    clip_sample: float | None = None,
    training_progress: float = 0.0,
    time_contract_start: float = 1.0,
    time_contract_end: float = 0.0,
    time_contract_schedule: str = "linear",
    time_contract_power: float = 2.0,
    min_middle_fraction: float = 0.0,
    t_sampler: str = "beta",
    t_beta_alpha: float = 1.0,
    t_beta_beta: float = 1.5,
    interval_sampler: str = "logit_normal",
    interval_mu: float = -0.2,
    interval_sigma: float = 1.0,
    min_self_consistency_step: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Create OFP self-consistency targets from an EMA teacher."""
    del min_middle_fraction
    
    # if batch is empty, raise error since we cannot create targets
    if x1.shape[0] == 0:
        raise ValueError("Input batch is empty. Cannot create self-consistency targets.")

    batch_size = x1.shape[0]
    device = x1.device
    dtype = x1.dtype
    
    # get start time t and r - t interval for self-consistency target
    t, self_consistency_interval = sample_self_consistency_times_and_interval(
        batch_size=batch_size,
        denoise_timesteps=int(denoise_timesteps),
        device=device,
        dtype=dtype,
        discrete_time=bool(discrete_time),
        t_sampler=str(t_sampler),
        t_beta_alpha=float(t_beta_alpha),
        t_beta_beta=float(t_beta_beta),
        interval_sampler=str(interval_sampler),
        interval_mu=float(interval_mu),
        interval_sigma=float(interval_sigma),
        min_self_consistency_step=min_self_consistency_step,
    )
    # interval endpoint
    r = t + self_consistency_interval
    
    # middle_fraction = sample_contracted_middle_fraction(
    #     batch_size=batch_size,
    #     device=device,
    #     dtype=dtype,
    #     training_progress=float(training_progress),
    #     time_contract_start=float(time_contract_start),
    #     time_contract_end=float(time_contract_end),
    #     time_contract_schedule=str(time_contract_schedule),
    #     time_contract_power=float(time_contract_power),
    #     min_middle_fraction=float(min_middle_fraction),
    # )
    # m = t + self_consistency_step * middle_fraction
    
    # intermediate time m
    m = sample_contracted_middle_time(
        t=t,
        r=r,
        training_progress=float(training_progress),
        time_contract_start=float(time_contract_start),
        time_contract_end=float(time_contract_end),
        time_contract_schedule=str(time_contract_schedule),
        time_contract_power=float(time_contract_power),
    )
    
    # get intermediate interval for ema prediction, r - m
    teacher_interval = r - m

    # sample gaussian noise
    x0 = torch.randn_like(x1)
    
    # interpolate to get noisy action sequences at times t and m
    z_t = (1.0 - _broadcast_like(t, x1)) * x0 + _broadcast_like(t, x1) * x1
    z_m = (1.0 - _broadcast_like(m, x1)) * x0 + _broadcast_like(m, x1) * x1
    
    z_t = torch.where(condition_mask, condition_data, z_t)
    z_m = torch.where(condition_mask, condition_data, z_m)

    # predict velocity from ema model at intermediate time m for interval r - m
    teacher_velocity = teacher_model_fn(
        z_m,
        m,
        global_cond=global_cond,
        shortcut_step=teacher_interval,
    )
    
    # action prediction at time r from ema velocity prediction, z_m + (r - m) * velocity
    z_r_hat = z_m + _broadcast_like(teacher_interval, z_m) * teacher_velocity
    if clip_sample is not None:
        z_r_hat = torch.clamp(z_r_hat, -float(clip_sample), float(clip_sample))
    z_r_hat = torch.where(condition_mask, condition_data, z_r_hat)

    # create target as (z_r_hat - z_t) / (r - t)
    target = (z_r_hat - z_t) / _broadcast_like(self_consistency_interval, z_t)
    if clip_sample is not None:
        target = torch.clamp(target, -float(clip_sample), float(clip_sample))

    return {
        "x_t": z_t.detach(),
        "target": target.detach(),
        "t": t.detach(),
        "m": m.detach(),
        "r": r.detach(),
        "self_consistency_step": self_consistency_interval.detach(),
        "teacher_self_consistency_step": teacher_interval.detach(),
        "teacher_velocity": teacher_velocity.detach(),
        # "middle_fraction": middle_fraction.detach(),
        "middle_fraction": ((m - t) / self_consistency_interval).detach(),
        
        # p(s)
        "time_contract_rho": torch.full_like(
            t,
            time_contract_rho(
                training_progress=float(training_progress),
                start=float(time_contract_start),
                end=float(time_contract_end),
                schedule=str(time_contract_schedule),
                power=float(time_contract_power),
            ),
        ).detach(),
    }


def make_one_step_flow_training_batch(
    x1: torch.Tensor,
    *,
    condition_data: torch.Tensor,
    condition_mask: torch.Tensor,
    global_cond: torch.Tensor,
    teacher_model_fn: Callable[..., torch.Tensor],
    denoise_timesteps: int,
    self_consistency_every: int,
    discrete_time: bool = True,
    clip_sample: float | None = None,
    training_progress: float = 0.0,
    time_contract_start: float = 1.0,
    time_contract_end: float = 0.0,
    time_contract_schedule: str = "linear",
    time_contract_power: float = 2.0,
    min_middle_fraction: float = 0.0,
    t_sampler: str = "beta",
    t_beta_alpha: float = 1.0,
    t_beta_beta: float = 1.5,
    interval_sampler: str = "logit_normal",
    interval_mu: float = -0.2,
    interval_sigma: float = 1.0,
    min_self_consistency_step: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Merge OFP flow-anchor and self-consistency targets."""
    
    if self_consistency_every <= 0:
        raise ValueError(
            "self_consistency_every must be positive, "
            f"got {self_consistency_every}."
        )

    batch_size = x1.shape[0]
    
    # self consistency size
    self_consistency_size = batch_size // int(self_consistency_every)
    # flow matching size is the rest of the batch after allocating self-consistency samples
    flow_size = batch_size - self_consistency_size

    # container to hold two sub batches of targets and related info before concatenating into one batch at the end
    pieces: list[Dict[str, torch.Tensor]] = []
    condition_masks = []
    global_conds = []
    is_self_consistency_parts = []

    if self_consistency_size > 0:
        # create self-consistency targets for the sc size in the batch
        self_consistency = make_self_consistency_targets(
            x1[:self_consistency_size],
            condition_data=condition_data[:self_consistency_size],
            condition_mask=condition_mask[:self_consistency_size],
            global_cond=global_cond[:self_consistency_size],
            teacher_model_fn=teacher_model_fn,
            denoise_timesteps=int(denoise_timesteps),
            discrete_time=bool(discrete_time),
            clip_sample=clip_sample,
            training_progress=float(training_progress),
            time_contract_start=float(time_contract_start),
            time_contract_end=float(time_contract_end),
            time_contract_schedule=str(time_contract_schedule),
            time_contract_power=float(time_contract_power),
            min_middle_fraction=float(min_middle_fraction),
            t_sampler=str(t_sampler),
            t_beta_alpha=float(t_beta_alpha),
            t_beta_beta=float(t_beta_beta),
            interval_sampler=str(interval_sampler),
            interval_mu=float(interval_mu),
            interval_sigma=float(interval_sigma),
            min_self_consistency_step=min_self_consistency_step,
        )
        pieces.append(self_consistency)
        condition_masks.append(condition_mask[:self_consistency_size])
        global_conds.append(global_cond[:self_consistency_size])
        
        # boolean mask to indicate which samples in the batch are self-consistency targets
        is_self_consistency_parts.append(
            torch.ones(self_consistency_size, device=x1.device, dtype=torch.bool)
        )

    if flow_size > 0:
        # create flow matching target for the rest of the batch after allocating self-consistency samples
        flow_start = self_consistency_size
        flow = make_flow_targets(
            x1[flow_start:],
            condition_data=condition_data[flow_start:],
            condition_mask=condition_mask[flow_start:],
            denoise_timesteps=int(denoise_timesteps),
            discrete_time=bool(discrete_time),
        )
        pieces.append(flow)
        condition_masks.append(condition_mask[flow_start:])
        global_conds.append(global_cond[flow_start:])
        
        # boolean mask to indicate which samples in the batch are flow matching targets
        is_self_consistency_parts.append(
            torch.zeros(flow_size, device=x1.device, dtype=torch.bool)
        )

    if not pieces:
        raise ValueError("Cannot build one-step flow targets from an empty batch.")

    def cat_or_none(key: str) -> torch.Tensor | None:
        values = [piece[key] for piece in pieces if key in piece]
        if len(values) != len(pieces):
            return None
        return torch.cat(values, dim=0)

    # concat all model inputs and targets into one batch
    out = {
        "x_t": torch.cat([piece["x_t"] for piece in pieces], dim=0),
        "target": torch.cat([piece["target"] for piece in pieces], dim=0).detach(),
        "t": torch.cat([piece["t"] for piece in pieces], dim=0),
        "r": torch.cat([piece["r"] for piece in pieces], dim=0),
        
        # self consistency interval
        "self_consistency_step": torch.cat(
            [piece["self_consistency_step"] for piece in pieces],
            dim=0,
        ),
        "global_cond": torch.cat(global_conds, dim=0),
        "condition_mask": torch.cat(condition_masks, dim=0),
        "is_self_consistency": torch.cat(is_self_consistency_parts, dim=0),
        "self_consistency_start_idx": x1.new_tensor(0.0),
        "self_consistency_end_idx": x1.new_tensor(float(self_consistency_size)),
        "flow_start_idx": x1.new_tensor(float(self_consistency_size)),
        "flow_end_idx": x1.new_tensor(float(batch_size)),
    }

    # for optional_key in (
    #     "m",
    #     "teacher_self_consistency_step",
    #     "teacher_velocity",
    #     "middle_fraction",
    #     "time_contract_rho",
    # ):
    #     value = cat_or_none(optional_key)
    #     if value is not None:
    #         out[optional_key] = value

    out["self_consistency_ratio"] = out["is_self_consistency"].to(dtype=x1.dtype).mean()
    return out


__all__ = [
    "make_flow_targets",
    "make_flow_anchor_targets",
    "make_self_consistency_targets",
    "make_one_step_flow_training_batch",
    "sample_self_consistency_times_and_interval",
    "sample_self_consistency_times_and_steps",
    "sample_contracted_middle_fraction",
    "time_contract_rho",
]
