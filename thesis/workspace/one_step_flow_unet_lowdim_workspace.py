from __future__ import annotations

import copy
import os
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ..model.common.lr_scheduler import get_scheduler
from ..model.diffusion.ema_model import EMAModel
from ..policy.one_step_flow_unet_lowdim_policy import OneStepFlowUnetLowdimPolicy
from ..utils.checkpoint_util import TopKCheckpointManager
from ..utils.json_logger import JsonLogger
from ..utils.pytorch_util import dict_apply, optimizer_to
from .base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

PROGRESS_LOSS_METRICS = (
    "loss_subgoal_stage",
    "loss_subgoal_progress",
    "subgoal_stage_weight",
    "subgoal_progress_weight",
    "subgoal_stage_acc",
    "subgoal_progress_mae",
)
INTERACTION_LOSS_METRICS = (
    "loss_action_progress_flow",
    "action_progress_flow_weight",
    "loss_interaction_stage",
    "loss_interaction_progress",
    "loss_interaction_object",
    "loss_interaction_contact",
    "loss_interaction_place",
    "loss_interaction_motion",
    "loss_interaction_aux",
    "interaction_aux_weight",
    "interaction_stage_weight",
    "interaction_progress_weight",
    "interaction_object_weight",
    "interaction_contact_weight",
    "interaction_place_weight",
    "interaction_motion_weight",
    "interaction_stage_log_var",
    "interaction_progress_log_var",
    "interaction_object_log_var",
    "interaction_contact_log_var",
    "interaction_place_log_var",
    "interaction_motion_log_var",
    "interaction_aux_schedule_scale",
    "interaction_stage_acc",
    "interaction_object_acc",
    "interaction_progress_mae",
    "interaction_context_norm",
)
OPTIONAL_LOSS_METRICS = PROGRESS_LOSS_METRICS + INTERACTION_LOSS_METRICS


def _postprocess_batch(dataset, batch, device: torch.device):
    if hasattr(dataset, "postprocess"):
        return dataset.postprocess(batch, device=device)
    return dict_apply(batch, lambda x: x.to(device, non_blocking=True))


def _clone_sampling_batch(batch, max_batch_size: int = 256):
    batch_size = next(iter(batch.values())).shape[0]
    sample_size = min(batch_size, max_batch_size)
    return dict_apply(batch, lambda x: x[:sample_size].detach().cpu().clone())


def _make_dataloader(dataset, dataloader_cfg):
    return DataLoader(
        dataset,
        **OmegaConf.to_container(dataloader_cfg, resolve=True),
    )


class TrainOneStepFlowUnetLowdimWorkspace(BaseWorkspace):
    """Lowdim One-Step Flow Policy workspace without latent state updater."""

    include_keys = ("global_step", "epoch")

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        if not isinstance(self.model, OneStepFlowUnetLowdimPolicy):
            raise TypeError(
                "TrainOneStepFlowUnetLowdimWorkspace requires "
                "OneStepFlowUnetLowdimPolicy, got "
                f"{self.model.__class__.__module__}.{self.model.__class__.__name__}."
            )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
        print(
            f"Trainable Parameters: {trainable_params:,} "
            f"({trainable_params / 1e6:.2f} M)"
        )

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=self.model.parameters(),
        )
        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def _training_progress(
        *,
        epoch: int,
        batch_idx: int,
        num_epochs: int,
        steps_per_epoch: int,
    ) -> float:
        total_steps = max(int(num_epochs) * max(int(steps_per_epoch), 1), 1)
        current_step = int(epoch) * max(int(steps_per_epoch), 1) + int(batch_idx)
        if total_steps <= 1:
            return 1.0
        return min(max(float(current_step) / float(total_steps - 1), 0.0), 1.0)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            resume_ckpt_path = cfg.training.get("resume_checkpoint_path", None)
            ckpt_path = (
                self.get_checkpoint_path()
                if resume_ckpt_path is None
                else os.path.expanduser(str(resume_ckpt_path))
            )
            if os.path.isfile(ckpt_path):
                print(f"Resuming from checkpoint {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)
            else:
                print(f"Resume requested, but checkpoint not found: {ckpt_path}")
            if cfg.training.get("resume_epoch", None) is not None:
                self.epoch = int(cfg.training.resume_epoch)
                print(f"Override resume epoch -> {self.epoch}")
            if cfg.training.get("resume_global_step", None) is not None:
                self.global_step = int(cfg.training.resume_global_step)
                print(f"Override resume global_step -> {self.global_step}")

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = _make_dataloader(dataset, cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = _make_dataloader(val_dataset, cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            )
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel | None = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None
        if cfg.training.rollout_every is not None:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir,
            )
            print(f"Environment Runner: {env_runner.__class__.__name__}")

        wandb_enabled = str(cfg.logging.mode).lower() != "disabled"
        if wandb_enabled:
            wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({"output_dir": self.output_dir}, allow_val_change=True)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self.optimizer.zero_grad(set_to_none=True)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        steps_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            steps_per_epoch = min(steps_per_epoch, int(cfg.training.max_train_steps))
        steps_per_epoch = max(int(steps_per_epoch), 1)

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}
                train_running: dict[str, list[float]] = {
                    "train_loss": [],
                    "train_loss_main": [],
                    "train_loss_flow": [],
                    "train_loss_self_consistency": [],
                    "train_loss_self_guidance": [],
                    "train_self_consistency_ratio": [],
                    "train_self_guidance_ratio": [],
                    "train_self_guidance_weight": [],
                    "train_self_guidance_pred_norm": [],
                    "train_self_guidance_delta_norm": [],
                    "train_condition_dropout_ratio": [],
                    "train_time_contract_rho": [],
                }

                self.model.train()
                if self.ema_model is not None:
                    self.ema_model.eval()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch_from_collate in enumerate(tepoch):
                        batch = _postprocess_batch(dataset, batch_from_collate, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = _clone_sampling_batch(batch)

                        training_progress = self._training_progress(
                            epoch=self.epoch,
                            batch_idx=batch_idx,
                            num_epochs=cfg.training.num_epochs,
                            steps_per_epoch=steps_per_epoch,
                        )
                        teacher = self.ema_model if self.ema_model is not None else self.model
                        losses = self.model.compute_loss(
                            batch,
                            teacher_policy=teacher,
                            loss_cfg=cfg.loss,
                            training_progress=training_progress,
                        )
                        raw_loss = losses["loss_total"]
                        if not torch.isfinite(raw_loss):
                            raise FloatingPointError(
                                f"Non-finite training loss at epoch {self.epoch}, "
                                f"batch {batch_idx}: {raw_loss.item()}"
                            )
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if (
                            (batch_idx + 1)
                            % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            lr_scheduler.step()
                            if ema is not None:
                                ema.step(self.model)

                        train_values = {
                            "train_loss": float(raw_loss.item()),
                            "train_loss_main": float(losses["loss_main"].item()),
                            "train_loss_flow": float(losses["loss_flow"].item()),
                            "train_loss_self_consistency": float(
                                losses["loss_self_consistency"].item()
                            ),
                            "train_loss_self_guidance": float(
                                losses["loss_self_guidance"].item()
                            ),
                            "train_self_consistency_ratio": float(
                                losses["self_consistency_ratio"].item()
                            ),
                            "train_self_guidance_ratio": float(
                                losses["self_guidance_ratio"].item()
                            ),
                            "train_self_guidance_weight": float(
                                losses["self_guidance_weight"].item()
                            ),
                            "train_self_guidance_pred_norm": float(
                                losses["self_guidance_pred_norm"].item()
                            ),
                            "train_self_guidance_delta_norm": float(
                                losses["self_guidance_delta_norm"].item()
                            ),
                            "train_condition_dropout_ratio": float(
                                losses["condition_dropout_ratio"].item()
                            ),
                            "train_time_contract_rho": float(
                                losses["time_contract_rho"].item()
                            ),
                        }
                        for metric_name in OPTIONAL_LOSS_METRICS:
                            if metric_name in losses:
                                train_values[f"train_{metric_name}"] = float(
                                    losses[metric_name].item()
                                )
                        tepoch.set_postfix(loss=train_values["train_loss"], refresh=False)
                        for key, value in train_values.items():
                            train_running.setdefault(key, []).append(value)
                        step_log = {
                            **train_values,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and (
                            batch_idx >= (cfg.training.max_train_steps - 1)
                        ):
                            break

                for key, values in train_running.items():
                    step_log[key] = float(np.mean(values)) if len(values) > 0 else None
                json_logger.log(step_log)

                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()

                if (
                    env_runner is not None
                    and (self.epoch % cfg.training.rollout_every) == 0
                ):
                    step_log.update(env_runner.run(policy))

                if (self.epoch % cfg.training.val_every) == 0:
                    self.model.eval()
                    if self.ema_model is not None:
                        self.ema_model.eval()
                    with torch.no_grad():
                        val_running: dict[str, list[torch.Tensor]] = {
                            "val_loss": [],
                            "val_loss_main": [],
                            "val_loss_flow": [],
                            "val_loss_self_consistency": [],
                            "val_loss_self_guidance": [],
                            "val_self_consistency_ratio": [],
                            "val_self_guidance_ratio": [],
                            "val_self_guidance_weight": [],
                            "val_self_guidance_pred_norm": [],
                            "val_self_guidance_delta_norm": [],
                            "val_condition_dropout_ratio": [],
                            "val_time_contract_rho": [],
                        }

                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch_from_collate in enumerate(tepoch):
                                batch = _postprocess_batch(
                                    val_dataset,
                                    batch_from_collate,
                                    device,
                                )
                                teacher = self.ema_model if self.ema_model is not None else self.model
                                validation_progress = self._training_progress(
                                    epoch=self.epoch,
                                    batch_idx=batch_idx,
                                    num_epochs=cfg.training.num_epochs,
                                    steps_per_epoch=steps_per_epoch,
                                )
                                losses = self.model.compute_loss(
                                    batch,
                                    teacher_policy=teacher,
                                    loss_cfg=cfg.loss,
                                    training_progress=validation_progress,
                                )
                                val_running["val_loss"].append(losses["loss_total"])
                                val_running["val_loss_main"].append(losses["loss_main"])
                                val_running["val_loss_flow"].append(losses["loss_flow"])
                                val_running["val_loss_self_consistency"].append(
                                    losses["loss_self_consistency"]
                                )
                                val_running["val_loss_self_guidance"].append(
                                    losses["loss_self_guidance"]
                                )
                                val_running["val_self_consistency_ratio"].append(
                                    losses["self_consistency_ratio"]
                                )
                                val_running["val_self_guidance_ratio"].append(
                                    losses["self_guidance_ratio"]
                                )
                                val_running["val_self_guidance_weight"].append(
                                    losses["self_guidance_weight"]
                                )
                                val_running["val_self_guidance_pred_norm"].append(
                                    losses["self_guidance_pred_norm"]
                                )
                                val_running["val_self_guidance_delta_norm"].append(
                                    losses["self_guidance_delta_norm"]
                                )
                                val_running["val_condition_dropout_ratio"].append(
                                    losses["condition_dropout_ratio"]
                                )
                                val_running["val_time_contract_rho"].append(
                                    losses["time_contract_rho"]
                                )
                                for metric_name in OPTIONAL_LOSS_METRICS:
                                    if metric_name in losses:
                                        val_running.setdefault(
                                            f"val_{metric_name}",
                                            [],
                                        ).append(losses[metric_name])

                                if (cfg.training.max_val_steps is not None) and (
                                    batch_idx >= (cfg.training.max_val_steps - 1)
                                ):
                                    break

                        for key, values in val_running.items():
                            if len(values) > 0:
                                step_log[key] = float(torch.mean(torch.stack(values)).item())

                if (
                    (self.epoch % cfg.training.sample_every) == 0
                    and train_sampling_batch is not None
                ):
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        policy.reset()
                        result = policy.predict_action({"obs": batch["obs"]})
                        pred_action = result["action_pred"]
                        gt_action = batch["action"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = float(mse.item())
                        policy.reset()

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {
                        key.replace("/", "_"): value
                        for key, value in step_log.items()
                    }
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                self.model.train()
                if self.ema_model is not None:
                    self.ema_model.eval()
                if wandb_enabled:
                    wandb.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


__all__ = ["TrainOneStepFlowUnetLowdimWorkspace"]
