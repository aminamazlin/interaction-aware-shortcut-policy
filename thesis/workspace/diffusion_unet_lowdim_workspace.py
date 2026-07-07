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
from torch.utils.data._utils.collate import default_collate

from ..model.common.lr_scheduler import get_scheduler
from ..model.diffusion.ema_model import EMAModel
from ..policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from ..utils.checkpoint_util import TopKCheckpointManager
from ..utils.json_logger import JsonLogger
from ..utils.pytorch_util import dict_apply, optimizer_to
from .base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _postprocess_batch(dataset, batch, device: torch.device):
    if hasattr(dataset, "postprocess"):
        return dataset.postprocess(batch, device=device)
    return dict_apply(batch, lambda x: x.to(device, non_blocking=True))


def _clone_sampling_batch(batch, max_batch_size: int = 256):
    batch_size = next(iter(batch.values())).shape[0]
    sample_size = min(batch_size, max_batch_size)
    return dict_apply(batch, lambda x: x[:sample_size].detach().cpu().clone())


def _make_repeat_pad_collate(batch_size: int):
    def collate_fn(batch):
        if 0 < len(batch) < batch_size:
            missing = batch_size - len(batch)
            batch = batch + [batch[i % len(batch)] for i in range(missing)]
        return default_collate(batch)

    return collate_fn


def _make_dataloader(dataset, dataloader_cfg):
    kwargs = OmegaConf.to_container(dataloader_cfg, resolve=True)
    pad_last_batch = bool(kwargs.pop("pad_last_batch", False))
    if pad_last_batch:
        kwargs["drop_last"] = False
        kwargs["collate_fn"] = _make_repeat_pad_collate(int(kwargs["batch_size"]))
    return DataLoader(dataset, **kwargs)


class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    """Regular lowdim Diffusion Policy UNet workspace."""

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        if not isinstance(self.model, DiffusionUnetLowdimPolicy):
            raise TypeError(
                "TrainDiffusionUnetLowdimWorkspace requires "
                "DiffusionUnetLowdimPolicy, got "
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

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

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

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}
                train_losses = []

                self.model.train()
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

                        raw_loss = self.model.compute_loss(batch)
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

                        raw_loss_cpu = float(raw_loss.item())
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
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

                step_log["train_loss"] = float(np.mean(train_losses))

                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()

                if (
                    env_runner is not None
                    and (self.epoch % cfg.training.rollout_every) == 0
                ):
                    step_log.update(env_runner.run(policy))

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = []
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
                                val_losses.append(self.model.compute_loss(batch))
                                if (cfg.training.max_val_steps is not None) and (
                                    batch_idx >= (cfg.training.max_val_steps - 1)
                                ):
                                    break
                        if len(val_losses) > 0:
                            step_log["val_loss"] = float(
                                torch.mean(torch.stack(val_losses)).item()
                            )

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        result = policy.predict_action({"obs": batch["obs"]})
                        pred_action = result["action_pred"]
                        gt_action = batch["action"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = float(mse.item())

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

                policy.train()
                wandb.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
