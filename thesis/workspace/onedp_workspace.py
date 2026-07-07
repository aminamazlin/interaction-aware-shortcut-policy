from __future__ import annotations

import copy
import json
import os
import pathlib
import random
import time
from typing import Any, Dict
import tqdm
import dill
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from thesis.policy.onedp_policy import OneStepDiffusionPolicy
from thesis.utils.checkpoint_util import TopKCheckpointManager


def _make_repeat_pad_collate(batch_size: int):
    def collate_fn(batch):
        if 0 < len(batch) < batch_size:
            missing = batch_size - len(batch)
            batch = batch + [batch[i % len(batch)] for i in range(missing)]
        return default_collate(batch)

    return collate_fn


def _get_int(config, key: str, default: int) -> int:
    value = config.get(key, None)
    if value is None:
        value = default
    return int(value)


def _check_finite_loss(loss: torch.Tensor, name: str, epoch: int, batch_idx: int) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite {name} at epoch {epoch}, batch {batch_idx}: "
            f"{float(loss.detach().cpu())}"
        )


def _prepare_gradients(module: torch.nn.Module, label: str) -> None:
    for name, param in module.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError(
                "Non-finite gradient before optimizer.step: "
                f"module={label}, name={name}, shape={tuple(grad.shape)}, "
                f"dtype={grad.dtype}, device={grad.device}, "
                f"contiguous={grad.is_contiguous()}"
            )
        if not grad.is_contiguous():
            param.grad = grad.contiguous()


class SeedWorker:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def __call__(self, worker_id: int):
        worker_seed = self.seed + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


class OneDPWorkspace:
    """Hydra-compatible workspace for One-Step Diffusion Policy distillation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = pathlib.Path(cfg.output_dir)
        self.repo_root = pathlib.Path(__file__).resolve().parents[2]
        self.project_storage_root = pathlib.Path(
            os.environ.get(
                "PROJECT_STORAGE_ROOT",
                "/gpfs/work5/0/prjs2121/amazlin/bachelor-thesis",
            )
        )

    def run(self):
        self.train(self.cfg)

    def train(self, cfg: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device(cfg.training.device)
        seed = int(cfg.training.get("seed", 42))
        self._set_seed(seed=seed)
        
        # init wandb run
        run = self._init_wandb(cfg)

        teacher_policy, teacher_cfg = self._load_teacher(cfg.teacher_checkpoint, device)
        teacher_cfg = self._resolve_relative_paths(copy.deepcopy(teacher_cfg))
        
        # configure train dataset & loader
        dataset = hydra.utils.instantiate(teacher_cfg.task.dataset)
        dataloader_cfg = self._merge_dataloader_cfg(
            teacher_cfg.dataloader,
            cfg.get("dataloader", None),
        )
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)
        dataloader = DataLoader(
            dataset,
            worker_init_fn=self._make_worker_init_fn(seed),
            generator=train_generator,
            **dataloader_cfg,
        )

        # configure val dataset & loader
        val_dataloader = None
        if hasattr(dataset, "get_validation_dataset"):
            val_dataset = dataset.get_validation_dataset()
            if val_dataset is not None:
                val_dataloader_cfg = copy.deepcopy(
                    teacher_cfg.val_dataloader
                    if "val_dataloader" in teacher_cfg
                    else teacher_cfg.dataloader
                )
                if "val_dataloader" not in teacher_cfg:
                    val_dataloader_cfg["shuffle"] = False
                    val_dataloader_cfg["drop_last"] = False
                    if "sampler" in val_dataloader_cfg:
                        val_dataloader_cfg["sampler"] = None
                    if "batch_sampler" in val_dataloader_cfg:
                        val_dataloader_cfg["batch_sampler"] = None

                val_dataloader_cfg = self._merge_dataloader_cfg(
                    val_dataloader_cfg,
                    cfg.get("val_dataloader", None),
                )
                val_generator = torch.Generator()
                val_generator.manual_seed(seed + 1)
                val_dataloader = DataLoader(
                    val_dataset,
                    worker_init_fn=self._make_worker_init_fn(seed + 1),
                    generator=val_generator,
                    **val_dataloader_cfg,
                )

        noise_scheduler = None
        if "noise_scheduler" in cfg:
            noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

        # Init policies
        policy = OneStepDiffusionPolicy(
            teacher_policy=teacher_policy,
            variant=cfg.variant,
            t_init=cfg.distillation.t_init,
            t_min=cfg.distillation.t_min,
            t_max=cfg.distillation.t_max,
            noise_scheduler=noise_scheduler,
        ).to(device)
        print(f"Using OneDP noise scheduler: {policy.schedule.name}")
        if run is not None:
            wandb.config.update({"onedp_noise_scheduler": policy.schedule.name}, allow_val_change=True)

        # Set student optimizer
        gen_optimizer = torch.optim.Adam(
            policy.generator_parameters(),
            lr=cfg.optimizer.generator_lr,
            betas=tuple(cfg.optimizer.betas),
        )
        
        # Set score network optimizer
        score_optimizer = None
        if cfg.variant == "stochastic":
            score_optimizer = torch.optim.Adam(
                policy.score_parameters(),
                lr=cfg.optimizer.score_lr,
                betas=tuple(cfg.optimizer.betas),
            )

        start_epoch = 0
        resume_checkpoint = cfg.training.get("resume_from_checkpoint", None)
        if resume_checkpoint:
            start_epoch = self._load_checkpoint(
                checkpoint_path=resume_checkpoint,
                policy=policy,
                gen_optimizer=gen_optimizer,
                score_optimizer=score_optimizer,
                device=device,
            ) + 1

        topk_manager = None
        if "checkpoint" in cfg and "topk" in cfg.checkpoint:
            topk_manager = TopKCheckpointManager(
                save_dir=str(self.output_dir / "checkpoints"),
                **OmegaConf.to_container(cfg.checkpoint.topk, resolve=True),
            )

        inference_batch = None
        teacher_inference_logged = False
        inference_time_every = cfg.training.get("inference_time_every", None)
        inference_time_warmup_steps = _get_int(cfg.training, "inference_time_warmup_steps", 5)
        inference_time_steps = _get_int(cfg.training, "inference_time_steps", 20)
        cuda_debug = bool(cfg.training.get("cuda_debug", False))

        def sync_cuda(stage: str) -> None:
            if cuda_debug and device.type == "cuda":
                try:
                    torch.cuda.synchronize(device)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"CUDA debug synchronize failed after {stage}."
                    ) from exc

        log_path = self.output_dir / "logs.json.txt"
        num_epochs = int(cfg.training.num_epochs)
        with tqdm.tqdm(total=max(num_epochs - start_epoch, 0), desc="Training") as pbar:
            for epoch in range(start_epoch, num_epochs):
                # put generator and score networks in training mode, teacher in eval mode
                policy.train()
                stats = {"loss_generator": 0.0, "loss_score": 0.0}
                n_steps = 0
                max_train_steps = cfg.training.get("max_train_steps", None)
                for batch_idx, batch in enumerate(dataloader):
                    batch = self._move_to_device(batch, device)
                    sync_cuda("move batch to device")

                    if score_optimizer is not None:
                        # clear gradients stored on score model params
                        score_optimizer.zero_grad(set_to_none=True)
                        
                        # compute score loss
                        score_losses = policy.compute_loss(batch)
                        sync_cuda("score compute_loss")
                        _check_finite_loss(
                            score_losses["loss_score"],
                            "score loss",
                            epoch,
                            batch_idx,
                        )
                        
                        # Compute gradients
                        score_losses["loss_score"].backward()
                        sync_cuda("score backward")
                        # limits how large score grads can get during training
                        torch.nn.utils.clip_grad_norm_(policy.score_parameters(), cfg.optimizer.max_grad_norm)
                        sync_cuda("score clip_grad_norm")
                        _prepare_gradients(policy.score_model, "score_model")
                        if policy.score_obs_encoder is not None:
                            _prepare_gradients(policy.score_obs_encoder, "score_obs_encoder")
                        
                        # update score network weights
                        sync_cuda("before score optimizer.step")
                        score_optimizer.step()
                        sync_cuda("score optimizer.step")
                        stats["loss_score"] += float(score_losses["loss_score"].detach().cpu())
                    else:
                        score_losses = None

                    # clear gradients stored on generator params
                    gen_optimizer.zero_grad(set_to_none=True)
                    
                    # compute generator loss
                    gen_losses = policy.compute_loss(batch)
                    sync_cuda("generator compute_loss")
                    _check_finite_loss(
                        gen_losses["loss_generator"],
                        "generator loss",
                        epoch,
                        batch_idx,
                    )
                    # compute gradients
                    gen_losses["loss_generator"].backward()
                    sync_cuda("generator backward")
                    
                    # limits how large generator grads can get during training
                    torch.nn.utils.clip_grad_norm_(policy.generator_parameters(), cfg.optimizer.max_grad_norm)
                    sync_cuda("generator clip_grad_norm")
                    _prepare_gradients(policy.generator_model, "generator_model")
                    if policy.generator_obs_encoder is not None:
                        _prepare_gradients(policy.generator_obs_encoder, "generator_obs_encoder")
                    
                    # update generator weights
                    sync_cuda("before generator optimizer.step")
                    gen_optimizer.step()
                    sync_cuda("generator optimizer.step")

                    stats["loss_generator"] += float(gen_losses["loss_generator"].detach().cpu())
                    if score_losses is None:
                        stats["loss_score"] += float(gen_losses["loss_score"].detach().cpu())
                        
                    n_steps += 1
                    if max_train_steps is not None and n_steps >= int(max_train_steps):
                        break

                # compute average losses for the epoch and log them
                row = {
                    "epoch": epoch,
                    "loss_generator": stats["loss_generator"] / max(n_steps, 1),
                    "loss_score": stats["loss_score"] / max(n_steps, 1),
                }

                # check if its time for validation
                val_every = cfg.training.get("val_every", None)
                if val_dataloader is not None and val_every and (epoch % int(val_every) == 0):
                    row.update(self._validate(policy, val_dataloader))

                # check if its time for eval (sim rollout)
                if cfg.training.eval_every and (epoch % int(cfg.training.eval_every) == 0):
                    row.update(self._evaluate(policy, teacher_cfg.task, epoch))

                if inference_time_every and (epoch % int(inference_time_every) == 0):
                    if inference_batch is None:
                        inference_source = val_dataloader if val_dataloader is not None else dataloader
                        inference_batch = self._move_to_device(next(iter(inference_source)), device)
                    row.update(self._benchmark_inference(
                        policy=policy,
                        batch=inference_batch,
                        prefix="student",
                        warmup_steps=inference_time_warmup_steps,
                        timing_steps=inference_time_steps,
                    ))
                    if not teacher_inference_logged:
                        row.update(self._benchmark_inference(
                            policy=teacher_policy,
                            batch=inference_batch,
                            prefix="teacher",
                            warmup_steps=inference_time_warmup_steps,
                            timing_steps=inference_time_steps,
                        ))
                        teacher_inference_logged = True
                        teacher_time = row.get("teacher/inference_time_sec")
                        student_time = row.get("student/inference_time_sec")
                        if teacher_time is not None and student_time:
                            row["student/inference_speedup_vs_teacher"] = teacher_time / student_time

                json_row = self._json_safe(row)
                with log_path.open("a") as f:
                    f.write(json.dumps(json_row, sort_keys=True) + "\n")
                print(json_row)
                if run is not None:
                    wandb.log(row, step=epoch)

                if topk_manager is not None:
                    topk_data = dict(row)
                    topk_data.update({
                        key.replace("/", "_"): value
                        for key, value in row.items()
                        if "/" in key
                    })
                    if topk_manager.monitor_key in topk_data:
                        topk_path = topk_manager.get_ckpt_path(topk_data)
                        if topk_path is not None:
                            self.save_checkpoint(
                                policy,
                                epoch,
                                teacher_cfg,
                                cfg,
                                gen_optimizer=gen_optimizer,
                                score_optimizer=score_optimizer,
                                path=topk_path,
                            )

                # Optional periodic checkpointing. When save_every is unset,
                # keep only top-k checkpoints plus the final latest checkpoint.
                save_every = cfg.training.get("save_every", None)
                if save_every and (epoch % int(save_every) == 0):
                    self.save_checkpoint(
                        policy,
                        epoch,
                        teacher_cfg,
                        cfg,
                        gen_optimizer=gen_optimizer,
                        score_optimizer=score_optimizer,
                        tag=f"epoch={epoch:04d}",
                    )

                pbar.update(1)

            self.save_checkpoint(
                policy,
                num_epochs - 1,
                teacher_cfg,
                cfg,
                gen_optimizer=gen_optimizer,
                score_optimizer=score_optimizer,
                tag="latest",
            )
            if run is not None:
                wandb.finish()


########### Helper functions #############################
    def evaluate(self, cfg: Any) -> Dict[str, float]:
        raise NotImplementedError("Use the saved OneDP checkpoint with the teacher env_runner for evaluation.")



    @torch.no_grad()
    def _validate(self, policy: OneStepDiffusionPolicy, val_dataloader) -> Dict[str, float]:
        was_training = policy.training
        
        # put models in eval mode for validation
        policy.eval()
        stats = {
            "val_loss_generator": 0.0,
            "val_loss_score": 0.0,
            "val_loss_total": 0.0,
        }
        n_steps = 0
        device = policy.device

        for batch in val_dataloader:
            batch = self._move_to_device(batch, device)
            # compute losses but dont backprop or update weights during validation
            losses = policy.compute_loss(batch)
            stats["val_loss_generator"] += float(losses["loss_generator"].detach().cpu())
            stats["val_loss_score"] += float(losses["loss_score"].detach().cpu())
            stats["val_loss_total"] += float(losses["loss_total"].detach().cpu())
            n_steps += 1

        # restore training mode, if it was training before
        if was_training:
            policy.train()

        # compute average validation losses and return them
        denom = max(n_steps, 1)
        return {
            "val_loss_generator": stats["val_loss_generator"] / denom,
            "val_loss_score": stats["val_loss_score"] / denom,
            "val_loss_total": stats["val_loss_total"] / denom,
        }

    def _evaluate(self, policy: OneStepDiffusionPolicy, task_cfg, epoch: int) -> Dict[str, Any]:
        # create eval dir
        eval_dir = self.output_dir / "eval" / f"epoch_{epoch:04d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "media").mkdir(parents=True, exist_ok=True)
        
        # set env runner
        env_runner = hydra.utils.instantiate(task_cfg.env_runner, output_dir=str(eval_dir))
        # check if it was in training mode
        was_training = policy.training
        
        # put models in eval mode
        policy.eval()
        
        # run sim rollout
        runner_log = env_runner.run(policy)
        
        # restore training mode, if it was training before
        if was_training:
            policy.train()
        return dict(runner_log)

    @torch.no_grad()
    def _benchmark_inference(
        self,
        policy,
        batch: Dict[str, Any],
        prefix: str,
        warmup_steps: int,
        timing_steps: int,
    ) -> Dict[str, float]:
        was_training = policy.training
        policy.eval()

        obs_dict = batch["obs"] if isinstance(batch["obs"], dict) else {"obs": batch["obs"]}
        value = next(iter(obs_dict.values())) if isinstance(obs_dict, dict) else obs_dict
        batch_size = int(value.shape[0])

        if hasattr(policy, "reset"):
            policy.reset()

        for _ in range(max(warmup_steps, 0)):
            policy.predict_action(obs_dict)

        self._sync_cuda(policy.device)
        elapsed = []
        for _ in range(max(timing_steps, 1)):
            start_time = time.perf_counter()
            result = policy.predict_action(obs_dict)
            self._sync_cuda(policy.device)
            elapsed.append(time.perf_counter() - start_time)

        if was_training:
            policy.train()

        elapsed = np.asarray(elapsed, dtype=np.float64)
        action = result["action"]
        n_action_steps = int(action.shape[1]) if action.ndim >= 3 else 1
        mean_sec = float(np.mean(elapsed))
        return {
            f"{prefix}/inference_time_sec": mean_sec,
            f"{prefix}/inference_time_ms": mean_sec * 1000.0,
            f"{prefix}/inference_time_std_ms": float(np.std(elapsed) * 1000.0),
            f"{prefix}/inference_time_per_action_ms": (
                mean_sec * 1000.0 / max(n_action_steps, 1)
            ),
            f"{prefix}/inference_batch_size": batch_size,
            f"{prefix}/inference_action_steps": n_action_steps,
        }

    @staticmethod
    def _sync_cuda(device):
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    @staticmethod
    def _move_to_device(value, device):
        if isinstance(value, dict):
            return {k: OneDPWorkspace._move_to_device(v, device) for k, v in value.items()}
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        return value

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Using OneDP training seed: {seed}")

    @staticmethod
    def _make_worker_init_fn(seed: int):
        return SeedWorker(seed)

    @staticmethod
    def _merge_dataloader_cfg(base_cfg, override_cfg=None):
        loader_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        if override_cfg is not None:
            loader_cfg = OmegaConf.merge(loader_cfg, override_cfg)

        loader_dict = OmegaConf.to_container(loader_cfg, resolve=True)
        pad_last_batch = bool(loader_dict.pop("pad_last_batch", False))
        if pad_last_batch:
            loader_dict["drop_last"] = False
            loader_dict["collate_fn"] = _make_repeat_pad_collate(int(loader_dict["batch_size"]))
        if int(loader_dict.get("num_workers", 0)) == 0:
            loader_dict.pop("persistent_workers", None)
            loader_dict.pop("multiprocessing_context", None)
        return loader_dict

    @staticmethod
    def _load_teacher(checkpoint_path: str, device):
        # load checkpoint payload with dill to support loading objects like env_runners that may be part of the teacher workspace
        payload = torch.load(open(checkpoint_path, "rb"), map_location="cpu", pickle_module=dill)
        # get orig training config saved inside the ckpt
        cfg = payload["cfg"]
        # get teacher workspace class 
        cls = hydra.utils.get_class(cfg._target_)
        try:
            # initialize teacher workspace
            workspace = cls(cfg, output_dir=None)
        except TypeError:
            workspace = cls(cfg)
        
        # load teacher model and other objects into the workspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        # chooses which model to use as the teacher (ema or regular)
        teacher = workspace.ema_model if cfg.training.use_ema else workspace.model
        # put teacher in eval mode
        teacher.to(device).eval()
        
        # freeze teacher params
        for param in teacher.parameters():
            param.requires_grad_(False)
        return teacher, cfg

    def save_checkpoint(
        self,
        policy: OneStepDiffusionPolicy,
        epoch: int,
        teacher_cfg,
        cfg,
        gen_optimizer=None,
        score_optimizer=None,
        tag: str | None = None,
        path: str | os.PathLike | None = None,
    ):
        if path is None:
            ckpt_dir = self.output_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"{tag}.ckpt"
        else:
            path = pathlib.Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model": policy.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "teacher_cfg": OmegaConf.to_container(teacher_cfg, resolve=True),
        }
        if gen_optimizer is not None:
            payload["gen_optimizer"] = gen_optimizer.state_dict()
        if score_optimizer is not None:
            payload["score_optimizer"] = score_optimizer.state_dict()

        torch.save(payload, path)
        print(f"Saved {path}")

    def _load_checkpoint(
        self,
        checkpoint_path: str | os.PathLike,
        policy: OneStepDiffusionPolicy,
        gen_optimizer,
        score_optimizer,
        device,
    ) -> int:
        checkpoint_path = pathlib.Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.output_dir / checkpoint_path
        payload = torch.load(
            open(checkpoint_path, "rb"),
            map_location=device,
            pickle_module=dill,
        )

        policy.load_state_dict(payload["model"])
        if "gen_optimizer" in payload:
            gen_optimizer.load_state_dict(payload["gen_optimizer"])
            self._optimizer_to(gen_optimizer, device)
        else:
            print(f"Resume checkpoint {checkpoint_path} has no generator optimizer state.")

        if score_optimizer is not None:
            if "score_optimizer" in payload:
                score_optimizer.load_state_dict(payload["score_optimizer"])
                self._optimizer_to(score_optimizer, device)
            else:
                print(f"Resume checkpoint {checkpoint_path} has no score optimizer state.")

        epoch = int(payload.get("epoch", -1))
        print(f"Resumed OneDP checkpoint {checkpoint_path} at epoch {epoch}.")
        return epoch

    @staticmethod
    def _optimizer_to(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _init_wandb(self, cfg: Any):
        logging_cfg = cfg.get("logging", None)
        if logging_cfg is None:
            return None

        mode = logging_cfg.get("mode", "online")
        if mode in (None, "disabled", "off"):
            return None

        config = OmegaConf.to_container(cfg, resolve=True)
        return wandb.init(
            project=logging_cfg.get("project", "thesis_tasks"),
            name=logging_cfg.get("name", None),
            tags=logging_cfg.get("tags", None),
            group=logging_cfg.get("group", None),
            id=logging_cfg.get("id", None),
            resume=logging_cfg.get("resume", False),
            mode=mode,
            dir=str(self.output_dir),
            config=config,
        )

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {k: OneDPWorkspace._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [OneDPWorkspace._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [OneDPWorkspace._json_safe(v) for v in value]
        if isinstance(value, pathlib.Path):
            return str(value)
        if type(value).__module__.startswith("wandb."):
            media_path = getattr(value, "_path", None) or getattr(value, "path", None)
            return str(media_path) if media_path is not None else repr(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

    def _resolve_relative_paths(self, cfg):
        target_rewrites = {
            "env.franka_kitchen_runner.FrankaKitchenLowdimRunner":
                "env.runners.franka_kitchen_runner.FrankaKitchenLowdimRunner",
        }

        def maybe_resolve(value):
            if not isinstance(value, str):
                return value
            if value.startswith("${"):
                return value
            if value in target_rewrites:
                return target_rewrites[value]
            if "training_data/" in value:
                rel_path = value.split("training_data/", 1)[1]
                return str(self.project_storage_root / "training_data" / rel_path)
            path = pathlib.Path(value)
            if path.is_absolute():
                return value
            candidate = self.repo_root / path
            if candidate.exists():
                return str(candidate)
            return value

        def walk(node):
            if OmegaConf.is_dict(node):
                for key in list(node.keys()):
                    value = node[key]
                    if isinstance(value, str) and (
                        key == "_target_" or key.endswith("_path") or key.endswith("_dir")
                    ):
                        node[key] = maybe_resolve(value)
                    else:
                        walk(value)
            elif OmegaConf.is_list(node):
                for value in node:
                    walk(value)

        walk(cfg)
        self._patch_teacher_eval_runner(cfg)
        return cfg

    @staticmethod
    def _patch_teacher_eval_runner(cfg):
        """Apply OneDP eval safety overrides to teacher checkpoint configs."""
        try:
            env_runner = cfg.task.env_runner
        except Exception:
            return

        target = env_runner.get("_target_", None)
        if target == "env.runners.franka_kitchen_runner.FrankaKitchenLowdimRunner":
            OmegaConf.update(env_runner, "n_envs", None, merge=False, force_add=True)
            OmegaConf.update(env_runner, "use_async_env", True, merge=False, force_add=True)
            OmegaConf.update(env_runner, "vector_env_context", "spawn", merge=False, force_add=True)
