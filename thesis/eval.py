"""
Evaluate the five best score-named checkpoints  for a run.

python thesis/eval.py \
    --run-dir /gpfs/work5/0/prjs2121/amazlin/bachelor-thesis/outputs/2026.05.30/13.33.08_train_csl_shortcut_longhorizon_unet_lowdim_coffee_preparation_d0 \
    --output_dir data/eval_outputs/csl_shortcut_2026_05_30_13_33_08
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


# Add local baselines and thesis to sys.path so checkpoint configs can import.
_THIS_FILE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
_PROJECT_STORAGE_ROOT = pathlib.Path(
    os.environ.get(
        "PROJECT_STORAGE_ROOT",
        "/gpfs/work5/0/prjs2121/amazlin/bachelor-thesis",
    )
).expanduser()
_repo_path = str(_REPO_ROOT)
_thesis_path = str(_REPO_ROOT / "thesis")
_baseline_paths = [
    str(_REPO_ROOT / "thesis" / "env" / "robomimic"),
    str(_REPO_ROOT / "thesis" / "env" / "robocasa" / "robocasa"),
    str(_REPO_ROOT / "thesis" / "env" / "robocasa" / "robosuite"),
    str(_REPO_ROOT / "baselines" / "diffusion-policy"),
    str(_REPO_ROOT / "baselines" / "consistency-policy"),
]

for _path in [_repo_path, _thesis_path, *_baseline_paths]:
    while _path in sys.path:
        sys.path.remove(_path)

for _path in reversed(_baseline_paths):
    sys.path.insert(0, _path)
sys.path.insert(0, _thesis_path)
sys.path.insert(0, _repo_path)

import click

_SCORE_CKPT_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)-(?P<monitor>test_mean_score|test_success_rate)="
    r"(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\.ckpt$"
)


@dataclass(frozen=True)
class CheckpointSpec:
    path: pathlib.Path
    label: str
    kind: str
    epoch: int | None = None
    monitor_key: str | None = None
    monitor_value: float | None = None
    test_mean_score: float | None = None
    test_success_rate: float | None = None


def get_workspace_class(cfg: OmegaConf):
    """Get workspace class, supporting both native DP and thesis configs."""
    import hydra

    if "_target_" in cfg:
        return hydra.utils.get_class(cfg._target_)

    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError(
            "Checkpoint config must specify either '_target_' or 'model_name'."
        )

    from thesis.workspace import get_workspace

    return get_workspace(model_name)


def resolve_serialized_config_paths(cfg):
    """Re-root paths saved in checkpoints so evaluation works off project storage."""
    from omegaconf import OmegaConf

    target_rewrites = {
        "env.franka_kitchen_runner.FrankaKitchenLowdimRunner":
            "env.runners.franka_kitchen_runner.FrankaKitchenLowdimRunner",
    }

    try:
        OmegaConf.set_readonly(cfg, False)
    except (AttributeError, TypeError):
        pass

    def resolve_value(value):
        if not isinstance(value, str):
            return value
        if value in target_rewrites:
            return target_rewrites[value]
        if value.startswith("${"):
            return value
        if "training_data/" in value:
            rel_path = value.split("training_data/", 1)[1]
            return str(_PROJECT_STORAGE_ROOT / "training_data" / rel_path)
        return value

    def walk(node):
        if OmegaConf.is_dict(node):
            for key in list(node.keys()):
                value = node[key]
                if OmegaConf.is_config(value):
                    walk(value)
                else:
                    node[key] = resolve_value(value)
        elif OmegaConf.is_list(node):
            for idx in range(len(node)):
                value = node[idx]
                if OmegaConf.is_config(value):
                    walk(value)
                else:
                    node[idx] = resolve_value(value)

    walk(cfg)
    return cfg


def resolve_checkpoint_dir(run_dir: str) -> tuple[pathlib.Path, pathlib.Path]:
    run_path = pathlib.Path(run_dir).expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run path does not exist: {run_path}")

    if run_path.is_dir() and run_path.name == "checkpoints":
        checkpoint_dir = run_path
        resolved_run_dir = run_path.parent
    else:
        resolved_run_dir = run_path
        checkpoint_dir = run_path / "checkpoints"

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find checkpoints directory at: {checkpoint_dir}"
        )
    return resolved_run_dir, checkpoint_dir


def parse_score_checkpoint(path: pathlib.Path) -> tuple[int, str, float] | None:
    match = _SCORE_CKPT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("epoch")), str(match.group("monitor")), float(match.group("score"))


def discover_checkpoints(checkpoint_dir: pathlib.Path, top_k: int = 5) -> list[CheckpointSpec]:
    scored = []
    for path in checkpoint_dir.glob("*.ckpt"):
        parsed = parse_score_checkpoint(path)
        if parsed is None:
            continue
        epoch, monitor, score = parsed
        scored.append((path, epoch, monitor, score))

    scored.sort(key=lambda item: (item[3], item[1]), reverse=True)
    selected = scored[:top_k]

    specs = [
        CheckpointSpec(
            path=path,
            label=f"top{rank}_{path.stem}",
            kind="top5",
            epoch=epoch,
            monitor_key=monitor,
            monitor_value=score,
            test_mean_score=score if monitor == "test_mean_score" else None,
            test_success_rate=score if monitor == "test_success_rate" else None,
        )
        for rank, (path, epoch, monitor, score) in enumerate(selected, start=1)
    ]

    latest_path = checkpoint_dir / "latest.ckpt"
    if latest_path.is_file():
        specs.append(CheckpointSpec(path=latest_path, label="latest", kind="latest"))

    if not specs:
        raise FileNotFoundError(
            f"No score-named checkpoints or latest.ckpt found in: {checkpoint_dir}"
        )
    return specs


def is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def to_jsonable(value: Any) -> Any:
    try:
        import wandb

        if isinstance(value, wandb.sdk.data_types.video.Video):
            return value._path
    except ImportError:
        pass
    if isinstance(value, pathlib.Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if is_json_scalar(value):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return repr(value)


def scalar_metrics(log_data: dict[str, Any]) -> dict[str, float | int | bool]:
    scalars = {}
    for key, value in log_data.items():
        try:
            import numpy as np

            if isinstance(value, np.generic):
                value = value.item()
        except ImportError:
            pass
        if isinstance(value, (bool, int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                scalars[key] = value
        elif isinstance(value, bool):
            scalars[key] = value
    return scalars


def aggregate_metrics(
    metrics_list: list[dict[str, float | int | bool]],
    prefix: str,
) -> dict[str, Any]:
    import numpy as np

    aggregate: dict[str, Any] = {f"{prefix}_num_checkpoints": len(metrics_list)}
    if not metrics_list:
        return aggregate

    keys = sorted(set().union(*(metrics.keys() for metrics in metrics_list)))
    for key in keys:
        values = [
            float(metrics[key])
            for metrics in metrics_list
            if key in metrics and isinstance(metrics[key], (bool, int, float))
        ]
        if not values:
            continue
        aggregate[f"{prefix}_mean/{key}"] = float(np.mean(values))
        aggregate[f"{prefix}_std/{key}"] = float(np.std(values))
    return aggregate


def checkpoint_metadata(spec: CheckpointSpec) -> dict[str, Any]:
    return {
        "path": str(spec.path),
        "label": spec.label,
        "kind": spec.kind,
        "epoch": spec.epoch,
        "monitor_key": spec.monitor_key,
        "monitor_value": spec.monitor_value,
        "test_mean_score": spec.test_mean_score,
        "test_success_rate": spec.test_success_rate,
    }


def load_policy_from_checkpoint(
    checkpoint_path: pathlib.Path,
    output_dir: pathlib.Path,
    device: torch.device,
):
    import hydra
    import dill
    import torch
    from omegaconf import OmegaConf

    from thesis.workspace.base_workspace import BaseWorkspace

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(
        checkpoint_path.open("rb"),
        map_location="cpu",
        pickle_module=dill,
    )
    if "cfg" not in payload and "config" in payload and "teacher_cfg" in payload:
        cfg = OmegaConf.create(payload["config"])
        teacher_cfg = OmegaConf.create(payload["teacher_cfg"])
        cfg = resolve_serialized_config_paths(cfg)
        teacher_cfg = resolve_serialized_config_paths(teacher_cfg)

        from thesis.policy.onedp_policy import OneStepDiffusionPolicy
        from thesis.workspace.onedp_workspace import OneDPWorkspace

        teacher_checkpoint = cfg.get("teacher_checkpoint", None)
        if not teacher_checkpoint:
            raise KeyError(
                f"OneDP checkpoint {checkpoint_path} has no teacher_checkpoint in config."
            )
        teacher_policy, teacher_cfg = OneDPWorkspace._load_teacher(
            teacher_checkpoint,
            device,
        )
        teacher_cfg = resolve_serialized_config_paths(teacher_cfg)

        noise_scheduler = None
        if "noise_scheduler" in cfg:
            noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

        policy = OneStepDiffusionPolicy(
            teacher_policy=teacher_policy,
            variant=cfg.variant,
            t_init=cfg.distillation.t_init,
            t_min=cfg.distillation.t_min,
            t_max=cfg.distillation.t_max,
            noise_scheduler=noise_scheduler,
        )
        policy.load_state_dict(payload["model"])
        policy.to(device)
        policy.eval()
        return teacher_cfg, policy

    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)
    cfg = resolve_serialized_config_paths(cfg)

    workspace_cls = get_workspace_class(cfg)
    workspace = workspace_cls(cfg, output_dir=str(output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.get("use_ema", False) and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model

    policy.to(device)
    policy.eval()
    return cfg, policy


def configure_test_only_runner(cfg) -> None:
    """Standalone evaluation should use held-out/test env initializations only."""
    env_runner_cfg = cfg.get("task", {}).get("env_runner", None)
    if env_runner_cfg is None:
        return
    if "n_train" in env_runner_cfg:
        env_runner_cfg.n_train = 0
    if "n_train_vis" in env_runner_cfg:
        env_runner_cfg.n_train_vis = 0

    task_cfg = cfg.get("task", {})
    task_name = str(
        env_runner_cfg.get(
            "task_name",
            task_cfg.get("task_name", task_cfg.get("name", cfg.get("task_name", ""))),
        )
    )
    old_max_steps = int(env_runner_cfg.get("max_steps", 0))
    if "CoffeePreparation" in task_name and old_max_steps < 1000:
        env_runner_cfg.max_steps = 1000

    print(
        "[eval] runner config: "
        f"task_name={task_name} "
        f"dataset_dir={env_runner_cfg.get('dataset_dir', 'unset')} "
        f"n_train={env_runner_cfg.get('n_train', 'unset')} "
        f"n_test={env_runner_cfg.get('n_test', 'unset')} "
        f"n_envs={env_runner_cfg.get('n_envs', 'unset')} "
        f"max_steps={env_runner_cfg.get('max_steps', 'unset')}"
    )


def run_checkpoint_eval(
    spec: CheckpointSpec,
    output_dir: pathlib.Path,
    device: torch.device,
    inference_cfg_scale: float = 1.0,
    num_inference_steps: int | None = None,
) -> dict[str, Any]:
    import hydra
    import wandb
    from omegaconf import OmegaConf

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, policy = load_policy_from_checkpoint(spec.path, output_dir, device)
    if num_inference_steps is not None:
        num_inference_steps = int(num_inference_steps)
        if not hasattr(policy, "num_inference_steps"):
            if num_inference_steps == 1 and type(policy).__name__ == "OneStepDiffusionPolicy":
                print(
                    "[eval] OneStepDiffusionPolicy is already one-step; "
                    "ignoring num_inference_steps=1"
                )
            else:
                raise ValueError(
                    "Selected policy does not expose num_inference_steps: "
                    f"{type(policy).__name__}"
                )
        else:
            policy.num_inference_steps = num_inference_steps
            if "policy" in cfg:
                OmegaConf.update(
                    cfg.policy,
                    "num_inference_steps",
                    num_inference_steps,
                    merge=False,
                    force_add=True,
                )
            print(f"[eval] using num_inference_steps={num_inference_steps}")
    if float(inference_cfg_scale) != 1.0:
        if not hasattr(policy, "inference_cfg_scale"):
            raise ValueError(
                "Selected policy does not support inference-time classifier-free "
                f"{type(policy).__name__}"
            )
        policy.inference_cfg_scale = float(inference_cfg_scale)
        if "policy" in cfg:
            OmegaConf.update(
                cfg.policy,
                "inference_cfg_scale",
                float(inference_cfg_scale),
                merge=False,
                force_add=True,
            )
        print(f"[eval] using inference_cfg_scale={float(inference_cfg_scale):.6g}")
    configure_test_only_runner(cfg)

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))
    wandb_run = wandb.init(
        project=cfg.logging.get("project", "thesis_eval"),
        name=f"eval_{spec.label}",
        dir=str(output_dir),
        mode=os.environ.get("WANDB_MODE", "disabled"),
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
    try:
        runner_log = env_runner.run(policy)
    finally:
        env = getattr(env_runner, "env", None)
        close = getattr(env, "close", None)
        if callable(close):
            close()
        if wandb_run is not None:
            wandb.finish()

    json_log = {key: to_jsonable(value) for key, value in runner_log.items()}
    json_log["checkpoint"] = checkpoint_metadata(spec)

    for filename in ("eval_log.json", "eval.json"):
        out_path = output_dir / filename
        with out_path.open("w") as f:
            json.dump(json_log, f, indent=2, sort_keys=True)
    return runner_log


@click.command()
@click.option(
    "-r",
    "--run-dir",
    required=True,
    help="Training run directory, or its nested checkpoints directory.",
)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--top-k", default=5, show_default=True, type=int)
@click.option("--inference-cfg-scale", default=1.0, show_default=True, type=float)
@click.option("--num-inference-steps", default=None, type=int)
def main(run_dir, output_dir, device, top_k, inference_cfg_scale, num_inference_steps):
    import torch

    if top_k < 1:
        raise click.BadParameter("--top-k must be at least 1")
    if inference_cfg_scale < 0.0:
        raise click.BadParameter("--inference-cfg-scale must be non-negative")
    if num_inference_steps is not None and num_inference_steps < 1:
        raise click.BadParameter("--num-inference-steps must be at least 1")

    output_path = pathlib.Path(output_dir)
    if output_path.exists():
        click.confirm(f"Output path {output_path} already exists! Overwrite?", abort=True)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_run_dir, checkpoint_dir = resolve_checkpoint_dir(run_dir)
    specs = discover_checkpoints(checkpoint_dir, top_k=top_k)
    device = torch.device(device)

    per_checkpoint: dict[str, dict[str, Any]] = {}
    top5_scalar_metrics = []
    all_scalar_metrics = []
    latest_scalar_metrics = None

    print(f"[eval] run_dir={resolved_run_dir}")
    print(f"[eval] checkpoint_dir={checkpoint_dir}")
    print("[eval] selected checkpoints:")
    for spec in specs:
        score_text = "latest"
        if spec.kind != "latest":
            score_text = f"{spec.monitor_key}={spec.monitor_value:.6g}"
        print(f"  - {spec.label}: {score_text} ({spec.path})")

    for spec in specs:
        ckpt_output_dir = output_path / spec.label
        print(f"[eval] evaluating {spec.label}")
        runner_log = run_checkpoint_eval(
            spec,
            ckpt_output_dir,
            device,
            inference_cfg_scale=float(inference_cfg_scale),
            num_inference_steps=num_inference_steps,
        )
        scalars = scalar_metrics(runner_log)
        per_checkpoint[spec.label] = {
            "checkpoint": checkpoint_metadata(spec),
            "metrics": scalars,
        }
        all_scalar_metrics.append(scalars)
        if spec.kind == "top5":
            top5_scalar_metrics.append(scalars)
        elif spec.kind == "latest":
            latest_scalar_metrics = scalars

    aggregate = {
        "run_dir": str(resolved_run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "inference_cfg_scale": float(inference_cfg_scale),
        "num_inference_steps": num_inference_steps,
        "top5_checkpoints": [
            checkpoint_metadata(spec) for spec in specs if spec.kind == "top5"
        ],
        "latest_checkpoint": next(
            (checkpoint_metadata(spec) for spec in specs if spec.kind == "latest"),
            None,
        ),
        "per_checkpoint": per_checkpoint,
    }
    aggregate.update(aggregate_metrics(all_scalar_metrics, prefix="all"))
    aggregate.update(aggregate_metrics(top5_scalar_metrics, prefix="top5"))
    if latest_scalar_metrics is not None:
        aggregate.update(
            {f"latest/{key}": value for key, value in latest_scalar_metrics.items()}
        )

    aggregate_json = to_jsonable(aggregate)
    for filename in ("eval_log.json", "mean_eval.json"):
        out_path = output_path / filename
        with out_path.open("w") as f:
            json.dump(aggregate_json, f, indent=2, sort_keys=True)
    print(f"[eval] wrote aggregate log to {output_path / 'eval_log.json'}")


if __name__ == "__main__":
    main()
