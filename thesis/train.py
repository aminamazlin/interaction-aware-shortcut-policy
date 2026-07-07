"""
Unified training entry point for all baselines.

1. Hydra loads the config YAML and composes defaults.
2. Look up the workspace for the requested model (via model_name or _target_).
3. Call workspace.train(cfg) — the baseline's full experiment loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from omegaconf import OmegaConf
import hydra


# Add local baselines and thesis to sys.path so they can be imported
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
# Add thesis directory so thesis.* packages can be imported
_repo_path = str(_REPO_ROOT)
_thesis_path = str(_REPO_ROOT / "thesis")
_baseline_paths = [
    str(_REPO_ROOT / "thesis" / "env"),
    str(_REPO_ROOT / "thesis" / "env" / "robomimic"),
    str(_REPO_ROOT / "thesis" / "env" / "robosuite"),
    str(_REPO_ROOT / "thesis" / "env" / "mimicgen"),
    str(_REPO_ROOT / "thesis" / "env" / "robocasa" / "robocasa"),
    str(_REPO_ROOT / "thesis" / "env" / "robocasa" / "robosuite"),
]


for _path in [_repo_path, _thesis_path, *_baseline_paths]:
    while _path in sys.path:
        sys.path.remove(_path)

for _path in reversed(_baseline_paths):
    sys.path.insert(0, _path)
sys.path.insert(0, _thesis_path)
sys.path.insert(0, _repo_path)

# Register custom resolvers before Hydra initializes
OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_workspace_class(cfg: OmegaConf):
    """Get the workspace class, supporting both model_name and _target_ patterns."""
    # If _target_ is present (native DP/upstream config style), use it directly
    if "_target_" in cfg:
        return hydra.utils.get_class(cfg._target_)

 
    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError(
            "Config must specify either '_target_' or 'model_name' at the top level."
        )

    from thesis.workspace import get_workspace
    return get_workspace(model_name)


@hydra.main(
    version_base=None,
    config_path=str(_THIS_FILE.parent / "config"),
)
def main(cfg: OmegaConf) -> None:
    # Resolve all interpolations after composition
    OmegaConf.resolve(cfg)

    # Get workspace class and instantiate
    WorkspaceClass = get_workspace_class(cfg)
    workspace = WorkspaceClass(cfg)

    model_name = cfg.get("model_name", cfg.get("_target_", "unknown"))
    print(f"[thesis] Training {model_name} — output: {cfg.get('output_dir', '.')}")
    workspace.run()
    print("[thesis] Training complete.")


if __name__ == "__main__":
    main()
