#!/bin/bash
#SBATCH --job-name=ia-pickplace
#SBATCH --output=/home/amazlin/bachelor-thesis/logs/slurm/%x_%j.out
#SBATCH --error=/home/amazlin/bachelor-thesis/logs/slurm/%x_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50G
#SBATCH --time=48:00:00

set -e
set -o pipefail

# Usage:
#   sbatch thesis/scripts/train_oneflow_longhorizon_unet_coffee.slurm
#   CONFIG_NAME=train_oneflow_longhorizon_unet_lowdim_coffee sbatch thesis/scripts/train_oneflow_longhorizon_unet_coffee.slurm
#   sbatch thesis/scripts/train_oneflow_longhorizon_unet_coffee.slurm training.num_epochs=500 logging.mode=offline
#   sbatch --export=ALL,CUDA_LAUNCH_BLOCKING=1,WANDB_MODE=offline thesis/train_longhorizon.sh training.cuda_debug=true training.max_train_steps=1 training.max_val_steps=0 training.rollout_every=null
#
# Defaults:
#   CONFIG_NAME=train_cumulative_one_step_flow_unet_image
#   CONDA_ENV_NAME=auto
#   WANDB_MODE=online

# --- 1. Path Discovery ---
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
    REPO_ROOT=$(pwd | sed 's|/bachelor-thesis.*|/bachelor-thesis|')
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT=$(echo "${SCRIPT_DIR}" | sed 's|/bachelor-thesis.*|/bachelor-thesis|')
fi

cd "${REPO_ROOT}"
mkdir -p logs/slurm thesis/logs/slurm

# --- 2. Environment Activation ---
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
SCRIPT_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config-name=*)
      export CONFIG_NAME="${1#--config-name=}"
      shift
      ;;
    --config-name)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --config-name requires a value" >&2
        exit 2
      fi
      export CONFIG_NAME="$2"
      shift 2
      ;;
    CONFIG_NAME=*)
      export CONFIG_NAME="${1#CONFIG_NAME=}"
      shift
      ;;
    *)
      SCRIPT_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${SCRIPT_ARGS[@]}"
export CONFIG_NAME="${CONFIG_NAME:-train_interaction_aware_shortcut_policy.yaml}"
CONFIG_BASENAME="$(basename "${CONFIG_NAME%.yaml}")"
HYDRA_CONFIG_ARGS=()
if [ -n "${CONFIG_PATH:-}" ]; then
  HYDRA_CONFIG_ARGS+=(--config-path "${CONFIG_PATH}")
fi

#  heavy run artifacts off the home quota. This directory stores Hydra
# outputs, checkpoints, rollout media, W&B run files, and W&B caches.
export PROJECT_STORAGE_ROOT="${PROJECT_STORAGE_ROOT:-/gpfs/work5/0/prjs2121/amazlin/bachelor-thesis}"
RUN_DATE="$(date +%Y.%m.%d)"
RUN_TIME="$(date +%H.%M.%S)"
TASK_CONFIG_OVERRIDE=""
for override in "$@"; do
  case "${override}" in
    task=*|+task=*)
      TASK_CONFIG_OVERRIDE="${override#*=}"
      ;;
  esac
done
CONFIG_TASK_NAME="$(
  "${PYTHON_BIN:-python}" - "${REPO_ROOT}" "${CONFIG_BASENAME}" "${TASK_CONFIG_OVERRIDE}" <<'PY' 2>/dev/null || true
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
config_name = sys.argv[2]
task_override = sys.argv[3] if len(sys.argv) > 3 else ""
config_path = repo_root / "thesis" / "config" / f"{config_name}.yaml"
task_config = task_override or None

if task_config is None and config_path.exists():
    lines = config_path.read_text().splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "- task: coffee_prep_lowdim":
            task_config = "coffee_prep_lowdim"
            break
        if stripped.startswith("- task: "):
            task_config = stripped.split(":", 1)[1].strip()
            break

if task_config:
    task_path = repo_root / "thesis" / "config" / "task" / f"{task_config}.yaml"
    if task_path.exists():
        for line in task_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("name:"):
                print(stripped.split(":", 1)[1].strip())
                raise SystemExit
PY
)"
CONFIG_TASK_NAME="${CONFIG_TASK_NAME:-run}"
if [ -z "${OUTPUT_DIR:-}" ]; then
  export OUTPUT_DIR="${PROJECT_STORAGE_ROOT}/outputs/${RUN_DATE}/${RUN_TIME}_${CONFIG_BASENAME}_${CONFIG_TASK_NAME}"
fi
mkdir -p "${OUTPUT_DIR}" "${PROJECT_STORAGE_ROOT}/wandb"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_STORAGE_ROOT}/wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_STORAGE_ROOT}/wandb/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${PROJECT_STORAGE_ROOT}/wandb/data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${PROJECT_STORAGE_ROOT}/wandb/artifacts}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PROJECT_STORAGE_ROOT}/numba_cache}"
mkdir -p \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${WANDB_ARTIFACT_DIR}" \
  "${NUMBA_CACHE_DIR}"

if [ -z "${CONDA_ENV_NAME:-}" ] || [ "${CONDA_ENV_NAME}" = "auto" ]; then
  if [[ "${CONFIG_BASENAME}" == *kitchen* ]] \
    || [[ "${CONFIG_TASK_NAME}" == franka_kitchen* ]] \
    || [[ "${CONFIG_TASK_NAME}" == *kitchen* ]] \
    || [[ "${CONFIG_TASK_NAME}" == mimicgen_pick_place* ]]; then
    export CONDA_ENV_NAME="flow_policy_kitchen"
  elif [[ "${CONFIG_BASENAME}" == *robocasa* ]] \
    || [[ "${CONFIG_TASK_NAME}" == load_dishwasher* ]] \
    || [[ "${CONFIG_TASK_NAME}" == ldw_* ]] \
    || [[ "${CONFIG_TASK_NAME}" == pack_identical_lunches* ]] \
    || [[ "${CONFIG_TASK_NAME}" == *robocasa* ]]; then
    export CONDA_ENV_NAME="robocasa"
  else
    export CONDA_ENV_NAME="flow_policy"
  fi
else
  export CONDA_ENV_NAME
fi

if command -v module >/dev/null 2>&1; then
  module load miniconda >/dev/null 2>&1 || true
fi

for conda_sh in \
  "${HOME}/miniforge3/etc/profile.d/conda.sh" \
  "/gpfs/home1/amazlin/miniforge3/etc/profile.d/conda.sh" \
  "/home/amazlin/miniforge3/etc/profile.d/conda.sh"
do
  if [ -f "${conda_sh}" ]; then
    # shellcheck disable=SC1090
    source "${conda_sh}"
    break
  fi
done

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}" || {
    echo "ERROR: failed to activate conda env '${CONDA_ENV_NAME}'." >&2
    exit 1
  }
else
  echo "ERROR: conda not found in PATH and no usable Miniforge / Miniconda setup was found." >&2
  exit 1
fi

CONDA_ENV_PYTHON="${CONDA_PREFIX}/bin/python"
if [ -x "${CONDA_ENV_PYTHON}" ]; then
  PYTHON_BIN="${CONDA_ENV_PYTHON}"
else
  PYTHON_BIN="$(command -v python)"
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

# --- 3. Runtime Setup ---
unset DISPLAY
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
unset OMNIGIBSON_HEADLESS

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/thesis:${REPO_ROOT}/thesis/env:${REPO_ROOT}/thesis/env/robomimic:${REPO_ROOT}/thesis/env/robosuite:${REPO_ROOT}/thesis/env/mimicgen:${REPO_ROOT}/thesis/env/robocasa/robocasa:${REPO_ROOT}/thesis/env/robocasa/robosuite:${PYTHONPATH:-}"

NVIDIA_LIB_DIR=""
for candidate in /usr/local/nvidia/lib64 /usr/lib/nvidia-* /usr/lib/x86_64-linux-gnu /usr/lib64; do
  if [ -e "${candidate}/libEGL.so.1" ] && [ -e "${candidate}/libOpenGL.so.0" ]; then
    NVIDIA_LIB_DIR="${candidate}"
    break
  fi
done

if [ -n "${NVIDIA_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${NVIDIA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# --- 4. Logging / Diagnostics ---
for override in "$@"; do
  if [ "${override}" = "training.cuda_debug=true" ] \
    || [ "${override}" = "+training.cuda_debug=true" ]; then
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
  fi
done

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "Config: ${CONFIG_BASENAME}"
echo "Config path: ${CONFIG_PATH:-<default>}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-<unset>}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PROJECT_STORAGE_ROOT=${PROJECT_STORAGE_ROOT}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
echo "WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR}"
echo "WANDB_DATA_DIR=${WANDB_DATA_DIR}"
echo "WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR}"
echo "NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
"${PYTHON_BIN}" -V
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

"${PYTHON_BIN}" - <<'PY'
import os
import sys

print("[check] python =", sys.executable)
print("[check] REPO_ROOT =", os.environ.get("REPO_ROOT"))
print("[check] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))

import torch
import h5py
import zarr
import numba
import hydra
import omegaconf
import numpy

print("[check] torch =", torch.__version__)
print("[check] numpy =", numpy.__version__)
print("[check] h5py =", h5py.__version__)
print("[check] zarr =", zarr.__version__)
print("[check] numba =", numba.__version__)
print("[check] hydra =", hydra.__version__)
print("[check] omegaconf =", omegaconf.__version__)
try:
    import mujoco
    print("[check] mujoco =", mujoco.__version__)
except Exception as exc:
    print("[check] mujoco unavailable =", repr(exc))
print("[check] cuda available =", torch.cuda.is_available())
config_name = os.environ.get("CONFIG_NAME", "")
if "image" in config_name:
    from thesis.dataset.lerobot_image_dataset import LeRobotImageDataset
    from thesis.policy.cumulative_one_step_flow_unet_img_policy import CumulativeOneStepFlowUnetImagePolicy
    from thesis.workspace.cumulative_one_step_flow_unet_img_workspace import TrainCumulativeOneStepFlowUnetImageWorkspace

    print("[check] image dataset import OK:", LeRobotImageDataset.__module__)
    print("[check] dataset has episode ids =", hasattr(LeRobotImageDataset, "get_sampler_episode_ids"))
    print("[check] image policy import OK:", CumulativeOneStepFlowUnetImagePolicy.__name__)
    print("[check] workspace import OK:", TrainCumulativeOneStepFlowUnetImageWorkspace.__name__)
elif "lowdim" in config_name:
    from thesis.dataset.mimicgen_lowdim_dataset import MimicGenLowdimDataset
    from thesis.policy.flow_lh_unet_lowdim_policy import OneFlowMatchingLongHorizonUnetLowdimPolicy
    from thesis.workspace.flow_longhorizon_unet_lowdim_workspace import TrainFlowMatchingUnetLowdimWorkspace

    print("[check] lowdim dataset import OK:", MimicGenLowdimDataset.__module__)
    print("[check] dataset has episode ids =", hasattr(MimicGenLowdimDataset, "get_sampler_episode_ids"))
    print("[check] lowdim policy import OK:", OneFlowMatchingLongHorizonUnetLowdimPolicy.__name__)
    print("[check] workspace import OK:", TrainFlowMatchingUnetLowdimWorkspace.__name__)
else:
    print("[check] skipped policy-specific import checks for CONFIG_NAME =", config_name)
PY

# --- 5. Hydra Overrides ---
HYDRA_OVERRIDES=(
  "training.device=cuda:0"
  "logging.mode=${WANDB_MODE}"
)

HYDRA_OVERRIDES+=("output_dir=${OUTPUT_DIR}")
HYDRA_OVERRIDES+=("hydra.run.dir=${OUTPUT_DIR}")

if [ "$#" -gt 0 ]; then
  echo "Extra Hydra overrides: $*"
fi

# --- 6. Training Launch ---
echo "Starting long-horizon one-flow UNet training..."
"${PYTHON_BIN}" thesis/train.py \
  "${HYDRA_CONFIG_ARGS[@]}" \
  --config-name="${CONFIG_BASENAME}" \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
