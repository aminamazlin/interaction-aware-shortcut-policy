#!/bin/bash
#SBATCH --job-name=eval-thesis
#SBATCH --output=/home/amazlin/bachelor-thesis/logs/slurm/%x_%j.out
#SBATCH --error=/home/amazlin/bachelor-thesis/logs/slurm/%x_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=1:00:00

set -e
set -o pipefail

# Usage:
#   RUN_DIR=/gpfs/work5/0/prjs2121/amazlin/bachelor-thesis/outputs/2026.05.30/13.33.08_train_csl_shortcut_longhorizon_unet_lowdim_coffee_preparation_d0 sbatch thesis/eval.sh
#   sbatch thesis/eval.sh /gpfs/work5/0/prjs2121/amazlin/bachelor-thesis/outputs/2026.05.30/13.33.08_train_csl_shortcut_longhorizon_unet_lowdim_coffee_preparation_d0
#   sbatch --export=ALL,NUM_INFERENCE_STEPS=1 thesis/eval.sh /gpfs/work5/0/prjs2121/amazlin/bachelor-thesis/outputs/2026.05.29/22.01.01_train_diffusion_unet_lowdim_ddim_coffee_preparation_d0
#   sbatch thesis/eval.sh /gpfs/work5/0/prjs2121/amazlin/bachelor-thesis/outputs/2026.06.14/15.10.38_train_onedp_lowdim_coffee_preparation_d0
#
# Optional environment variables:
#   OUTPUT_DIR=/gpfs/work5/0/prjs2121/amazlin/bachelor-thesis/eval_outputs/my_eval
#   CONDA_ENV_NAME=auto
#   TOP_K=5
#   DEVICE=cuda:0
#   INFERENCE_CFG_SCALE=1.0
#   NUM_INFERENCE_STEPS=1
#   WANDB_MODE=offline

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

# --- 2. Arguments / Output Directory ---
RUN_DIR="${RUN_DIR:-${1:-}}"

if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: provide RUN_DIR as an env var or first sbatch argument." >&2
  echo "Example: RUN_DIR=/path/to/run sbatch thesis/eval.sh" >&2
  exit 1
fi

RUN_DIR="${RUN_DIR%/}"
RUN_BASENAME="$(basename "${RUN_DIR}")"
if [ "${RUN_BASENAME}" = "checkpoints" ]; then
  RUN_BASENAME="$(basename "$(dirname "${RUN_DIR}")")"
fi

export PROJECT_STORAGE_ROOT="${PROJECT_STORAGE_ROOT:-/gpfs/work5/0/prjs2121/amazlin/bachelor-thesis}"
EVAL_DATE="$(date +%Y.%m.%d)"
EVAL_TIME="$(date +%H.%M.%S)"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_STORAGE_ROOT}/eval_outputs/${EVAL_DATE}/${EVAL_TIME}_${RUN_BASENAME}}"
export TOP_K="${TOP_K:-5}"
export DEVICE="${DEVICE:-cuda:0}"
export INFERENCE_CFG_SCALE="${INFERENCE_CFG_SCALE:-1.0}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"

mkdir -p "${PROJECT_STORAGE_ROOT}/wandb"

# --- 3. Environment Activation ---
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_STORAGE_ROOT}/wandb/eval}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_STORAGE_ROOT}/wandb/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_STORAGE_ROOT}/wandb/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

if [ -z "${CONDA_ENV_NAME:-}" ] || [ "${CONDA_ENV_NAME}" = "auto" ]; then
  if [[ "${RUN_DIR}" == *robocasa* ]] \
    || [[ "${RUN_DIR}" == *load_dishwasher* ]] \
    || [[ "${RUN_DIR}" == *pack_identical_lunches* ]]; then
    export CONDA_ENV_NAME="robocasa"
  elif [[ "${RUN_DIR}" == *kitchen* ]] \
    || [[ "${RUN_BASENAME}" == *kitchen* ]] \
    || [[ "${RUN_DIR}" == *mimicgen_pick_place* ]] \
    || [[ "${RUN_BASENAME}" == *mimicgen_pick_place* ]]; then
    export CONDA_ENV_NAME="flow_policy_kitchen"
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

CONDA_ENV_PYTHON="${HOME}/miniforge3/envs/${CONDA_ENV_NAME}/bin/python"
if [ -x "${CONDA_ENV_PYTHON}" ]; then
  PYTHON_BIN="${CONDA_ENV_PYTHON}"
else
  PYTHON_BIN="$(command -v python)"
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

# --- 4. Runtime Setup ---
unset DISPLAY
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
unset OMNIGIBSON_HEADLESS

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/thesis:${REPO_ROOT}/baselines/diffusion-policy:${PYTHONPATH:-}"

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

# --- 5. Logging / Diagnostics ---
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "RUN_DIR=${RUN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TOP_K=${TOP_K}"
echo "DEVICE=${DEVICE}"
echo "INFERENCE_CFG_SCALE=${INFERENCE_CFG_SCALE}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-<checkpoint default>}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
"${PYTHON_BIN}" -V
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

"${PYTHON_BIN}" thesis/eval.py --help >/dev/null

# --- 6. Evaluation Launch ---
echo "Starting evaluation..."
EVAL_ARGS=(
  --run-dir "${RUN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --top-k "${TOP_K}" \
  --inference-cfg-scale "${INFERENCE_CFG_SCALE}"
)

if [ -n "${NUM_INFERENCE_STEPS}" ]; then
  EVAL_ARGS+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi

"${PYTHON_BIN}" thesis/eval.py "${EVAL_ARGS[@]}"
