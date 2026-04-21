#!/bin/bash
#SBATCH --job-name=pusht-diffusion
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

# --- env ---------------------------------------------------------------
ENV_PATH="${ENV_PATH:-/mnt/home/${USER%%+*}+cwb607/.mamba/envs/aimo3-art}"
export PATH="${ENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/mnt/home/${USER%%+*}+cwb607/.cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-physical-ai-workshop}"
export WANDB_DIR="${WANDB_DIR:-${PWD}/workspace/wandb}"
mkdir -p "${WANDB_DIR}" slurm-logs

# --- run ---------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-workspace/runs/diffusion-pusht-${SLURM_JOB_ID}}"
RUN_NAME="${RUN_NAME:-diffusion-pusht-${SLURM_JOB_ID}}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

srun python scripts/01_train_diffusion_pusht.py \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --wandb-project "${WANDB_PROJECT}"
