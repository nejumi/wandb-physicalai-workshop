#!/bin/bash
#SBATCH --job-name=pusht-eval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

ENV_PATH="${ENV_PATH:-/mnt/home/${USER%%+*}+cwb607/.mamba/envs/aimo3-art}"
export PATH="${ENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/mnt/home/${USER%%+*}+cwb607/.cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-physical-ai-workshop}"
mkdir -p slurm-logs

: "${CHECKPOINT:?set CHECKPOINT=/path/to/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT}/../../eval}"
TAG="${TAG:-policy}"
NUM_EPISODES="${NUM_EPISODES:-20}"
WEAVE_PROJECT="${WEAVE_PROJECT:-physical-ai-workshop}"

srun python scripts/03_evaluate.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-episodes "${NUM_EPISODES}" \
    --tag "${TAG}" \
    --wandb-project "${WANDB_PROJECT}" \
    --weave-project "${WEAVE_PROJECT}"
