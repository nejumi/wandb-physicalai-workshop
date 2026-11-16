#!/usr/bin/env bash
set -euo pipefail

workshop_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaaclab_root="${ISAACLAB_DIR:-/content/IsaacLab}"
uv_bin="${UV_BIN:-uv}"
"${uv_bin}" --version
export WANDB_PROJECT="${WANDB_PROJECT:-physical-ai-workshop}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')}"
export WANDB_NAME="${WANDB_NAME:-isaaclab-cartpole-colab}"
export WANDB_RESUME=allow

cd "${isaaclab_root}"
"${uv_bin}" run --locked --python 3.12 --extra wandb,video isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Cartpole \
  --experiment_name wandb-course-cartpole \
  --run_timestamp "${WANDB_RUN_ID}" \
  --num_envs "${ISAACLAB_NUM_ENVS:-32}" \
  --max_iterations "${ISAACLAB_ITERATIONS:-2}" \
  --logger wandb \
  --log_project_name "${WANDB_PROJECT}" \
  --video \
  --video_length "${ISAACLAB_VIDEO_LENGTH:-60}" \
  --video_interval "${ISAACLAB_VIDEO_INTERVAL:-100}"

"${uv_bin}" run --locked --python 3.12 --extra wandb,video python "${workshop_root}/scripts/10_log_isaac_outputs.py" \
  --run-id "${WANDB_RUN_ID}" \
  --log-root "${isaaclab_root}/logs/rsl_rl/wandb-course-cartpole/${WANDB_RUN_ID}" \
  --project "${WANDB_PROJECT}"
