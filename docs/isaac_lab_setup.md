# Isaac Lab + W&B Setup Guide

This guide covers running `scripts/08_isaac_lab_train.py` — PPO on Franka Reach
inside Isaac Lab, with automatic W&B metric and video logging.

## Why Isaac Lab + W&B

| LeRobot (scripts 01–07) | Isaac Lab (script 08) |
|---|---|
| 2-D PushT, behaviour cloning | 3-D Franka manipulation, RL |
| gym-pusht (CPU, pymunk) | Isaac Lab (GPU, PhysX CUDA) |
| 1 env, sequential rollout | 4096 parallel envs |
| W&B: loss curve + eval video | W&B: reward curve + rollout video |
| Weave: per-episode traces | Weave: same pattern, same API |

Same W&B project, same Weave project — unified dashboard across both frameworks.

## Environment Setup

### Option A — NVIDIA NGC Container (recommended)

```bash
# Requires NVIDIA account (free): https://ngc.nvidia.com
docker pull nvcr.io/nvidia/isaac-lab:2.0.0

# Run interactively on a GPU node
docker run --gpus all -it --rm \
    -v /mnt/home/$USER/wandb-physicalai-workshop:/workspace \
    nvcr.io/nvidia/isaac-lab:2.0.0 bash

# Inside the container
cd /workspace
pip install -e ".[dev]"   # installs workshop package
wandb login               # or export WANDB_API_KEY=...
```

### Option B — From Source (Isaac Sim required)

```bash
# 1. Install Isaac Sim 4.x from NVIDIA Omniverse Launcher
#    https://developer.nvidia.com/isaac-sim

# 2. Clone Isaac Lab alongside Isaac Sim
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install rsl_rl   # installs RSL-RL + Isaac Lab

# 3. Install workshop package into the same Python env
pip install -e /path/to/wandb-physicalai-workshop
```

## Running the Training

```bash
# Workshop-quality run (~30 min on H100, success rate > 90%)
python scripts/08_isaac_lab_train.py \
    --num-envs 4096 \
    --max-iterations 3000 \
    --run-name franka-reach-workshop \
    --wandb-entity wandb-japan

# Quick smoke test (~5 min)
python scripts/08_isaac_lab_train.py \
    --num-envs 256 \
    --max-iterations 300 \
    --run-name franka-reach-smoke
```

## What You See in W&B

### Metrics (auto-logged by Isaac Lab `--logger wandb`)
- `Train/mean_reward` — episode return, rises from ~0 to ~8 over 3k iterations
- `Train/mean_episode_length` — typically converges to 80–120 steps
- `Loss/value_function`, `Loss/surrogate` — PPO losses
- `Policy/mean_noise_std`, `Policy/mean_action_norm`

### Videos (auto-logged via `--video` flag)
Isaac Lab records a rollout clip every `--video-interval` iterations and
uploads it to W&B as a `wandb.Video`.  By iteration 1000 you can see the
Franka arm confidently reaching to the target sphere.

### Config
Full task + PPO hyperparameters attached to the run — same as LeRobot runs.

## Connecting Isaac Results to Weave

After training completes, the W&B run ID is printed.  Pass it to Weave the
same way the eval pipeline does for LeRobot runs:

```python
import weave

client = weave.init("wandb-japan/physical-ai-workshop")
client.set_wandb_run_context(run_id="<wandb-run-id>", step=0)

@weave.op()
def log_isaac_episode(episode: dict) -> dict:
    return episode

log_isaac_episode({
    "task": "Isaac-Reach-Franka-v0",
    "iteration": 3000,
    "mean_reward": 7.8,
    "success_rate": 0.94,
})
client.clear_wandb_run_context()
```

## Key Takeaway for the Workshop

> Whether you're doing **imitation learning on a 2-D task with LeRobot** or
> **RL on a 3-D manipulation task with Isaac Lab**, the observability layer is
> identical: W&B logs every run, Weave traces every episode, Models tracks
> every checkpoint.  The framework changes; the workflow doesn't.
