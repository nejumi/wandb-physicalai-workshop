"""Train a PPO agent on Franka Reach in Isaac Lab and log to W&B.

This script is the Isaac Lab counterpart of ``01_train_diffusion_pusht.py``.
Where that script trains a Diffusion Policy on a 2-D pushing task via
behaviour cloning, this script trains a PPO agent on a 3-D reaching task via
RL — demonstrating that the same W&B + Weave observability stack works
regardless of simulation backend or learning algorithm.

What gets logged automatically:
- **W&B metrics**: per-iteration reward, episode length, value loss, policy
  loss, KL divergence (via Isaac Lab's ``--logger wandb`` flag).
- **W&B video**: rollout clips at every ``--video-interval`` iterations, shown
  inline in the W&B run UI (via ``--video`` flag).
- **W&B config**: full task + PPO hyperparameter config attached to the run.

Prerequisites
-------------
Isaac Lab must be installed.  The recommended path is the NVIDIA NGC container::

    # Pull the Isaac Lab container (requires NVIDIA account)
    docker pull nvcr.io/nvidia/isaac-lab:2.0.0

    # Or install from source (requires Isaac Sim):
    # https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/

    # Inside the container / environment:
    pip install isaaclab_rl[rsl-rl]

Quick start
-----------
::

    # Fast smoke test (fewer envs, fewer iterations, ~5 min on H100)
    python scripts/08_isaac_lab_train.py \\
        --num-envs 256 \\
        --max-iterations 300 \\
        --run-name franka-reach-smoke

    # Workshop-quality run (~30 min on H100)
    python scripts/08_isaac_lab_train.py \\
        --num-envs 4096 \\
        --max-iterations 3000 \\
        --run-name franka-reach-workshop \\
        --wandb-entity wandb-japan

Comparison with the LeRobot run
--------------------------------
+-------------------+-----------------------------+-------------------------+
| Dimension         | LeRobot (script 01)         | Isaac Lab (this script) |
+-------------------+-----------------------------+-------------------------+
| Task              | PushT (2-D pushing)         | Franka Reach (3-D)      |
| Learning          | Diffusion / BC              | PPO (RL)                |
| Sim backend       | gym-pusht (pymunk)          | Isaac Lab (PhysX/CUDA)  |
| Parallel envs     | 1                           | 4096                    |
| W&B metrics       | loss, lr                    | reward, ep-len, losses  |
| W&B video         | eval rollout MP4            | rollout MP4 (every N)   |
| Weave             | replay_rollout traces       | (extend: see below)     |
+-------------------+-----------------------------+-------------------------+

Extending with Weave
--------------------
After training, pass the W&B run ID to the Weave helper to link the Isaac
run the same way the eval pipeline links LeRobot runs::

    import weave
    client = weave.init("wandb-japan/physical-ai-workshop")
    client.set_wandb_run_context(run_id="<isaac-run-id>", step=0)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physical_ai_workshop.isaac_engine import IsaacTrainArgs, run_isaac_training


def build_args(ns: argparse.Namespace) -> IsaacTrainArgs:
    return IsaacTrainArgs(
        task=ns.task,
        num_envs=ns.num_envs,
        max_iterations=ns.max_iterations,
        seed=ns.seed,
        output_dir=Path(ns.output_dir),
        run_name=ns.run_name,
        wandb_project=ns.wandb_project,
        wandb_entity=ns.wandb_entity or None,
        record_video=not ns.no_video,
        video_interval=ns.video_interval,
        video_length=ns.video_length,
        wandb_tags=["isaac-lab", "franka", "ppo", "workshop"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on Isaac Lab Franka Reach, log to W&B."
    )
    parser.add_argument(
        "--task",
        default="Isaac-Reach-Franka-v0",
        help="Isaac Lab registered task ID.",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "workspace" / "runs" / "isaac-franka-reach"),
    )
    parser.add_argument("--run-name", default="franka-reach-ppo")
    parser.add_argument("--wandb-project", default="physical-ai-workshop")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-video", action="store_true", help="Disable video recording.")
    parser.add_argument(
        "--video-interval",
        type=int,
        default=500,
        help="Log a rollout video every N training iterations.",
    )
    parser.add_argument("--video-length", type=int, default=200, help="Frames per clip.")
    ns = parser.parse_args()

    train_args = build_args(ns)
    print(f"[isaac] task={train_args.task}  num_envs={train_args.num_envs}")
    print(f"[isaac] wandb={train_args.wandb_project}  run={train_args.run_name}")
    run_isaac_training(train_args)
