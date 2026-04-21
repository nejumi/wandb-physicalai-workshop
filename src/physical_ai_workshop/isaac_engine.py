"""Isaac Lab training utilities for the workshop.

Wraps the Isaac Lab + RSL-RL training pipeline the same way ``train_engine``
wraps LeRobot — a thin subprocess shim that adds W&B project naming and a
deterministic output path, leaving the actual physics simulation and RL loop
entirely to Isaac Lab.

Typical call from ``scripts/08_isaac_lab_train.py``::

    args = IsaacTrainArgs(
        task="Isaac-Reach-Franka-v0",
        num_envs=4096,
        max_iterations=3000,
        wandb_project="physical-ai-workshop",
        wandb_entity="wandb-japan",
    )
    run_isaac_training(args)

The subprocess runs ``python -m isaaclab_rl.scripts.rsl_rl.train``, which is
the canonical Isaac Lab entry-point for RSL-RL based policies.  Isaac Lab must
be installed in the active Python environment (NVIDIA NGC container recommended).

See ``docs/isaac_lab_setup.md`` for environment setup instructions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IsaacTrainArgs:
    # Task
    task: str = "Isaac-Reach-Franka-v0"
    num_envs: int = 4096
    max_iterations: int = 3000
    seed: int = 42
    device: str = "cuda"

    # Output
    output_dir: Path = Path("workspace/runs/isaac")
    run_name: str = "franka-reach-ppo"

    # W&B
    wandb_project: str = "physical-ai-workshop"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    # Video recording (logged to W&B automatically)
    record_video: bool = True
    video_interval: int = 500  # log video every N iterations
    video_length: int = 200    # frames per clip

    # Pass-through extras (e.g. custom PPO overrides)
    extra: List[str] = field(default_factory=list)


ISAACLAB_TRAIN_SCRIPT = os.environ.get(
    "ISAACLAB_RSL_RL_TRAIN",
    "/mnt/home/yyamamoto+cwb607/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py",
)


def build_isaac_command(args: IsaacTrainArgs) -> List[str]:
    """Build the subprocess argv for Isaac Lab's ``rsl_rl/train.py``.

    Isaac Lab's RSL-RL trainer accepts ``--logger wandb`` and
    ``--log_project_name`` as first-class flags, so metrics/videos stream to
    W&B natively. Entity + tags come from env vars (``WANDB_ENTITY`` etc.).
    """
    cmd: List[str] = [
        sys.executable,
        ISAACLAB_TRAIN_SCRIPT,
        f"--task={args.task}",
        f"--num_envs={args.num_envs}",
        f"--max_iterations={args.max_iterations}",
        f"--seed={args.seed}",
        "--headless",
        "--logger=wandb",
        f"--log_project_name={args.wandb_project}",
        f"--run_name={args.run_name}",
    ]
    if args.record_video:
        cmd += [
            "--video",
            f"--video_interval={args.video_interval}",
            f"--video_length={args.video_length}",
        ]
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def run_isaac_training(
    args: IsaacTrainArgs,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    """Launch Isaac Lab training as a subprocess and return the output dir.

    Isaac Lab requires ``AppLauncher`` to be the first thing initialised in
    the process, which is incompatible with running inside the same interpreter
    as other code (e.g. our eval pipeline).  Running as a subprocess sidesteps
    this constraint cleanly.
    """
    args.output_dir = Path(args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_isaac_command(args)
    process_env = os.environ.copy()
    if args.wandb_entity:
        process_env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_tags:
        process_env["WANDB_TAGS"] = ",".join(args.wandb_tags)
    if env:
        process_env.update(env)

    print("[isaac] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=process_env)
    return args.output_dir
