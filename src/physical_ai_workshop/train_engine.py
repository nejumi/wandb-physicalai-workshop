"""Wrapper around ``lerobot-train`` for workshop jobs.

We keep this thin on purpose: LeRobot already ships a production-grade training
pipeline, and rewriting it would only add bugs. The workshop value is in the
glue — W&B run naming, deterministic output paths, and a final artifact upload
so the registry promotion step has something to grab.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainArgs:
    policy: str = "diffusion"          # "diffusion" | "act"
    dataset_repo_id: str = "lerobot/pusht"
    env_type: str = "pusht"
    output_dir: Path = Path("workspace/runs/train")
    steps: int = 5000
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    eval_freq: int = 1000
    save_freq: int = 5000
    log_freq: int = 50
    wandb_enable: bool = True
    wandb_mode: Optional[str] = None   # "online" | "offline" | "disabled"
    wandb_project: str = "physical-ai-workshop"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    extra: List[str] = None  # type: ignore[assignment]


def build_command(args: TrainArgs) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.type={args.policy}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={args.dataset_repo_id}",
        "--dataset.video_backend=pyav",
        f"--env.type={args.env_type}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--wandb.enable={'true' if args.wandb_enable else 'false'}",
        f"--wandb.project={args.wandb_project}",
    ]
    if args.wandb_mode:
        cmd.append(f"--wandb.mode={args.wandb_mode}")
    if args.wandb_entity:
        cmd.append(f"--wandb.entity={args.wandb_entity}")
    if args.run_name:
        cmd.append(f"--job_name={args.run_name}")
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def run_training(args: TrainArgs, env: Optional[Dict[str, str]] = None) -> Path:
    """Run the training job as a subprocess and return its output dir.

    LeRobot refuses to start if ``output_dir`` already exists (to protect
    existing runs). We only ensure the parent directory exists and let
    ``lerobot-train`` create the run dir itself.
    """
    args.output_dir = Path(args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args)
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    print("[train] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=process_env)
    return args.output_dir


def latest_checkpoint(output_dir: Path) -> Path:
    """Return the most recent ``pretrained_model`` directory inside ``output_dir``.

    LeRobot saves checkpoints under ``checkpoints/<step>/pretrained_model`` and
    a rolling ``checkpoints/last`` symlink. We prefer ``last`` and fall back to
    the largest numeric step directory.
    """
    output_dir = Path(output_dir)
    last = output_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.exists():
        raise FileNotFoundError(f"No checkpoints under {checkpoints}")
    numeric = [p for p in checkpoints.iterdir() if p.is_dir() and p.name.isdigit()]
    if not numeric:
        raise FileNotFoundError(f"No numeric checkpoint directories in {checkpoints}")
    best = max(numeric, key=lambda p: int(p.name))
    return best / "pretrained_model"
