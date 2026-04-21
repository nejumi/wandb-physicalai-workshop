"""Train a Diffusion Policy on PushT and log to W&B.

This is the canonical workshop training run. It delegates to ``lerobot-train``
under the hood — we only add sensible defaults, naming, and a W&B project.

Usage:
    python scripts/01_train_diffusion_pusht.py \
        --output-dir workspace/runs/diffusion-pusht \
        --run-name diffusion-pusht-h100 \
        --steps 10000

For a fast smoke test on H100 (about 2-3 minutes):
    python scripts/01_train_diffusion_pusht.py \
        --output-dir workspace/runs/smoke \
        --run-name smoke \
        --steps 200 --eval-freq 0 --save-freq 200 --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physical_ai_workshop.cli import main


def build_argv(ns: argparse.Namespace) -> list[str]:
    argv = [
        "train",
        "--policy", "diffusion",
        "--output-dir", ns.output_dir,
        "--steps", str(ns.steps),
        "--batch-size", str(ns.batch_size),
        "--num-workers", str(ns.num_workers),
        "--seed", str(ns.seed),
        "--eval-freq", str(ns.eval_freq),
        "--save-freq", str(ns.save_freq),
        "--log-freq", str(ns.log_freq),
        "--wandb-project", ns.wandb_project,
    ]
    if ns.no_wandb:
        argv += ["--no-wandb"]
    if ns.wandb_mode:
        argv += ["--wandb-mode", ns.wandb_mode]
    if ns.wandb_entity:
        argv += ["--wandb-entity", ns.wandb_entity]
    if ns.run_name:
        argv += ["--run-name", ns.run_name]
    return argv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "workspace" / "runs" / "diffusion-pusht"))
    parser.add_argument("--run-name", default="diffusion-pusht")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--wandb-project", default="physical-ai-workshop")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    ns = parser.parse_args()
    raise SystemExit(main(build_argv(ns)))
