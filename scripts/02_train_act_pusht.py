"""Train an ACT policy on PushT for a baseline comparison.

ACT converges in fewer steps than Diffusion Policy on the PushT benchmark
but has lower asymptotic success. That contrast is a good workshop talking
point when we compare two runs side by side in the W&B UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physical_ai_workshop.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "workspace" / "runs" / "act-pusht"))
    parser.add_argument("--run-name", default="act-pusht")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=4000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--wandb-project", default="physical-ai-workshop")
    parser.add_argument("--wandb-entity", default=None)
    ns = parser.parse_args()

    argv = [
        "train",
        "--policy", "act",
        "--output-dir", ns.output_dir,
        "--steps", str(ns.steps),
        "--batch-size", str(ns.batch_size),
        "--num-workers", str(ns.num_workers),
        "--seed", str(ns.seed),
        "--eval-freq", str(ns.eval_freq),
        "--save-freq", str(ns.save_freq),
        "--log-freq", str(ns.log_freq),
        "--wandb-project", ns.wandb_project,
        "--run-name", ns.run_name,
    ]
    if ns.wandb_entity:
        argv += ["--wandb-entity", ns.wandb_entity]
    raise SystemExit(main(argv))
