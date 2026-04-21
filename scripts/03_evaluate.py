"""Run rollouts against a trained checkpoint and log videos + Weave traces.

Given a LeRobot ``pretrained_model`` directory this script:

- rolls out N episodes with deterministic seeds;
- writes ``eval_summary.json`` for downstream comparison;
- uploads rollout videos as a W&B Table (one row per episode, plus a
  failure-only table for quick debugging); and
- optionally traces each episode with Weave so the instructor can walk
  through a failure in the Weave UI during the workshop.

Example (post-training smoke test):
    python scripts/03_evaluate.py \
        --checkpoint workspace/runs/diffusion-pusht/checkpoints/last/pretrained_model \
        --output-dir workspace/runs/diffusion-pusht/eval \
        --tag diffusion \
        --num-episodes 10 \
        --weave-project physical-ai-workshop
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-every", type=int, default=2)
    parser.add_argument("--tag", default="policy")
    parser.add_argument("--wandb-project", default="physical-ai-workshop")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--weave-project", default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    ns = parser.parse_args()

    argv = [
        "eval",
        "--checkpoint", ns.checkpoint,
        "--output-dir", ns.output_dir,
        "--num-episodes", str(ns.num_episodes),
        "--seed-start", str(ns.seed_start),
        "--device", ns.device,
        "--record-every", str(ns.record_every),
        "--tag", ns.tag,
        "--wandb-project", ns.wandb_project,
        "--video-fps", str(ns.video_fps),
    ]
    if ns.wandb_entity:
        argv += ["--wandb-entity", ns.wandb_entity]
    if ns.wandb_run_name:
        argv += ["--wandb-run-name", ns.wandb_run_name]
    if ns.weave_project:
        argv += ["--weave-project", ns.weave_project]
    raise SystemExit(main(argv))
