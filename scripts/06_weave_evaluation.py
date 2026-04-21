"""Publish a formal ``weave.Evaluation`` for a set of rollouts.

The Evaluation combines a tiny dataset (one row per rollout) with a small
set of scorers — success, coverage, efficiency, failure-mode — so attendees
can see how to grow the evaluation suite without re-running any rollouts.

Example:
    python scripts/06_weave_evaluation.py \
        --eval-summary workspace/runs/diffusion-pusht/eval/eval_summary.json \
        --weave-project physical-ai-workshop \
        --dataset-name pusht-rollouts-2026-04 \
        --policy-name diffusion-10k
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physical_ai_workshop.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["weave-eval", *sys.argv[1:]]))
