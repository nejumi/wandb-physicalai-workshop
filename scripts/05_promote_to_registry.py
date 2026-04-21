"""Promote the best checkpoint to W&B Models.

The checkpoint is uploaded as a W&B Artifact (type=``model``) and can
optionally be linked into a Registry collection. Eval metrics from the
corresponding ``eval_summary.json`` are attached to the artifact so the
Registry UI shows how the model performed before promotion.

Example:
    python scripts/05_promote_to_registry.py \
        --checkpoint workspace/runs/diffusion-pusht/checkpoints/last/pretrained_model \
        --eval-summary workspace/runs/diffusion-pusht/eval/eval_summary.json \
        --artifact-name pusht-diffusion \
        --alias workshop-candidate \
        --registry-collection wandb-registry-model/physical-ai-workshop
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physical_ai_workshop.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["promote", *sys.argv[1:]]))
