"""Compare two policies side by side.

Works with either local ``eval_summary.json`` files (produced by
``03_evaluate.py``) or W&B run paths (``entity/project/run_id``).

Example:
    python scripts/04_compare.py \
        --left workspace/runs/diffusion-pusht/eval/eval_summary.json \
        --right workspace/runs/act-pusht/eval/eval_summary.json \
        --left-label diffusion \
        --right-label act \
        --output workspace/reports/latest-comparison.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physical_ai_workshop.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["compare", *sys.argv[1:]]))
