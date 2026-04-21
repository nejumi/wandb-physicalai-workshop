"""Run the ART rollout-critic pipeline over an ``eval_summary.json``.

Dry-run mode (default) exercises the whole data path without touching a GPU:

    python scripts/07_art_critic.py \
        --eval-summary workspace/runs/diffusion-10k/eval/eval_summary.json \
        --output workspace/runs/diffusion-10k/critic/summary.json

Attendees who have a GPU + vLLM can pass ``--no-dry-run`` and a ``--weave-project``
to stream critiques into W&B Weave; the actual ART training step is left to the
attendee so they pick their own base model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physical_ai_workshop.art_critic import run_art_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-summary", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--weave-project", default=None)
    parser.add_argument("--project", default="physical-ai-workshop")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", default=True)
    ns = parser.parse_args()

    summary = run_art_smoke(
        summary_path=Path(ns.eval_summary),
        project=ns.project,
        weave_project=ns.weave_project,
        dry_run=ns.dry_run,
        output_path=Path(ns.output) if ns.output else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
