# Example eval summaries

Two hand-authored `eval_summary.json` files so attendees on a laptop (no GPU)
can still run `scripts/04_compare.py`, `scripts/06_weave_evaluation.py`, and
`scripts/07_art_critic.py` without training anything.

| File | Shape |
| --- | --- |
| `eval_summary_example.json` | 20 episodes, 60% success, mix of near-miss / no-contact / unstable-push failures |
| `eval_summary_candidate.json` | 20 episodes, 80% success — the "after" run |

Example:

```bash
python scripts/04_compare.py \
    --left examples/eval_summary_example.json \
    --right examples/eval_summary_candidate.json \
    --output workspace/reports/example-compare.md

python scripts/07_art_critic.py \
    --eval-summary examples/eval_summary_candidate.json \
    --output workspace/runs/example/critic/summary.json
```
