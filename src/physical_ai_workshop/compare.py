"""Compare two evaluation summaries (local JSON or W&B run IDs).

We keep local JSON compatibility for quick offline demos while adding a
``from_wandb_run`` helper that pulls the summary section of a finished W&B
run. That matters at the workshop because attendees usually train on one
node and compare from their laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Candidate:
    label: str
    metrics: Dict[str, float]
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def from_summary(path: str | Path, label: str | None = None) -> Candidate:
    payload = _load_json(path)
    metrics = payload.get("metrics", {})
    numeric_metrics: Dict[str, float] = {
        k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
    }
    return Candidate(
        label=label or payload.get("tag", str(path)),
        metrics=numeric_metrics,
        episodes=payload.get("episodes", []),
        source=str(path),
    )


def from_wandb_run(run_path: str, label: str | None = None) -> Candidate:
    """Fetch summary metrics from a finished W&B run via the public API."""
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    summary: Dict[str, float] = {}
    for key, value in dict(run.summary).items():
        if not key.startswith("eval/"):
            continue
        if isinstance(value, (int, float)):
            summary[key.replace("eval/", "", 1)] = float(value)
    return Candidate(label=label or run.name, metrics=summary, source=run_path)


def compare_candidates(left: Candidate, right: Candidate) -> Tuple[str, Dict[str, Any]]:
    keys = sorted(set(left.metrics) | set(right.metrics))
    deltas: Dict[str, Any] = {}
    rows: List[str] = []
    for key in keys:
        lv = left.metrics.get(key)
        rv = right.metrics.get(key)
        delta = None
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            delta = round(float(rv) - float(lv), 4)
        deltas[key] = {"left": lv, "right": rv, "delta": delta}
        rows.append(f"| {key} | {lv} | {rv} | {delta} |")

    markdown = "\n".join(
        [
            "# Policy Comparison",
            "",
            f"- Left: `{left.label}` (source: {left.source or 'local'})",
            f"- Right: `{right.label}` (source: {right.source or 'local'})",
            "",
            "| Metric | Left | Right | Delta |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
        ]
    )
    return markdown, deltas


# ---------- backwards-compatible helpers (kept for the old mock CLI) ----------


def load_summary(path: str | Path) -> Dict[str, Any]:
    return _load_json(path)


def compare_summaries(left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    left_cand = Candidate(
        label=left.get("tag") or left.get("job_name") or "left",
        metrics={k: float(v) for k, v in left.get("metrics", {}).items() if isinstance(v, (int, float))},
        source=left.get("tag") or "",
    )
    right_cand = Candidate(
        label=right.get("tag") or right.get("job_name") or "right",
        metrics={k: float(v) for k, v in right.get("metrics", {}).items() if isinstance(v, (int, float))},
        source=right.get("tag") or "",
    )
    return compare_candidates(left_cand, right_cand)
