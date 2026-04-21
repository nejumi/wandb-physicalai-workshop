"""Compare-summary helpers are the only pieces attendees run on their laptops,
so they should keep working without torch/gym/wandb available.
"""

from __future__ import annotations

import json
from pathlib import Path

from physical_ai_workshop.compare import (
    Candidate,
    compare_candidates,
    compare_summaries,
    from_summary,
)


def _write_summary(path: Path, tag: str, metrics: dict) -> Path:
    payload = {"tag": tag, "metrics": metrics, "episodes": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_from_summary_picks_up_tag_and_metrics(tmp_path: Path) -> None:
    p = _write_summary(tmp_path / "a.json", "baseline", {"success_rate": 0.4, "avg_steps": 150})
    cand = from_summary(p)
    assert cand.label == "baseline"
    assert cand.metrics == {"success_rate": 0.4, "avg_steps": 150.0}


def test_from_summary_explicit_label_overrides_tag(tmp_path: Path) -> None:
    p = _write_summary(tmp_path / "b.json", "ignored", {"success_rate": 0.1})
    cand = from_summary(p, label="override")
    assert cand.label == "override"


def test_compare_candidates_renders_markdown_and_deltas() -> None:
    left = Candidate(label="baseline", metrics={"success_rate": 0.3, "avg_steps": 200})
    right = Candidate(label="candidate", metrics={"success_rate": 0.6, "avg_steps": 150})

    markdown, deltas = compare_candidates(left, right)

    assert "# Policy Comparison" in markdown
    assert "| success_rate |" in markdown
    assert deltas["success_rate"] == {"left": 0.3, "right": 0.6, "delta": 0.3}
    assert deltas["avg_steps"] == {"left": 200, "right": 150, "delta": -50.0}


def test_compare_candidates_handles_missing_metric() -> None:
    left = Candidate(label="l", metrics={"a": 1.0})
    right = Candidate(label="r", metrics={"b": 2.0})

    _markdown, deltas = compare_candidates(left, right)

    assert deltas["a"]["delta"] is None
    assert deltas["b"]["delta"] is None


def test_compare_summaries_backcompat_paths_are_labelled(tmp_path: Path) -> None:
    _, deltas = compare_summaries(
        {"tag": "old-left", "metrics": {"x": 1}},
        {"tag": "old-right", "metrics": {"x": 2}},
    )
    assert deltas["x"] == {"left": 1.0, "right": 2.0, "delta": 1.0}
