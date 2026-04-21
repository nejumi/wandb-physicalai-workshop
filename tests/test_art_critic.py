"""ART critic tests — pure data pipeline, no GPU / network."""

from __future__ import annotations

import json
from pathlib import Path

from physical_ai_workshop.art_critic import (
    RECOVERY_ACTIONS,
    build_critique_prompt,
    build_trajectories_from_summary,
    run_art_smoke,
    score_critique,
    summarise_critiques,
)


def _fake_summary(path: Path) -> Path:
    payload = {
        "tag": "unit-test",
        "metrics": {"success_rate": 1 / 3},
        "episodes": [
            {
                "episode_id": "u-000", "seed": 1, "success": True,
                "max_coverage": 0.99, "final_coverage": 0.99, "reward_sum": 10,
                "steps": 80, "failure_mode": "none",
            },
            {
                "episode_id": "u-001", "seed": 2, "success": False,
                "max_coverage": 0.02, "final_coverage": 0.02, "reward_sum": 0.1,
                "steps": 300, "failure_mode": "no_contact",
            },
            {
                "episode_id": "u-002", "seed": 3, "success": False,
                "max_coverage": 0.6, "final_coverage": 0.5, "reward_sum": 3.0,
                "steps": 250, "failure_mode": "unstable_push",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_prompt_has_system_and_user_roles() -> None:
    prompt = build_critique_prompt({
        "episode_id": "x", "seed": 0, "success": True,
        "max_coverage": 0.9, "final_coverage": 0.9, "steps": 100,
        "failure_mode": "none",
    })
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert "PushT" in prompt[0]["content"]
    assert "episode_id" not in prompt[1]["content"].lower() or "x" in prompt[1]["content"]


def test_score_success_case_expects_retry() -> None:
    reward, action = score_critique({"success": True, "failure_mode": "none"}, "retry\nlgtm")
    assert reward == 1.0
    assert action == "retry"


def test_score_no_contact_case_expects_abort() -> None:
    reward, _ = score_critique(
        {"success": False, "failure_mode": "no_contact"},
        "abort\ntoo far from the block",
    )
    assert reward == 1.0


def test_score_format_only_reward_when_wrong_action() -> None:
    reward, action = score_critique(
        {"success": True, "failure_mode": "none"},
        "abort\nmisdiagnosed",
    )
    assert action == "abort"
    assert 0.1 < reward < 0.3


def test_score_zero_reward_for_malformed_response() -> None:
    reward, action = score_critique({"success": True, "failure_mode": "none"}, "I don't know")
    assert reward == 0.0
    assert action not in RECOVERY_ACTIONS


def test_build_trajectories_synthetic_matches_expert(tmp_path: Path) -> None:
    path = _fake_summary(tmp_path / "summary.json")
    critiques = build_trajectories_from_summary(path, synthetic=True)
    assert [c.recovery_action for c in critiques] == ["retry", "abort", "recovery_stroke"]
    assert all(c.reward == 1.0 for c in critiques)


def test_run_art_smoke_dry_run_writes_output(tmp_path: Path) -> None:
    path = _fake_summary(tmp_path / "s.json")
    out = tmp_path / "critic.json"
    summary = run_art_smoke(
        summary_path=path,
        dry_run=True,
        output_path=out,
    )
    assert summary["mode"] == "dry_run"
    assert summary["n"] == 3
    assert summary["action_match_rate"] == 1.0
    assert out.exists()
    saved = json.loads(out.read_text())
    assert len(saved["critiques"]) == 3


def test_summarise_empty_list() -> None:
    s = summarise_critiques([])
    assert s == {"n": 0, "mean_reward": 0.0, "action_match_rate": 0.0}
