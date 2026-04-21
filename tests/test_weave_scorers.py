"""Scorers are pure functions; they should work without a live Weave backend."""

from physical_ai_workshop.weave_scorers import (
    coverage_scorer,
    efficiency_scorer,
    failure_mode_scorer,
    success_scorer,
)


def test_success_scorer_defaults_to_false() -> None:
    assert success_scorer({})["success"] is False
    assert success_scorer({"success": True})["success"] is True


def test_coverage_scorer_flags_near_miss_only_on_failure() -> None:
    near_miss = coverage_scorer({"max_coverage": 0.92, "success": False})
    clean_success = coverage_scorer({"max_coverage": 0.99, "success": True})
    assert near_miss["near_miss"] is True
    assert near_miss["clears_half"] is True
    assert clean_success["near_miss"] is False
    assert clean_success["clears_half"] is True


def test_efficiency_scorer_requires_both_success_and_short_steps() -> None:
    efficient = efficiency_scorer({"success": True, "steps": 90})
    slow = efficiency_scorer({"success": True, "steps": 200})
    failed = efficiency_scorer({"success": False, "steps": 50})
    assert efficient["efficient_success"] is True
    assert slow["efficient_success"] is False
    assert failed["efficient_success"] is False


def test_failure_mode_scorer_defaults_to_none() -> None:
    result = failure_mode_scorer({})
    assert result["success"] is True
    assert result["no_contact"] is False

def test_failure_mode_scorer_classifies_modes() -> None:
    assert failure_mode_scorer({"failure_mode": "timeout"})["timeout"] is True
    assert failure_mode_scorer({"failure_mode": "no_contact"})["no_contact"] is True
    assert failure_mode_scorer({"failure_mode": "near_miss"})["near_miss"] is True
    assert failure_mode_scorer({"failure_mode": "unstable_push"})["unstable_push"] is True
    assert failure_mode_scorer({"failure_mode": "none"})["success"] is True
