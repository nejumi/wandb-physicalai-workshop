"""Weave scorers and evaluation harness for PushT rollouts.

The workshop uses ``weave.Evaluation`` to score a set of rollouts against a
small list of criteria. Scorers are intentionally simple — they exist to show
*how* a team can extend evaluation, not to be the final word on robot quality.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List


def _weave():
    import weave  # type: ignore

    return weave


def success_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": bool(output.get("success", False))}


def coverage_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
    coverage = float(output.get("max_coverage", 0.0))
    return {
        "max_coverage": coverage,
        "clears_half": coverage >= 0.5,
        "near_miss": coverage >= 0.9 and not output.get("success", False),
    }


def efficiency_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
    steps = int(output.get("steps", 0))
    success = bool(output.get("success", False))
    return {
        "steps": steps,
        "efficient_success": success and steps <= 120,
    }


def failure_mode_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(output.get("failure_mode", "none"))
    return {
        "no_contact": mode == "no_contact",
        "near_miss": mode == "near_miss",
        "unstable_push": mode == "unstable_push",
        "timeout": mode == "timeout",
        "success": mode == "none",
    }


def _ensure_weave_cache_dir() -> None:
    """Guarantee a writable cache dir for Weave's diskcache backend.

    Some cluster nodes restrict writes to /tmp. Setting WEAVE_SERVER_CACHE_DIR
    to a project-local path before weave.init() avoids the sqlite3 error.
    We do this unconditionally so the workshop env is predictable.
    """
    import os, pathlib
    if "WEAVE_SERVER_CACHE_DIR" not in os.environ:
        fallback = pathlib.Path("workspace/.weave-cache")
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["WEAVE_SERVER_CACHE_DIR"] = str(fallback.resolve())


def run_weave_evaluation(
    project: str,
    dataset_name: str,
    episodes: List[Dict[str, Any]],
    policy_name: str,
    artifact_id: str = "",
) -> Dict[str, Any]:
    """Publish a Weave Evaluation for ``episodes`` under ``project``.

    ``episodes`` are the dicts produced by ``RolloutRecord.summary()``. We
    rebuild a lightweight ``weave.Model`` that simply returns the stored
    rollout summary so attendees can run scorers over it in the Weave UI.
    """
    _ensure_weave_cache_dir()
    weave = _weave()
    weave.init(project)

    dataset = [
        {
            "id": ep["episode_id"],
            "seed": ep["seed"],
            "expected_success": True,
            "rollout": ep,
        }
        for ep in episodes
    ]

    _policy_name = policy_name
    _artifact_id = artifact_id

    class RolloutReplayModel(weave.Model):
        name: str = _policy_name
        artifact_id: str = _artifact_id

        @weave.op()
        def predict(self, rollout: Dict[str, Any]) -> Dict[str, Any]:
            return rollout

    model = RolloutReplayModel()
    evaluation = weave.Evaluation(
        name=dataset_name,
        dataset=dataset,
        scorers=[success_scorer, coverage_scorer, efficiency_scorer, failure_mode_scorer],
    )
    return asyncio.run(evaluation.evaluate(model))
