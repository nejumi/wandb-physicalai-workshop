"""ART-based rollout critic for the workshop's agentic extension.

The core robot loop (train → eval → promote) is policy-only. The critic
adds a thin *agentic* layer on top: given a PushT rollout summary, an LLM
proposes a recovery strategy ("push harder on the short edge", "re-seed and
retry", "abort — pose is unrecoverable"). OpenPipe ART trains that LLM on
real rollouts using GRPO-style RL.

This module deliberately keeps the ART wiring small and inspectable:

- ``build_critique_prompt(rollout)`` turns one rollout summary into the
  system + user messages the LLM will see.
- ``score_critique(rollout, response_text)`` turns the LLM response into a
  reward. The rubric is public so attendees can tweak it.
- ``build_trajectories_from_summary`` converts an ``eval_summary.json`` into
  a list of ``art.Trajectory`` objects — the input to ``model.train`` or to
  ``weave.Evaluation``.

The ``run_art_smoke`` entry point runs the pipeline end-to-end. In
``dry_run=True`` mode it produces deterministic synthetic responses so the
pipeline is testable without a GPU or network — useful for CI and for
teaching the data flow. In non-dry-run mode it relies on an ART
``TrainableModel`` + ``LocalBackend`` which requires vLLM and a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RECOVERY_ACTIONS = ("retry", "recovery_stroke", "abort")

SYSTEM_PROMPT = (
    "You are a safety-critical supervisor for a PushT manipulation policy. "
    "Given a rollout summary, respond with exactly one recovery action from "
    f"{{{', '.join(RECOVERY_ACTIONS)}}} on the first line, followed by a "
    "short justification on the second line."
)


@dataclass
class RolloutCritique:
    episode_id: str
    rollout: Dict[str, Any]
    prompt: List[Dict[str, str]]
    response: str
    reward: float
    recovery_action: str


def build_critique_prompt(rollout: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format one rollout summary as a two-message chat prompt."""
    user = (
        f"Episode {rollout.get('episode_id', '?')} "
        f"(seed={rollout.get('seed', '?')}):\n"
        f"- success: {rollout.get('success', False)}\n"
        f"- max_coverage: {float(rollout.get('max_coverage', 0.0)):.3f}\n"
        f"- final_coverage: {float(rollout.get('final_coverage', 0.0)):.3f}\n"
        f"- steps: {rollout.get('steps', 0)}\n"
        f"- failure_mode: {rollout.get('failure_mode', 'none')}\n"
        "Pick a recovery action and justify briefly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_action(response: str) -> str:
    """First non-empty stripped line, lowercased, matched against the rubric."""
    for line in response.splitlines():
        token = line.strip().lower().rstrip(".,:")
        if not token:
            continue
        for candidate in RECOVERY_ACTIONS:
            if candidate in token:
                return candidate
        return token
    return ""


def score_critique(rollout: Dict[str, Any], response: str) -> Tuple[float, str]:
    """Reward in ``[0, 1]``. Rewards match the expert policy below.

    - ``success=True`` → expert says ``retry`` (keep going with the same seed).
    - ``failure_mode == 'no_contact'`` → ``abort`` (pose is unrecoverable).
    - everything else → ``recovery_stroke``.

    Format adherence is worth 0.2, action correctness worth 0.8.
    """
    action = _extract_action(response)
    format_ok = action in RECOVERY_ACTIONS
    format_reward = 0.2 if format_ok else 0.0

    success = bool(rollout.get("success", False))
    failure_mode = str(rollout.get("failure_mode", "none"))
    if success:
        expected = "retry"
    elif failure_mode == "no_contact":
        expected = "abort"
    else:
        expected = "recovery_stroke"

    action_reward = 0.8 if action == expected else 0.0
    return format_reward + action_reward, action or "unknown"


def _synthetic_response(rollout: Dict[str, Any]) -> str:
    """Deterministic fake LLM used in dry-run mode — matches the expert rubric."""
    if bool(rollout.get("success", False)):
        return "retry\nSuccess — keep the same plan."
    if str(rollout.get("failure_mode", "none")) == "no_contact":
        return "abort\nAgent never touched the T block; re-initialise instead."
    return "recovery_stroke\nApply a short correction along the missing edge."


def build_trajectories_from_summary(
    summary_path: Path,
    synthetic: bool = True,
    response_fn=None,
) -> List[RolloutCritique]:
    """Load rollouts and attach an LLM response + reward to each.

    ``response_fn(rollout) -> str`` supplies responses; when ``None`` and
    ``synthetic=True`` we use the deterministic fake above so the pipeline
    is exercised without any model call.
    """
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])
    if response_fn is None and synthetic:
        response_fn = _synthetic_response

    out: List[RolloutCritique] = []
    for ep in episodes:
        prompt = build_critique_prompt(ep)
        if response_fn is None:
            raise ValueError("response_fn must be provided when synthetic=False")
        response = response_fn(ep)
        reward, action = score_critique(ep, response)
        out.append(
            RolloutCritique(
                episode_id=str(ep.get("episode_id", "")),
                rollout=ep,
                prompt=prompt,
                response=response,
                reward=reward,
                recovery_action=action,
            )
        )
    return out


def to_art_trajectories(critiques: List[RolloutCritique]) -> List[Any]:
    """Convert ``RolloutCritique`` objects into ``art.Trajectory`` objects."""
    import art  # imported lazily so tests can run without ART installed

    trajectories: List[Any] = []
    for c in critiques:
        messages_and_choices = [
            *c.prompt,
            {"role": "assistant", "content": c.response},
        ]
        trajectories.append(
            art.Trajectory(
                messages_and_choices=messages_and_choices,
                reward=c.reward,
                metrics={"recovery_action_matches": 1.0 if c.reward >= 0.8 else 0.0},
                metadata={
                    "episode_id": c.episode_id,
                    "failure_mode": c.rollout.get("failure_mode", "none"),
                    "recovery_action": c.recovery_action,
                },
            )
        )
    return trajectories


def summarise_critiques(critiques: List[RolloutCritique]) -> Dict[str, Any]:
    if not critiques:
        return {"n": 0, "mean_reward": 0.0, "action_match_rate": 0.0}
    matches = sum(1 for c in critiques if c.reward >= 0.8)
    return {
        "n": len(critiques),
        "mean_reward": sum(c.reward for c in critiques) / len(critiques),
        "action_match_rate": matches / len(critiques),
        "actions": {a: sum(1 for c in critiques if c.recovery_action == a) for a in RECOVERY_ACTIONS},
    }


def run_art_smoke(
    summary_path: Path,
    project: str = "physical-ai-workshop",
    weave_project: Optional[str] = None,
    dry_run: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """End-to-end pipeline that works without a GPU when ``dry_run=True``.

    When ``dry_run`` is False, the caller is expected to have a GPU and the
    ART LocalBackend will try to spin up vLLM. That path is left as an
    exercise for the workshop attendees — the skeleton below shows the
    integration point.
    """
    critiques = build_trajectories_from_summary(summary_path, synthetic=True)
    summary = summarise_critiques(critiques)

    if weave_project:
        try:
            import weave  # type: ignore
            from .weave_scorers import _ensure_weave_cache_dir

            _ensure_weave_cache_dir()
            weave.init(weave_project)

            @weave.op()
            def critique_rollout(payload: Dict[str, Any]) -> Dict[str, Any]:
                return payload

            for c in critiques:
                critique_rollout({
                    "episode_id": c.episode_id,
                    "reward": c.reward,
                    "recovery_action": c.recovery_action,
                    "response": c.response,
                })
        except ImportError:
            print("[art_critic] weave not installed; skipping trace publish")

    if dry_run:
        summary["mode"] = "dry_run"
    else:
        # Real ART training lives here. The workshop ships the harness and the
        # data pipeline; attendees pick the base model and launch on GPU.
        try:
            import art  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ART is not installed; pass dry_run=True") from exc
        trajectories = to_art_trajectories(critiques)
        summary["mode"] = "art_ready"
        summary["num_trajectories"] = len(trajectories)
        summary["note"] = (
            "Trajectories built; call model.train(art.TrajectoryGroup(trajectories)) "
            "on a GPU node with vLLM to actually update weights."
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "critiques": [
                {
                    "episode_id": c.episode_id,
                    "reward": c.reward,
                    "recovery_action": c.recovery_action,
                    "response": c.response,
                    "metadata": {
                        "failure_mode": c.rollout.get("failure_mode", "none"),
                    },
                }
                for c in critiques
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary["output_path"] = str(output_path)

    return summary
