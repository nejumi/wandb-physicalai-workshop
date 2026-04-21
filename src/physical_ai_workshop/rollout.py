"""Policy rollout loop for PushT.

This is the heart of the workshop's evaluation story. For each episode we:

1. reset the environment with a deterministic seed;
2. query the LeRobot policy once per step with a batched observation tensor;
3. record the high-res render frame so the instructor can replay the rollout
   as a W&B Video; and
4. return a ``RolloutRecord`` that contains everything needed for comparison,
   Weave tracing and failure analysis.

The observation pre-processing follows the LeRobot convention:

- ``observation.image`` is ``(B, C, H, W)`` with values in ``[0, 1]``
- ``observation.state`` is ``(B, D)`` float32 (the 2-D agent position)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class RolloutRecord:
    episode_id: str
    seed: int
    success: bool
    max_coverage: float
    final_coverage: float
    reward_sum: float
    steps: int
    frames: List[np.ndarray] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)
    coverages: List[float] = field(default_factory=list)
    failure_mode: str = "none"

    def summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "success": bool(self.success),
            "max_coverage": float(self.max_coverage),
            "final_coverage": float(self.final_coverage),
            "reward_sum": float(self.reward_sum),
            "steps": int(self.steps),
            "failure_mode": self.failure_mode,
        }


def _observation_to_batch(
    obs: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    image = obs["pixels"].astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    state = obs["agent_pos"].astype(np.float32)
    return {
        "observation.image": torch.from_numpy(image).unsqueeze(0).to(device),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }


def _diagnose_failure(record: RolloutRecord) -> str:
    if record.success:
        return "none"
    if record.max_coverage < 0.1:
        return "no_contact"
    if record.max_coverage >= 0.9 and not record.success:
        return "near_miss"
    if record.steps >= 280:
        return "timeout"
    return "unstable_push"


def run_episode(
    env,
    policy,
    seed: int,
    episode_id: str,
    device: torch.device,
    record_every: int = 1,
    success_threshold: float = 0.95,
    preprocessor=None,
    postprocessor=None,
) -> RolloutRecord:
    """Run one episode with ``policy`` and return a RolloutRecord.

    If ``preprocessor``/``postprocessor`` are provided (the LeRobot processor
    pipelines returned from ``make_pre_post_processors``) they are applied
    around ``policy.select_action``. This is required for correct inference:
    the preprocessor normalises the observation (image mean/std, state min/max)
    and the postprocessor unnormalises the action back to the env's pixel
    coordinates (e.g. [0, 512] for PushT). Without them, actions stay in the
    [-1, 1] normalised range and success rate collapses to 0 %.
    """
    obs, _info = env.reset(seed=int(seed))
    policy.reset()

    frames: List[np.ndarray] = []
    step_rewards: List[float] = []
    coverages: List[float] = []
    reward_sum = 0.0
    max_coverage = 0.0
    final_coverage = 0.0
    success = False
    steps = 0

    done = False
    truncated = False
    while not (done or truncated):
        batch = _observation_to_batch(obs, device)
        if preprocessor is not None:
            batch["task"] = [""]
            batch = preprocessor(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        if postprocessor is not None:
            action = postprocessor(action)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        obs, reward, done, truncated, info = env.step(action_np)

        reward = float(reward)
        coverage = float(info.get("coverage", reward))
        reward_sum += reward
        max_coverage = max(max_coverage, coverage)
        final_coverage = coverage
        step_rewards.append(reward)
        coverages.append(coverage)
        if steps % record_every == 0:
            frames.append(env.render())
        if info.get("is_success", coverage >= success_threshold):
            success = True
        steps += 1

    if not frames or (steps - 1) % record_every != 0:
        frames.append(env.render())

    record = RolloutRecord(
        episode_id=episode_id,
        seed=int(seed),
        success=bool(success),
        max_coverage=max_coverage,
        final_coverage=final_coverage,
        reward_sum=reward_sum,
        steps=steps,
        frames=frames,
        step_rewards=step_rewards,
        coverages=coverages,
    )
    record.failure_mode = _diagnose_failure(record)
    return record


def run_rollouts(
    env,
    policy,
    seeds: List[int],
    device: torch.device,
    tag: str = "policy",
    **episode_kwargs,
) -> List[RolloutRecord]:
    records: List[RolloutRecord] = []
    for index, seed in enumerate(seeds):
        records.append(
            run_episode(
                env=env,
                policy=policy,
                seed=seed,
                episode_id=f"{tag}-ep-{index:03d}",
                device=device,
                **episode_kwargs,
            )
        )
    return records
