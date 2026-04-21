"""Test the rollout loop with a fake env and fake policy — no GPU / gym needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from physical_ai_workshop.rollout import (
    RolloutRecord,
    _diagnose_failure,
    run_episode,
    run_rollouts,
)


@dataclass
class _FakePolicy:
    """Policy that returns a constant action tensor the env understands."""

    action_dim: int = 2
    reset_count: int = 0
    select_calls: int = 0

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, batch):
        self.select_calls += 1
        return torch.zeros((1, self.action_dim), dtype=torch.float32)


@dataclass
class _FakePushTEnv:
    """Minimal stand-in for gym-pusht.

    The env walks coverage up linearly until ``steps_until_done`` and then
    reports success. ``truncate_at`` lets tests force a timeout path.
    """

    steps_until_done: int = 10
    truncate_at: int | None = None
    final_coverage: float = 1.0
    render_shape: tuple = (4, 4, 3)
    _step: int = field(default=0, init=False)

    def reset(self, seed: int | None = None):
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        self._step += 1
        progress = min(self._step / self.steps_until_done, 1.0)
        coverage = float(progress * self.final_coverage)
        done = self._step >= self.steps_until_done
        truncated = bool(self.truncate_at is not None and self._step >= self.truncate_at)
        reward = coverage
        info = {"coverage": coverage, "is_success": done and self.final_coverage >= 0.95}
        return self._obs(), reward, done, truncated, info

    def render(self):
        return np.full(self.render_shape, self._step % 255, dtype=np.uint8)

    def _obs(self):
        return {
            "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
            "agent_pos": np.array([0.0, 0.0], dtype=np.float32),
        }


def test_run_episode_reaches_success() -> None:
    env = _FakePushTEnv(steps_until_done=5, final_coverage=1.0)
    policy = _FakePolicy()
    record = run_episode(env, policy, seed=0, episode_id="t-000", device=torch.device("cpu"))
    assert record.success is True
    assert record.steps == 5
    assert record.max_coverage == 1.0
    assert record.failure_mode == "none"
    assert policy.reset_count == 1
    assert policy.select_calls == 5


def test_run_episode_records_frames_every_n_steps() -> None:
    env = _FakePushTEnv(steps_until_done=8)
    policy = _FakePolicy()
    record = run_episode(
        env, policy, seed=0, episode_id="t-001",
        device=torch.device("cpu"), record_every=3,
    )
    # steps 0,3,6 recorded inline + final frame appended if not aligned
    assert len(record.frames) >= 3
    assert all(isinstance(f, np.ndarray) for f in record.frames)


def test_run_rollouts_produces_distinct_ids() -> None:
    env = _FakePushTEnv(steps_until_done=3)
    policy = _FakePolicy()
    records = run_rollouts(env, policy, seeds=[1, 2, 3], device=torch.device("cpu"), tag="demo")
    assert [r.episode_id for r in records] == ["demo-ep-000", "demo-ep-001", "demo-ep-002"]
    assert [r.seed for r in records] == [1, 2, 3]


def test_diagnose_failure_classifies_modes() -> None:
    timeout = RolloutRecord(
        episode_id="x", seed=0, success=False, max_coverage=0.5,
        final_coverage=0.5, reward_sum=0.0, steps=300,
    )
    near_miss = RolloutRecord(
        episode_id="x", seed=0, success=False, max_coverage=0.92,
        final_coverage=0.92, reward_sum=0.0, steps=100,
    )
    no_contact = RolloutRecord(
        episode_id="x", seed=0, success=False, max_coverage=0.02,
        final_coverage=0.02, reward_sum=0.0, steps=50,
    )
    unstable = RolloutRecord(
        episode_id="x", seed=0, success=False, max_coverage=0.5,
        final_coverage=0.4, reward_sum=0.0, steps=50,
    )

    assert _diagnose_failure(timeout) == "timeout"
    assert _diagnose_failure(near_miss) == "near_miss"
    assert _diagnose_failure(no_contact) == "no_contact"
    assert _diagnose_failure(unstable) == "unstable_push"


def test_record_summary_is_json_friendly() -> None:
    env = _FakePushTEnv(steps_until_done=4)
    policy = _FakePolicy()
    record = run_episode(env, policy, seed=7, episode_id="t-x", device=torch.device("cpu"))
    payload = record.summary()
    assert set(payload) == {
        "episode_id", "seed", "success", "max_coverage",
        "final_coverage", "reward_sum", "steps", "failure_mode",
    }
    # all leaf values json-serializable
    import json
    json.dumps(payload)
