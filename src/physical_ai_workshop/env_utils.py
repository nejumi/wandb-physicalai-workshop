"""PushT environment helpers.

Thin wrappers around ``gym_pusht`` that keep workshop scripts short. We expose
two utilities:

- ``make_pusht_env`` returns a single ``gymnasium`` env configured for either
  visual rollouts (render high-resolution frames for W&B videos) or fast
  headless evaluation.
- ``reset_with_seed`` resets an env reproducibly and returns the initial
  observation together with the first high-res frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the PushT env with gymnasium)
import numpy as np


DEFAULT_ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class EnvSpec:
    env_id: str = DEFAULT_ENV_ID
    obs_type: str = "pixels_agent_pos"
    visualization_width: int = 384
    visualization_height: int = 384
    observation_width: int = 96
    observation_height: int = 96
    max_episode_steps: int = 300


def make_pusht_env(spec: EnvSpec | None = None) -> gym.Env:
    """Create a PushT gymnasium env with visualization enabled."""
    spec = spec or EnvSpec()
    env = gym.make(
        spec.env_id,
        obs_type=spec.obs_type,
        render_mode="rgb_array",
        visualization_width=spec.visualization_width,
        visualization_height=spec.visualization_height,
        observation_width=spec.observation_width,
        observation_height=spec.observation_height,
        max_episode_steps=spec.max_episode_steps,
    )
    return env


def reset_with_seed(env: gym.Env, seed: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Reset env with a given seed and also return the first render frame."""
    obs, _info = env.reset(seed=int(seed))
    frame = env.render()
    return obs, frame
