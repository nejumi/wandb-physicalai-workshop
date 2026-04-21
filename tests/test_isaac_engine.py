"""Unit tests for IsaacTrainArgs and build_isaac_command.

Isaac Lab does not need to be installed to run these tests — we only exercise
the command-builder, not the actual training subprocess.
"""

import sys
from pathlib import Path

from physical_ai_workshop.isaac_engine import IsaacTrainArgs, build_isaac_command


def _joined(args: IsaacTrainArgs) -> str:
    return " ".join(build_isaac_command(args))


def test_default_command_contains_required_flags() -> None:
    cmd = build_isaac_command(IsaacTrainArgs())
    joined = " ".join(cmd)
    assert cmd[:3] == [sys.executable, "-m", "isaaclab_rl.scripts.rsl_rl.train"]
    assert "--task=Isaac-Reach-Franka-v0" in joined
    assert "--headless" in joined
    assert "--logger=wandb" in joined
    assert "--wandb_project=physical-ai-workshop" in joined
    assert "--video" in joined


def test_video_disabled_when_flag_set() -> None:
    args = IsaacTrainArgs(record_video=False)
    joined = _joined(args)
    assert "--video" not in joined
    assert "--video_interval" not in joined


def test_wandb_entity_only_added_when_set() -> None:
    without = _joined(IsaacTrainArgs(wandb_entity=None))
    with_entity = _joined(IsaacTrainArgs(wandb_entity="wandb-japan"))
    assert "--wandb_entity" not in without
    assert "--wandb_entity=wandb-japan" in with_entity


def test_num_envs_and_iterations_are_forwarded() -> None:
    joined = _joined(IsaacTrainArgs(num_envs=256, max_iterations=300))
    assert "--num_envs=256" in joined
    assert "--max_iterations=300" in joined


def test_extra_args_are_appended() -> None:
    args = IsaacTrainArgs(extra=["--device=cpu", "--disable_fabric=true"])
    cmd = build_isaac_command(args)
    assert cmd[-2] == "--device=cpu"
    assert cmd[-1] == "--disable_fabric=true"
