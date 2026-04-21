"""Command-builder tests for TrainArgs → lerobot_train argv."""

import sys
from pathlib import Path

from physical_ai_workshop.train_engine import TrainArgs, build_command


def test_build_command_defaults_emit_lerobot_flags() -> None:
    cmd = build_command(TrainArgs(output_dir=Path("runs/demo")))
    joined = " ".join(cmd)
    assert cmd[:3] == [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    assert "--policy.type=diffusion" in joined
    assert "--policy.push_to_hub=false" in joined
    assert "--dataset.repo_id=lerobot/pusht" in joined
    assert "--dataset.video_backend=pyav" in joined
    assert "--env.type=pusht" in joined
    assert "--output_dir=runs/demo" in joined
    assert "--wandb.enable=true" in joined


def test_build_command_disables_wandb_when_flag_false() -> None:
    cmd = build_command(TrainArgs(output_dir=Path("x"), wandb_enable=False))
    assert "--wandb.enable=false" in " ".join(cmd)
    assert not any(c.startswith("--wandb.mode=") for c in cmd)


def test_build_command_passes_optional_wandb_knobs() -> None:
    cmd = build_command(TrainArgs(
        output_dir=Path("x"),
        wandb_mode="offline",
        wandb_entity="acme",
        run_name="nightly-1",
    ))
    joined = " ".join(cmd)
    assert "--wandb.mode=offline" in joined
    assert "--wandb.entity=acme" in joined
    assert "--job_name=nightly-1" in joined


def test_build_command_appends_extra_args_verbatim() -> None:
    cmd = build_command(TrainArgs(
        output_dir=Path("x"),
        extra=["--policy.n_action_steps=4", "--optimizer.lr=5e-5"],
    ))
    assert cmd[-2] == "--policy.n_action_steps=4"
    assert cmd[-1] == "--optimizer.lr=5e-5"
