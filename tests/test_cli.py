"""Exercise the argparse surface — catches regressions in flag names."""

import pytest

from physical_ai_workshop.cli import build_parser


@pytest.mark.parametrize("command,required", [
    ("train", ["--output-dir", "x"]),
    ("eval", ["--checkpoint", "c", "--output-dir", "x"]),
    ("promote", ["--checkpoint", "c"]),
    ("weave-eval", ["--eval-summary", "s", "--weave-project", "p"]),
])
def test_subcommands_accept_minimal_args(command: str, required: list[str]) -> None:
    parser = build_parser()
    args = parser.parse_args([command, *required])
    assert args.command == command


def test_train_defaults_enable_wandb() -> None:
    args = build_parser().parse_args(["train", "--output-dir", "x"])
    assert args.wandb_enable is True


def test_train_no_wandb_disables() -> None:
    args = build_parser().parse_args(["train", "--output-dir", "x", "--no-wandb"])
    assert args.wandb_enable is False


def test_compare_accepts_run_paths() -> None:
    args = build_parser().parse_args([
        "compare",
        "--left-run", "ent/proj/r1",
        "--right-run", "ent/proj/r2",
    ])
    assert args.left_run == "ent/proj/r1"
    assert args.right_run == "ent/proj/r2"
