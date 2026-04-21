"""Unified CLI for the workshop.

Most attendees only need the four commands below. They map 1:1 to the short
demo scripts under ``scripts/`` so the instructor can either run ``scripts/XX_*.py``
in a notebook or invoke the Python CLI directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare_candidates, from_summary, from_wandb_run
from .eval_engine import EvalArgs, run_evaluation
from .registry import PromoteArgs, promote_checkpoint
from .train_engine import TrainArgs, latest_checkpoint, run_training
from .weave_scorers import run_weave_evaluation


def cmd_train(args: argparse.Namespace) -> int:
    extra = args.extra or []
    targs = TrainArgs(
        policy=args.policy,
        dataset_repo_id=args.dataset,
        env_type=args.env,
        output_dir=Path(args.output_dir),
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        wandb_enable=args.wandb_enable,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        extra=extra,
    )
    output_dir = run_training(targs)
    ckpt = latest_checkpoint(output_dir)
    print(f"[cli] checkpoint ready: {ckpt}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    eargs = EvalArgs(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        device=args.device,
        record_every=args.record_every,
        success_threshold=args.success_threshold,
        tag=args.tag,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        weave_project=args.weave_project,
        video_fps=args.video_fps,
    )
    result = run_evaluation(eargs)
    print(json.dumps(result, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if args.left_run:
        left = from_wandb_run(args.left_run, label=args.left_label or "left")
    else:
        left = from_summary(args.left, label=args.left_label or "left")
    if args.right_run:
        right = from_wandb_run(args.right_run, label=args.right_label or "right")
    else:
        right = from_summary(args.right, label=args.right_label or "right")

    markdown, deltas = compare_candidates(left, right)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(deltas, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    pargs = PromoteArgs(
        checkpoint_path=Path(args.checkpoint),
        eval_summary_path=Path(args.eval_summary) if args.eval_summary else None,
        artifact_name=args.artifact_name,
        artifact_type=args.artifact_type,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        registry_collection=args.registry_collection,
        alias=args.alias,
        description=args.description,
    )
    result = promote_checkpoint(pargs)
    print(json.dumps(result, indent=2))
    return 0


def cmd_weave_eval(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.eval_summary).read_text(encoding="utf-8"))
    episodes = summary.get("episodes", [])
    result = run_weave_evaluation(
        project=args.weave_project,
        dataset_name=args.dataset_name,
        episodes=episodes,
        policy_name=args.policy_name,
        artifact_id=getattr(args, "artifact_id", ""),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physical-ai-workshop")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a LeRobot policy on PushT.")
    p_train.add_argument("--policy", default="diffusion", choices=["diffusion", "act"])
    p_train.add_argument("--dataset", default="lerobot/pusht")
    p_train.add_argument("--env", default="pusht")
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--steps", type=int, default=5000)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--eval-freq", type=int, default=1000)
    p_train.add_argument("--save-freq", type=int, default=5000)
    p_train.add_argument("--log-freq", type=int, default=50)
    p_train.add_argument("--wandb-enable", dest="wandb_enable", action="store_true", default=True)
    p_train.add_argument("--no-wandb", dest="wandb_enable", action="store_false")
    p_train.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    p_train.add_argument("--wandb-project", default="physical-ai-workshop")
    p_train.add_argument("--wandb-entity", default=None)
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--extra", nargs=argparse.REMAINDER)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a trained checkpoint with rollouts.")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--output-dir", required=True)
    p_eval.add_argument("--num-episodes", type=int, default=20)
    p_eval.add_argument("--seed-start", type=int, default=1000)
    p_eval.add_argument("--device", default="cuda")
    p_eval.add_argument("--record-every", type=int, default=2)
    p_eval.add_argument("--success-threshold", type=float, default=0.95)
    p_eval.add_argument("--tag", default="policy")
    p_eval.add_argument("--wandb-project", default="physical-ai-workshop")
    p_eval.add_argument("--wandb-entity", default=None)
    p_eval.add_argument("--wandb-run-name", default=None)
    p_eval.add_argument("--weave-project", default=None)
    p_eval.add_argument("--video-fps", type=int, default=10)
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare", help="Compare two eval summaries or W&B runs.")
    p_cmp.add_argument("--left", help="Path to left eval_summary.json")
    p_cmp.add_argument("--right", help="Path to right eval_summary.json")
    p_cmp.add_argument("--left-run", help="W&B run path, e.g. entity/project/run_id")
    p_cmp.add_argument("--right-run", help="W&B run path")
    p_cmp.add_argument("--left-label", default=None)
    p_cmp.add_argument("--right-label", default=None)
    p_cmp.add_argument("--output", help="Markdown output path")
    p_cmp.add_argument("--json-output", help="JSON delta output path")
    p_cmp.set_defaults(func=cmd_compare)

    p_pro = sub.add_parser("promote", help="Upload a checkpoint to W&B Models.")
    p_pro.add_argument("--checkpoint", required=True)
    p_pro.add_argument("--eval-summary", default=None)
    p_pro.add_argument("--artifact-name", default="pusht-diffusion")
    p_pro.add_argument("--artifact-type", default="model")
    p_pro.add_argument("--wandb-project", default="physical-ai-workshop")
    p_pro.add_argument("--wandb-entity", default=None)
    p_pro.add_argument("--registry-collection", default=None)
    p_pro.add_argument("--alias", default="candidate")
    p_pro.add_argument("--description", default="")
    p_pro.set_defaults(func=cmd_promote)

    p_weave = sub.add_parser("weave-eval", help="Run a Weave Evaluation on rollouts.")
    p_weave.add_argument("--eval-summary", required=True)
    p_weave.add_argument("--weave-project", required=True)
    p_weave.add_argument("--dataset-name", default="pusht-rollouts")
    p_weave.add_argument("--policy-name", default="diffusion")
    p_weave.add_argument("--artifact-id", default="", help="wandb-artifact:/// URI for the model artifact")
    p_weave.set_defaults(func=cmd_weave_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
