#!/usr/bin/env python3
"""Append Isaac Lab checkpoint and video to the native W&B run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb


def newest(paths: list[Path], kind: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"Isaac Lab {kind} was not produced")
    return max(paths, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    checkpoint = newest(list(args.log_root.rglob("model_*.pt")), "checkpoint")
    video = newest(list(args.log_root.rglob("*.mp4")), "video")
    with wandb.init(
        id=args.run_id, resume="allow", project=args.project,
        entity=os.getenv("WANDB_ENTITY") or None,
    ) as run:
        artifact = wandb.Artifact(
            "isaaclab-cartpole-policy", type="model",
            description="RSL-RL checkpoint from Isaac-Cartpole",
            metadata={"task": "Isaac-Cartpole", "rl_library": "rsl_rl"},
        )
        artifact.add_file(str(checkpoint), name=checkpoint.name)
        run.log_artifact(artifact, aliases=["latest"])
        run.log({"eval/simulation": wandb.Video(str(video), format="mp4")})
        run.summary["logging_contract"] = {
            "metrics": "native Isaac Lab / RSL-RL metrics",
            "artifact": "isaaclab-cartpole-policy",
            "video": "eval/simulation",
        }
        mode = run.settings.mode
        run_info = {
            "run_id": args.run_id,
            "mode": mode,
            "url": run.url if mode == "online" else None,
            "project": run.project,
            "entity": run.entity,
        }

    # Persist after finish so the notebook can display the completed run independently of subprocess stdout.
    (args.log_root / "wandb-run.json").write_text(json.dumps(run_info, indent=2))
    if run_info["url"]:
        print(f"W&B Run: {run_info['url']}", flush=True)
    else:
        print(f"W&B mode={mode}: no online Run URL was created.", flush=True)


if __name__ == "__main__":
    main()
