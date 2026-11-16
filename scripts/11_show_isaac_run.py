#!/usr/bin/env python3
"""Read a completed Isaac quickstart run's URL without training or creating a run."""

import argparse
import json
import os
from pathlib import Path


def get_run_info(isaaclab: Path, run_id: str | None = None) -> dict:
    root = isaaclab / "logs" / "rsl_rl" / "wandb-course-cartpole"
    if run_id:
        log_dir = root / run_id
        if not log_dir.is_dir():
            raise FileNotFoundError(f"No Isaac training directory: {log_dir}")
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
        if not candidates:
            raise FileNotFoundError(f"No Isaac training runs found in {root}. Run the training cell first.")
        log_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        run_id = log_dir.name

    manifest = log_dir / "wandb-run.json"
    if manifest.is_file():
        return json.loads(manifest.read_text())

    # Recover runs made before the manifest was introduced, without uploading anything.
    records = list((isaaclab / "wandb").glob(f"*run-*-{run_id}"))
    latest = max(records, key=lambda p: p.stat().st_mtime) if records else None
    if latest is not None and latest.name.startswith("offline-run-"):
        return {"run_id": run_id, "mode": "offline", "url": None}

    import wandb

    api = wandb.Api()
    entity = os.getenv("WANDB_ENTITY") or api.default_entity
    project = os.getenv("WANDB_PROJECT", "physical-ai-workshop")
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY to the team or user that owns this run.")
    run = api.run(f"{entity}/{project}/{run_id}")
    return {"run_id": run_id, "mode": "online", "url": run.url}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(get_run_info(args.isaaclab, args.run_id)))
