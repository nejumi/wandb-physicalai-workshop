"""Promote a trained policy checkpoint into W&B Models.

The registry integration is deliberately small: we upload the ``pretrained_model``
directory as a W&B Artifact, optionally link it into a Registry collection and
record the eval metrics on the artifact so reviewers see how the model performed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PromoteArgs:
    checkpoint_path: Path
    eval_summary_path: Optional[Path] = None
    artifact_name: str = "pusht-diffusion"
    artifact_type: str = "model"
    wandb_project: str = "physical-ai-workshop"
    wandb_entity: Optional[str] = None
    registry_collection: Optional[str] = None  # e.g. "wandb-registry-model/physical-ai-workshop"
    alias: str = "candidate"
    description: str = ""


def _load_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def promote_checkpoint(args: PromoteArgs) -> Dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wandb must be installed to promote a model") from exc

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    summary = _load_summary(args.eval_summary_path)
    metrics = summary.get("metrics", {})

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"promote-{args.artifact_name}",
        job_type="model-promotion",
        settings=wandb.Settings(init_timeout=300),
        config={
            "checkpoint_path": str(checkpoint_path),
            "alias": args.alias,
            "registry_collection": args.registry_collection,
        },
    )

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.description or f"PushT policy promoted from {checkpoint_path.name}",
        metadata={"metrics": metrics, "tag": summary.get("tag")},
    )
    artifact.add_dir(str(checkpoint_path))
    logged = run.log_artifact(artifact, aliases=[args.alias])

    offline = bool(getattr(run.settings, "_offline", False))
    version: Optional[str] = None
    linked_to: Optional[str] = None
    if offline:
        print("[registry] offline mode — skipping artifact.wait()/link()")
    else:
        logged.wait()
        version = logged.version
        if args.registry_collection:
            try:
                logged.link(args.registry_collection, aliases=[args.alias])
                linked_to = args.registry_collection
            except Exception as exc:  # pragma: no cover - network/permission dependent
                print(f"[registry] link to {args.registry_collection} failed: {exc}")

    run.summary.update({f"promoted/{k}": v for k, v in metrics.items()})
    run.finish()

    return {
        "artifact_name": args.artifact_name,
        "artifact_version": version,
        "alias": args.alias,
        "registry_collection": linked_to,
        "metrics": metrics,
        "offline": offline,
    }
