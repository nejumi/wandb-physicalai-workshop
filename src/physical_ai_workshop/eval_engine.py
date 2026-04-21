"""Offline evaluation for the workshop.

The evaluation job does three things that would be tedious to re-implement
for every attendee:

1. Load a LeRobot policy checkpoint with ``make_policy``.
2. Run a configurable number of rollouts on the PushT env and collect
   ``RolloutRecord`` objects (frames + per-step rewards + coverages).
3. Log the rollouts to W&B as videos/tables **and** stream them through
   Weave so attendees can inspect individual episodes from the Weave UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .env_utils import EnvSpec, make_pusht_env
from .rollout import RolloutRecord, run_rollouts


@dataclass
class EvalArgs:
    checkpoint_path: Path
    output_dir: Path
    num_episodes: int = 20
    seed_start: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    record_every: int = 2
    success_threshold: float = 0.95
    tag: str = "policy"
    wandb_project: Optional[str] = "physical-ai-workshop"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    weave_project: Optional[str] = None
    video_fps: int = 10


def _load_policy(checkpoint_path: Path, device: torch.device):
    """Load a LeRobot policy + its preprocessor/postprocessor pipelines.

    The processor pipelines are mandatory for correct inference.  Without the
    preprocessor the policy receives un-normalised observations (images in
    [0, 1] instead of mean-std centred); without the postprocessor the action
    output stays in the normalised [-1, 1] range instead of being mapped back
    to the environment action space (e.g. [0, 512] for PushT) — which makes
    the agent effectively stand still and eval success drops to 0 %.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(checkpoint_path))
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, cfg, preprocessor, postprocessor


def _aggregate_metrics(records: List[RolloutRecord]) -> Dict[str, float]:
    if not records:
        return {}
    successes = [int(r.success) for r in records]
    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean([r.reward_sum for r in records])),
        "avg_steps": float(np.mean([r.steps for r in records])),
        "avg_max_coverage": float(np.mean([r.max_coverage for r in records])),
        "avg_final_coverage": float(np.mean([r.final_coverage for r in records])),
        "num_episodes": float(len(records)),
    }


def _write_summary(output_dir: Path, records: List[RolloutRecord], metrics: Dict[str, float], tag: str) -> Path:
    payload = {
        "tag": tag,
        "metrics": metrics,
        "episodes": [r.summary() for r in records],
    }
    target = output_dir / "eval_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _encode_mp4(frames: List[np.ndarray], path: Path, fps: int) -> Optional[Path]:
    """Encode a list of ``(H, W, 3)`` uint8 frames to an mp4 via imageio.

    We intentionally keep a single code path (file-on-disk → ``wandb.Video``).
    Passing raw numpy to ``wandb.Video`` triggers a moviepy import which is
    not in the base ``wandb`` install.
    """
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[eval] imageio not installed; cannot encode mp4")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    return path


def _log_to_wandb(args: EvalArgs, records: List[RolloutRecord], metrics: Dict[str, float]) -> Optional[str]:
    try:
        import wandb
    except ImportError:
        print("[eval] wandb not installed; skipping remote logging")
        return None
    if not args.wandb_project:
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"eval-{args.tag}",
        job_type="evaluation",
        settings=wandb.Settings(init_timeout=300),
        config={
            "checkpoint": str(args.checkpoint_path),
            "num_episodes": args.num_episodes,
            "seed_start": args.seed_start,
            "tag": args.tag,
        },
    )
    run.summary.update({f"eval/{k}": v for k, v in metrics.items()})

    videos_dir = args.output_dir / "videos"
    video_paths: Dict[str, Optional[Path]] = {}
    for rec in records:
        target = videos_dir / f"{rec.episode_id}.mp4"
        video_paths[rec.episode_id] = _encode_mp4(rec.frames, target, args.video_fps)

    columns = ["episode_id", "seed", "success", "steps", "reward_sum", "max_coverage", "failure_mode", "video"]
    table = wandb.Table(columns=columns)
    for rec in records:
        video_path = video_paths.get(rec.episode_id)
        video = wandb.Video(str(video_path), format="mp4") if video_path else None
        table.add_data(
            rec.episode_id,
            rec.seed,
            rec.success,
            rec.steps,
            rec.reward_sum,
            rec.max_coverage,
            rec.failure_mode,
            video,
        )
    run.log({"eval/rollouts": table})

    failures = [r for r in records if not r.success]
    if failures:
        fail_columns = ["episode_id", "seed", "steps", "max_coverage", "failure_mode", "video"]
        fail_table = wandb.Table(columns=fail_columns)
        for rec in failures[:10]:
            video_path = video_paths.get(rec.episode_id)
            fail_table.add_data(
                rec.episode_id,
                rec.seed,
                rec.steps,
                rec.max_coverage,
                rec.failure_mode,
                wandb.Video(str(video_path), format="mp4") if video_path else None,
            )
        run.log({"eval/failures": fail_table})

    run_id = run.id
    run.finish()
    return run_id


def _log_to_weave(
    args: EvalArgs,
    records: List[RolloutRecord],
    wandb_run_id: Optional[str] = None,
    videos_dir: Optional[Path] = None,
) -> None:
    if not args.weave_project:
        return
    try:
        import weave
        from weave import Content
        from typing import Annotated
    except ImportError:
        print("[eval] weave not installed; skipping trace publishing")
        return

    from .weave_scorers import _ensure_weave_cache_dir
    _ensure_weave_cache_dir()
    client = weave.init(args.weave_project)

    @weave.op()
    def replay_rollout(episode: Dict[str, Any], video_bytes: Optional[bytes]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "episode_id": episode["episode_id"],
            "tag": episode["tag"],
            "success": episode["success"],
            "steps": episode["steps"],
            "max_coverage": episode["max_coverage"],
            "final_coverage": episode["final_coverage"],
            "failure_mode": episode["failure_mode"],
            "reward_sum": episode["reward_sum"],
        }
        if video_bytes is not None:
            result["video"] = Content.from_bytes(video_bytes, extension=".mp4")
        return result

    for i, rec in enumerate(records):
        if wandb_run_id:
            client.set_wandb_run_context(run_id=wandb_run_id, step=i)

        video_bytes: Optional[bytes] = None
        if videos_dir is not None:
            video_path = videos_dir / f"{rec.episode_id}.mp4"
            if video_path.exists():
                video_bytes = video_path.read_bytes()

        payload = rec.summary()
        payload["tag"] = args.tag
        replay_rollout(payload, video_bytes)

    if wandb_run_id:
        client.clear_wandb_run_context()


def run_evaluation(args: EvalArgs) -> Dict[str, Any]:
    device = torch.device(args.device)
    args.output_dir = Path(args.output_dir)
    args.checkpoint_path = Path(args.checkpoint_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading policy from {args.checkpoint_path}")
    policy, cfg, preprocessor, postprocessor = _load_policy(args.checkpoint_path, device)
    print(f"[eval] loaded policy type={cfg.type} + pre/postprocessor pipelines")

    env = make_pusht_env(EnvSpec())
    seeds = [args.seed_start + i for i in range(args.num_episodes)]
    print(f"[eval] running {len(seeds)} rollouts on device={device}")
    records = run_rollouts(
        env=env,
        policy=policy,
        seeds=seeds,
        device=device,
        tag=args.tag,
        record_every=args.record_every,
        success_threshold=args.success_threshold,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    env.close()

    metrics = _aggregate_metrics(records)
    summary_path = _write_summary(args.output_dir, records, metrics, args.tag)
    print(f"[eval] summary -> {summary_path}")
    print(f"[eval] metrics: {metrics}")

    run_id = _log_to_wandb(args, records, metrics)
    _log_to_weave(args, records, wandb_run_id=run_id, videos_dir=args.output_dir / "videos")

    return {
        "tag": args.tag,
        "metrics": metrics,
        "summary_path": str(summary_path),
        "num_records": len(records),
        "wandb_run_id": run_id,
    }
