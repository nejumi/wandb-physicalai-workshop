#!/usr/bin/env python3
"""Colab-sized LeRobotDataset behavior cloning with W&B metrics/artifact/video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


ROOT = Path(__file__).resolve().parents[1]


def expert_action(position: np.ndarray, goal: np.ndarray, max_step: float = 0.12) -> np.ndarray:
    delta = goal - position
    norm = float(np.linalg.norm(delta))
    if norm <= max_step:
        return delta.astype(np.float32)
    return (delta / norm * max_step).astype(np.float32)


def create_dataset(root: Path, episodes: int, horizon: int, seed: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        shutil.rmtree(root)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "goal_x", "goal_y"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["delta_x", "delta_y"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="wandb-course/tiny-reach", root=root, fps=10, features=features
    )
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        position = rng.uniform(-0.9, 0.9, size=2).astype(np.float32)
        goal = rng.uniform(-0.7, 0.7, size=2).astype(np.float32)
        for _ in range(horizon):
            action = expert_action(position, goal)
            dataset.add_frame(
                {
                    "observation.state": np.concatenate([position, goal]).astype(np.float32),
                    "action": action,
                    "task": "Move the blue dot to the red target",
                }
            )
            position = position + action
        dataset.save_episode()
    dataset.finalize()


def load_tensors(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="wandb-course/tiny-reach", root=root)
    states = torch.stack([dataset[i]["observation.state"] for i in range(len(dataset))]).float()
    actions = torch.stack([dataset[i]["action"] for i in range(len(dataset))]).float()
    return states, actions


def rollout_video(model: torch.nn.Module, output: Path, seed: int) -> float:
    rng = np.random.default_rng(seed + 1000)
    position = rng.uniform(-0.9, -0.4, size=2).astype(np.float32)
    goal = rng.uniform(0.4, 0.8, size=2).astype(np.float32)
    frames: list[np.ndarray] = []
    trajectory = [position.copy()]
    model.eval()

    for _ in range(25):
        state = torch.from_numpy(np.concatenate([position, goal])).float()
        with torch.no_grad():
            action = model(state).clamp(-0.12, 0.12).numpy()
        position = np.clip(position + action, -1.0, 1.0)
        trajectory.append(position.copy())
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        points = np.asarray(trajectory)
        ax.plot(points[:, 0], points[:, 1], color="#2F80ED", linewidth=2)
        ax.scatter(*position, s=120, color="#2F80ED", label="robot")
        ax.scatter(*goal, s=180, marker="*", color="#EB5757", label="target")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="x", ylabel="y", title="LeRobot tiny reach")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=10, codec="libx264", quality=7)
    return float(np.linalg.norm(goal - position))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "quickstart" / "lerobot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = args.output_dir / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = args.output_dir / "dataset"
    create_dataset(dataset_root, args.episodes, args.horizon, args.seed)
    states, actions = load_tensors(dataset_root)
    states, actions = states.to(device), actions.to(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    init_kwargs = {
        "project": os.getenv("WANDB_PROJECT", "physical-ai-workshop"),
        "entity": os.getenv("WANDB_ENTITY") or None,
        "name": "lerobot-tiny-bc",
        "job_type": "train",
        "tags": ["course", "colab-t4", "lerobot", "behavior-cloning"],
        "config": vars(args) | {"output_dir": str(args.output_dir), "device": str(device)},
    }
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode

    with wandb.init(**init_kwargs) as run:
        run.define_metric("trainer/step")
        run.define_metric("train/*", step_metric="trainer/step")
        for step in range(1, args.steps + 1):
            indices = torch.randint(len(states), (min(64, len(states)),), device=device)
            loss = torch.nn.functional.mse_loss(model(states[indices]), actions[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run.log({"trainer/step": step, "train/loss": float(loss.detach())})

        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output_dir / "policy.pt"
        torch.save({"model_state_dict": model.state_dict(), "input_dim": 4, "output_dim": 2}, checkpoint)
        artifact = wandb.Artifact(
            "lerobot-tiny-bc-policy", type="model",
            description="Tiny behavior-cloning policy trained from a LeRobotDataset",
            metadata={"framework": "pytorch", "dataset_format": "LeRobotDataset"},
        )
        artifact.add_file(str(checkpoint), name="policy.pt")
        run.log_artifact(artifact, aliases=["latest"])

        video_path = args.output_dir / "rollout.mp4"
        final_distance = rollout_video(model.cpu(), video_path, args.seed)
        run.log({
            "eval/final_distance": final_distance,
            "eval/rollout": wandb.Video(str(video_path), format="mp4"),
        })
        run.summary["logging_contract"] = {
            "metrics": ["train/loss", "eval/final_distance"],
            "artifact": "lerobot-tiny-bc-policy",
            "video": "eval/rollout",
        }
        run_url = run.url

    print(f"completed: {run_url or 'offline run'}")
    print(f"final distance: {final_distance:.4f}")


if __name__ == "__main__":
    main()

