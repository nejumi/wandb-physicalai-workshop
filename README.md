# Physical AI W&B Workshop

## Colab T4 quickstart（短時間講座）

週明けの入門講座では、次の Notebook を上から順に実行します。Colab の
`ランタイム > ランタイムのタイプを変更 > T4 GPU` を選択してください。

| Lab | Colab | 所要時間 | W&B へ残るもの |
|---|---|---:|---|
| LeRobot tiny behavior cloning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nejumi/wandb-physicalai-workshop/blob/main/notebooks/01_lerobot_wandb_quickstart.ipynb) | 15–20分 | loss、model Artifact、rollout video |
| Isaac Lab Cartpole PPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nejumi/wandb-physicalai-workshop/blob/main/notebooks/02_isaac_lab_wandb_quickstart.ipynb) | 20–25分 | reward / loss、checkpoint Artifact、simulation video |

LeRobot Lab は外部データや実機なしで完走します。Isaac Lab Lab は GUI 版 Isaac
Sim ではなく standalone Newton backend を使う構成です。
両 Lab とも metric / checkpoint / video を同じ W&B Run にまとめます。

検証範囲: ローカル TITAN RTX でスクリプトを実行済みです。
Google Colab T4 実環境での全セル完走は未確認です。CI は CPU テストであり、
Colab や GPU の動作を保証するものではありません。
Isaac Lab はカーネルの Python と分離した Python 3.12 環境で実行します。

最後に [wandb/ART-Embodied の紹介](docs/art_embodied_intro.md) を5〜10分行います。
より長い H100 / cluster 向け教材は、この README の後半にそのまま残しています。

End-to-end workshop that trains a LeRobot policy on PushT, evaluates it with
rollout videos, compares two candidates, and promotes the best one into W&B
Models — with rollout traces streamed into W&B Weave.

The pipeline uses real LeRobot 0.4.4 training under the hood. Nothing is mocked.

## What attendees walk away with

- A Diffusion Policy (and optionally an ACT) checkpoint trained on the
  `lerobot/pusht` dataset, logged to W&B.
- An evaluation W&B run with per-episode rollout videos, a failures table, and
  aggregate metrics.
- A Weave project with per-episode traces plus four scorers
  (`success`, `coverage`, `efficiency`, `failure_mode`) attached to a
  `weave.Evaluation`.
- A W&B Artifact and optional Registry-collection link for the winning policy.
- A markdown comparison of any two candidates (local JSON or live W&B runs).

## Repo layout

```
scripts/                    thin CLIs that call the Python package below
  01_train_diffusion_pusht.py   LeRobot training (Diffusion Policy)
  02_train_act_pusht.py         LeRobot training (ACT)
  03_evaluate.py                rollouts + W&B video + Weave traces
  04_compare.py                 two-candidate comparison
  05_promote_to_registry.py     W&B Artifact + Registry
  06_weave_evaluation.py        weave.Evaluation with 4 scorers
  07_art_critic.py              OpenPipe ART rollout critic
  08_isaac_lab_train.py         Isaac Lab PPO on Franka Reach (optional)
slurm/                      SBATCH wrappers for an H100 cluster
src/physical_ai_workshop/
  cli.py                    unified argparse CLI (train / eval / compare / promote / weave-eval)
  train_engine.py           wrapper around lerobot.scripts.lerobot_train
  eval_engine.py            rollouts + W&B video logging + Weave traces
  compare.py                candidate comparison (local JSON or W&B run path)
  registry.py               artifact upload + optional registry link
  weave_scorers.py          weave.Model + weave.Evaluation scorers
  art_critic.py             OpenPipe ART agentic extension over rollouts
  isaac_engine.py           Isaac Lab subprocess wrapper
  rollout.py                PushT rollout loop + failure-mode diagnosis
  env_utils.py              PushT gym env helpers
tests/                      pytest suite (no GPU / network needed)
docs/
  isaac_lab_setup.md        NGC container + W&B setup for script 08
```

## Environment

Tested on a CoreWeave H100 node with:

- Python 3.11
- PyTorch 2.9.0 + CUDA 12.8
- LeRobot 0.4.4 (installed with extras `pusht`)
- `ffmpeg` 5 or 6 from conda-forge (needed for `imageio[ffmpeg]`)
- `wandb >= 0.19`, `weave >= 0.51`

Install the Python package in editable mode with the relevant extras:

```bash
pip install -e ".[wandb,weave,lerobot]"
```

LeRobot pulls in `torchcodec` for video decoding. The workshop pipeline
passes `--dataset.video_backend=pyav` so attendees never have to debug the
torchcodec/ffmpeg ABI matrix.

## Quick start on H100

```bash
# 1. Train Diffusion Policy (~3 min for 200 steps on a single H100)
python scripts/01_train_diffusion_pusht.py \
    --output-dir workspace/runs/smoke \
    --run-name smoke \
    --steps 200 --eval-freq 0 --save-freq 200 --batch-size 32

# 2. Evaluate the checkpoint on 20 rollouts
python scripts/03_evaluate.py \
    --checkpoint workspace/runs/smoke/checkpoints/last/pretrained_model \
    --output-dir workspace/runs/smoke/eval \
    --num-episodes 20 --record-every 2 --tag smoke

# 3. Compare two candidates
python scripts/04_compare.py \
    --left workspace/runs/baseline/eval/eval_summary.json \
    --right workspace/runs/candidate/eval/eval_summary.json \
    --output workspace/reports/compare.md

# 4. Promote the winner into W&B Models
python scripts/05_promote_to_registry.py \
    --checkpoint workspace/runs/candidate/checkpoints/last/pretrained_model \
    --eval-summary workspace/runs/candidate/eval/eval_summary.json \
    --artifact-name pusht-diffusion \
    --alias candidate

# 5. Run a Weave Evaluation over the recorded rollouts
python scripts/06_weave_evaluation.py \
    --eval-summary workspace/runs/candidate/eval/eval_summary.json \
    --weave-project physical-ai-workshop

# 6. Optional — ART rollout critic (dry-run is CPU-only)
python scripts/07_art_critic.py \
    --eval-summary workspace/runs/candidate/eval/eval_summary.json \
    --output workspace/runs/candidate/critic/summary.json \
    --weave-project physical-ai-workshop
```

## Isaac Lab (optional, NVIDIA stack)

`scripts/08_isaac_lab_train.py` is the counterpart demo for NVIDIA Isaac Lab —
PPO on Franka Reach, 4096 parallel envs on a single H100, with the same W&B
project and `--video` logging the Isaac Lab rsl_rl integration produces.

```bash
python scripts/08_isaac_lab_train.py \
    --task Isaac-Reach-Franka-v0 \
    --num-envs 4096 \
    --max-iterations 1000 \
    --wandb-project physical-ai-workshop \
    --video --video-interval 500
```

See [docs/isaac_lab_setup.md](docs/isaac_lab_setup.md) for the NGC container
setup and the full flag list.

## SLURM

`slurm/train_diffusion.sbatch`, `slurm/train_act.sbatch`, and
`slurm/evaluate.sbatch` wrap the training and evaluation scripts for a
single-GPU H100 partition. Set `OUTPUT_DIR`, `CHECKPOINT`, and the W&B env
vars, then `sbatch slurm/train_diffusion.sbatch`.

## W&B surfaces exercised

- **Runs** — training and evaluation both create standard W&B runs with
  config, metrics, and summary keys.
- **Tables + Video** — evaluation logs `eval/rollouts` as a `wandb.Table` of
  per-episode MP4s; failing episodes get a dedicated `eval/failures` table.
- **Artifacts + Registry** — the promote step uploads the `pretrained_model`
  directory as a model artifact and optionally links it into a Registry
  collection (e.g. `wandb-registry-model/physical-ai-workshop`).
- **Weave traces** — every evaluated episode produces a `replay_rollout` op
  call; the optional `weave-eval` step runs a `weave.Evaluation` with four
  scorers over the recorded rollouts.

## Tests

```bash
pip install -e .[dev]
pytest
```

The suite (39 tests, <5 s) exercises the rollout loop, compare/registry
logic, CLI surface, ART critic pipeline, and Isaac Lab command builder
without needing a GPU, the PushT env, or network.

## Offline smoke testing

The whole pipeline works with `WANDB_MODE=offline` and `--wandb-mode offline`
on the train step — useful for an air-gapped demo node. Weave still requires
network access for its backend.
