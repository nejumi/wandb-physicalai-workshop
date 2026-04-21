#!/usr/bin/env bash
# Full pipeline: train → eval → promote → weave-eval → art-critic
# Usage: bash run_pipeline.sh [steps]
set -euo pipefail

PYTHON=/mnt/home/yyamamoto+cwb607/.mamba/envs/aimo3-art/bin/python3.11
STEPS=${1:-50000}
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT/workspace/runs/diffusion-${STEPS}k"
ENTITY="wandb-japan"
PROJECT="physical-ai-workshop"
WEAVE_PROJECT="wandb-japan/physical-ai-workshop"
ARTIFACT_NAME="pusht-diffusion"
TAG="diffusion-${STEPS}k"
DATE=$(date +%Y%m%d-%H%M)

export WEAVE_SERVER_CACHE_DIR="$ROOT/workspace/.weave-cache"
mkdir -p "$WEAVE_SERVER_CACHE_DIR"

echo "=========================================="
echo "  Physical AI Workshop Pipeline"
echo "  steps=$STEPS  tag=$TAG"
echo "  $(date)"
echo "=========================================="

# ------------------------------------------------------------------
# 1. TRAIN
# ------------------------------------------------------------------
echo ""
echo "[1/5] Training Diffusion Policy for $STEPS steps..."
$PYTHON scripts/01_train_diffusion_pusht.py \
    --output-dir "$RUN_DIR" \
    --run-name "$TAG-$DATE" \
    --steps "$STEPS" \
    --batch-size 64 \
    --num-workers 4 \
    --save-freq "$STEPS" \
    --log-freq 100 \
    --wandb-entity "$ENTITY" \
    --wandb-project "$PROJECT"

CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
echo "[1/5] Done. Checkpoint: $CKPT"

# ------------------------------------------------------------------
# 2. EVAL
# ------------------------------------------------------------------
echo ""
echo "[2/5] Evaluating (20 episodes, Weave multimedia traces)..."
$PYTHON scripts/03_evaluate.py \
    --checkpoint "$CKPT" \
    --output-dir "$RUN_DIR/eval" \
    --tag "$TAG" \
    --num-episodes 20 \
    --record-every 1 \
    --wandb-entity "$ENTITY" \
    --wandb-project "$PROJECT" \
    --wandb-run-name "eval-$TAG-$DATE" \
    --weave-project "$WEAVE_PROJECT"

EVAL_SUMMARY="$RUN_DIR/eval/eval_summary.json"
echo "[2/5] Done. Summary: $EVAL_SUMMARY"
$PYTHON -c "
import json; d=json.load(open('$EVAL_SUMMARY'))
m=d['metrics']
print(f'  success_rate={m[\"success_rate\"]:.2%}  avg_coverage={m[\"avg_max_coverage\"]:.3f}')
"

# ------------------------------------------------------------------
# 3. PROMOTE
# ------------------------------------------------------------------
echo ""
echo "[3/5] Promoting to W&B Models..."
$PYTHON scripts/05_promote_to_registry.py \
    --checkpoint "$CKPT" \
    --eval-summary "$EVAL_SUMMARY" \
    --artifact-name "$ARTIFACT_NAME" \
    --alias "${TAG}-candidate" \
    --wandb-entity "$ENTITY" \
    --wandb-project "$PROJECT" \
    --registry-collection "wandb-registry-model/physical-ai-workshop"

echo "[3/5] Done."

# ------------------------------------------------------------------
# 4. WEAVE EVALUATION
# ------------------------------------------------------------------
echo ""
echo "[4/5] Running Weave Evaluation (4 scorers)..."
DATASET_NAME="pusht-rollouts-${TAG}-${DATE}"
$PYTHON scripts/06_weave_evaluation.py \
    --eval-summary "$EVAL_SUMMARY" \
    --weave-project "$WEAVE_PROJECT" \
    --dataset-name "$DATASET_NAME" \
    --policy-name "$TAG" \
    --artifact-id "wandb-artifact:///$ENTITY/$PROJECT/$ARTIFACT_NAME:latest"

echo "[4/5] Done."

# ------------------------------------------------------------------
# 5. ART CRITIC
# ------------------------------------------------------------------
echo ""
echo "[5/5] Running ART Critic (dry-run)..."
mkdir -p "$RUN_DIR/critic"
$PYTHON scripts/07_art_critic.py \
    --eval-summary "$EVAL_SUMMARY" \
    --output "$RUN_DIR/critic/summary.json" \
    --weave-project "$WEAVE_PROJECT" \
    --project "$PROJECT"

echo "[5/5] Done."

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "  W&B:   https://wandb.ai/$ENTITY/$PROJECT"
echo "  Weave: https://wandb.ai/$WEAVE_PROJECT/weave"
echo "=========================================="
