#!/usr/bin/env bash
# Resume pipeline from eval (step 2–5). Use when training already completed.
# Usage: bash run_pipeline_eval_only.sh [steps]
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

CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
EVAL_SUMMARY="$RUN_DIR/eval/eval_summary.json"

if [ ! -d "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

echo "=========================================="
echo "  Pipeline (eval → promote → weave → art)"
echo "  steps=$STEPS  tag=$TAG"
echo "  $(date)"
echo "=========================================="

# ------------------------------------------------------------------
# 2. EVAL (CPU — avoids OOM after GPU training)
# ------------------------------------------------------------------
echo ""
echo "[2/5] Evaluating (20 episodes, Weave multimedia traces)..."
$PYTHON scripts/03_evaluate.py \
    --checkpoint "$CKPT" \
    --output-dir "$RUN_DIR/eval" \
    --tag "$TAG" \
    --num-episodes 20 \
    --record-every 1 \
    --device cpu \
    --wandb-entity "$ENTITY" \
    --wandb-project "$PROJECT" \
    --wandb-run-name "eval-$TAG-$DATE" \
    --weave-project "$WEAVE_PROJECT"

echo "[2/5] Done."
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
