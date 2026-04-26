#!/usr/bin/env bash
set -euo pipefail

# Continue fine-tuning ModernBERT on simulated_errors.
# Edit the variables below, then run:
#   bash training/experiments/run_multiclass_continue_finetune.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -------- configurable paths --------
TRAIN_SCRIPT="${REPO_ROOT}/training/experiments/src/train_multiclass_modernbert.py"
DATASET_DIR="${REPO_ROOT}/training/data/simulated_errors"
TRAIN_FILE="${DATASET_DIR}/train.jsonl"

# Since this dataset currently has only train.jsonl, we reuse it for eval/test.
# Replace these with dedicated files if/when you add them.
EVAL_FILE="${DATASET_DIR}/train.jsonl"
TEST_FILE="${DATASET_DIR}/train.jsonl"

# Must match the label IDs used in your dataset.
LABEL2ID_FILE="${REPO_ROOT}/training/data/bitext_category/label2id.json"

# Point this to your previously fine-tuned checkpoint directory
# (it should contain config.json + model weights).
LOCAL_BASE_MODEL_DIR="${REPO_ROOT}/training\models\bitext_multiclass_finetuned\winner"

# Output folder base for new run artifacts.
OUTPUT_DIR="${REPO_ROOT}/training/models/simulated_errors_multiclass_continue"

# -------- training knobs --------
NUM_EPOCHS=2
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
LEARNING_RATE=5e-5
EVAL_STEPS=50

python "${TRAIN_SCRIPT}" \
  --dataset-dir "${DATASET_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --test-file "${TEST_FILE}" \
  --label2id-file "${LABEL2ID_FILE}" \
  --local-base-model-dir "${LOCAL_BASE_MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-epochs "${NUM_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --eval-steps "${EVAL_STEPS}"
