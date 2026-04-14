# ModernBERT issue / no_issue fine-tuning (BitBot)

This document describes how to build the Bitext-derived dataset, map it to binary **issue** vs **no_issue**, fine-tune `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`, and serve the checkpoint with the BentoML container.

## Prerequisites

- Python 3.11+ recommended
- GPU optional but strongly recommended for training
- Hugging Face account not required for public datasets

Install training dependencies from the repo root:

```bash
pip install -r training/requirements-train.txt
```

## 1. Multiclass Bitext dataset (`dataset_full.json`)

**Script:** [`training/scripts/create_bitext_dataset.py`](../training/scripts/create_bitext_dataset.py)

Loads **`bitext/Bitext-customer-support-llm-chatbot-training-dataset`** from Hugging Face (`instruction` → `text`, `intent` → `label`) and optionally merges synthetic **no_issue** rows from JSON files matching `no_issue_*.json` under `--synthetic-dir`.

### Option A — Bitext + synthetic no_issue (default)

Uses committed samples under `training/data/samples/synthetic_no_issue/`:

```bash
python training/scripts/create_bitext_dataset.py --output-dir training/data/bitext
```

### Option B — Bitext only (no synthetic files)

```bash
python training/scripts/create_bitext_dataset.py --bitext-only --output-dir training/data/bitext
```

**Outputs** (under `--output-dir`):

| File | Purpose |
|------|---------|
| `dataset_full.json` | `[{ "text", "label" }, ...]` string labels |
| `label2id.json` | Multiclass label → integer id |
| `dataset_stats.json` | Build statistics |

## 2. Multiclass train/eval/test JSONL (optional)

**Script:** [`training/scripts/build_bitext_training_dataset.py`](../training/scripts/build_bitext_training_dataset.py)

Stratified split of `dataset_full.json` into `train.jsonl` / `eval.jsonl` / `test.jsonl` with **integer** labels from `label2id.json`.

```bash
python training/scripts/build_bitext_training_dataset.py \
  --dataset-full training/data/bitext/dataset_full.json \
  --label2id training/data/bitext/label2id.json \
  --output-dir training/data/bitext
```

## 3. Binary issue / no_issue JSONL

**Script:** [`training/scripts/build_binary_issue_training_dataset.py`](../training/scripts/build_binary_issue_training_dataset.py)

Maps multiclass string labels to binary:

- `no_issue` → **0**
- any other intent → **1** (treated as issue-like)

```bash
python training/scripts/build_binary_issue_training_dataset.py \
  --dataset-full training/data/bitext/dataset_full.json \
  --output-dir training/data/bitext_binary_issue
```

Outputs: `train.jsonl`, `eval.jsonl`, `test.jsonl`, `dataset_split_stats.json`.

## 4. Fine-tune ModernBERT (binary)

**Script:** [`training/scripts/train_modernbert.py`](../training/scripts/train_modernbert.py)

Expected JSONL lines:

```json
{"text": "...", "label": 0}
{"text": "...", "label": 1}
```

- `0` = **no_issue**
- `1` = **issue**

Base model (default): **`MoritzLaurer/ModernBERT-base-zeroshot-v2.0`** with `ignore_mismatched_sizes=True` for the classification head.

Example:

```bash
python training/scripts/train_modernbert.py \
  --train-file training/data/bitext_binary_issue/train.jsonl \
  --eval-file training/data/bitext_binary_issue/eval.jsonl \
  --output-dir training/models/modernbert_finetuned
```

Artifacts are written to `--output-dir` (tokenizer + model + `metrics.json`). **Do not commit large weights**; `.gitignore` excludes them.

### Tiny sample data (smoke test)

The repo includes [`training/data/samples/bitext_binary_mini/`](../training/data/samples/bitext_binary_mini/) for a minimal local run (not for production quality).

## 5. Evaluate a checkpoint

**Script:** [`training/scripts/eval_modernbert.py`](../training/scripts/eval_modernbert.py)

```bash
python training/scripts/eval_modernbert.py \
  --checkpoint training/models/modernbert_finetuned \
  --data-file training/data/bitext_binary_issue/test.jsonl
```

Optional: `--compare-with-base` to compare against the zeroshot base checkpoint.

## 6. Serve with BentoML (Docker)

The **`modernbert`** service loads `MODERNBERT_MODEL_DIR` (compose: `/models/modernbert_finetuned` mounted from `./training/models/modernbert_finetuned`).

1. Ensure the fine-tuned Hugging Face export exists in `training/models/modernbert_finetuned/`.
2. Start the stack: `docker compose up --build`.

The backend calls **`CLASSIFIER_BENTOML_URL`** (e.g. `http://modernbert:3000/classify`) with JSON `{"text": "..."}`.

Thresholding is applied inside the Bento service via **`CLASSIFIER_THRESHOLD`** (default `0.7`).

## 7. LangGraph runtime

The FastAPI app runs a minimal LangGraph (`backend/agent/issue_graph.py`): single **classify** node → **END**, invoking the same Bento endpoint via `backend/rag/query_classifier.py`. Extend the graph when you add tools or RAG.

---

**See also:** [README.md](../README.md) for architecture overview and `docker compose` quickstart.
