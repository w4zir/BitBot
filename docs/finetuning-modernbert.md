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

## 1. Bitext + synthetic no_issue → train/eval/test (single script)

**Script:** [`training/scripts/create_bitext_dataset.py`](../training/scripts/create_bitext_dataset.py)

Loads **`bitext/Bitext-customer-support-llm-chatbot-training-dataset`** from Hugging Face (`instruction` → `text`; `category` and `intent` are used depending on mode) and optionally merges synthetic **no_issue** rows from JSON files matching `no_issue_*.json` under `--input-dir` / `--synthetic-dir` (default: `data/raw/synthetic/no_issue`).

Choose a **`--mode`**:

| Mode | Labels |
|------|--------|
| `binary` | Bitext → `issue`, synthetic → `no_issue`; JSONL uses `0` / `1` |
| `category` | Bitext → HF `category`, synthetic → `no_issue`; JSONL uses `label2id.json` |
| `intent` | Bitext → HF `intent`, synthetic → `no_issue`; JSONL uses `label2id.json` |

Default split: **70% / 15% / 15%** (`--train-ratio`, `--eval-ratio`).

### Examples

**Intent multiclass + synthetic no_issue** (writes `label2id.json`):

```bash
python training/scripts/create_bitext_dataset.py --mode intent \
  --input-dir data/raw/synthetic/no_issue \
  --output-dir training/data/bitext
```

**Binary issue / no_issue**:

```bash
python training/scripts/create_bitext_dataset.py --mode binary \
  --input-dir data/raw/synthetic/no_issue \
  --output-dir training/data/bitext_binary_issue
```

**Bitext only** (no synthetic files):

```bash
python training/scripts/create_bitext_dataset.py --mode intent --bitext-only --output-dir training/data/bitext
```

**Optional** `--write-dataset-full` also writes `dataset_full.json` (deduped string labels for the chosen mode).

**Outputs** (under `--output-dir`):

| File | Purpose |
|------|---------|
| `train.jsonl`, `eval.jsonl`, `test.jsonl` | Ready for training (`label` int; binary uses 0/1) |
| `label2id.json` | Present for `category` and `intent` modes |
| `dataset_stats.json` | Build + split statistics |

## 2. Multiclass train/eval/test JSONL from `dataset_full.json` (optional two-step)

**Script:** [`training/scripts/build_bitext_training_dataset.py`](../training/scripts/build_bitext_training_dataset.py)

Use this if you already have `dataset_full.json` + `label2id.json` from another source.

```bash
python training/scripts/build_bitext_training_dataset.py \
  --dataset-full training/data/bitext/dataset_full.json \
  --label2id training/data/bitext/label2id.json \
  --output-dir training/data/bitext
```

## 3. Binary issue / no_issue from `dataset_full.json` (optional two-step)

**Script:** [`training/scripts/build_binary_issue_training_dataset.py`](../training/scripts/build_binary_issue_training_dataset.py)

Maps string labels to binary (`no_issue` → **0**, anything else → **1**). Prefer **`create_bitext_dataset.py --mode binary`** for a single step when building from Bitext + synthetic.

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
