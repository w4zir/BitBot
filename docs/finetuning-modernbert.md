# ModernBERT fine-tuning (current workflow)

This guide reflects the current training and serving flow in this repo:

- build dataset splits with `training/scripts/create_bitext_dataset.py`
- fine-tune with scripts in `training/experiments/src/`
- serve a `winner/` checkpoint through `services/modernbert_bento/service.py`

Base model: `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`.

## Prerequisites

- Python 3.11+ recommended
- GPU strongly recommended for training speed
- Internet access to download Hugging Face Bitext data and (first run) base model

From repo root:

```bash
pip install -r training/requirements-train.txt
```

## 1) Build train/eval/test JSONL from Bitext (+ optional synthetic no_issue)

Script: [`training/scripts/create_bitext_dataset.py`](../training/scripts/create_bitext_dataset.py)

The script loads `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (`instruction` -> `text`) and can merge synthetic `no_issue` records from `data/raw/synthetic/no_issue/no_issue_*.json` by default.

Supported modes:

| Mode | Labeling |
|------|----------|
| `binary` | Bitext -> `issue`, synthetic -> `no_issue`, output labels are `0/1` |
| `category` | Bitext -> HF `category`, synthetic -> `no_issue`, outputs `label2id.json` |
| `intent` | Bitext -> HF `intent`, synthetic -> `no_issue`, outputs `label2id.json` |

Default split is stratified `70/15/15` (`--train-ratio`, `--eval-ratio`, remainder test).

Examples:

```bash
# Multiclass by category (recommended for current backend routing)
python training/scripts/create_bitext_dataset.py --mode category \
  --input-dir data/raw/synthetic/no_issue \
  --output-dir training/data/bitext_category
```

```bash
# Binary issue vs no_issue
python training/scripts/create_bitext_dataset.py --mode binary \
  --input-dir data/raw/synthetic/no_issue \
  --output-dir training/data/bitext_binary_issue
```

```bash
# Bitext-only (skip synthetic no_issue merge)
python training/scripts/create_bitext_dataset.py --mode category --bitext-only \
  --output-dir training/data/bitext_category
```

Primary outputs in `--output-dir`:

- `train.jsonl`, `eval.jsonl`, `test.jsonl`
- `dataset_stats.json`
- `label2id.json` (multiclass modes only)
- optional `dataset_full.json` with `--write-dataset-full`

## 2) Optional: generate synthetic is_issue data with an LLM

Script: [`training/scripts/build_is_issue_dataset.py`](../training/scripts/build_is_issue_dataset.py)

This generates synthetic rows in the format:

```json
{"user_message": "...", "is_issue": true|false, "non_issue_category": "...|null", "notes": "..."}
```

It supports `ollama` and `cerebras`, and can resume/checkpoint during long runs.

Example:

```bash
python training/scripts/build_is_issue_dataset.py \
  --total-needed 10000 \
  --issue-percent 50 \
  --provider ollama \
  --output data/raw/synthetic/no_issue/generated_is_issue_dataset.json
```

## 3) Fine-tune ModernBERT (multiclass, current default path)

Script: [`training/experiments/src/train_multiclass_modernbert.py`](../training/experiments/src/train_multiclass_modernbert.py)

Expected dataset directory:

- `train.jsonl`
- `eval.jsonl`
- `test.jsonl`
- `label2id.json`

Example:

```bash
python training/experiments/src/train_multiclass_modernbert.py \
  --dataset-dir training/data/bitext_category \
  --num-epochs 5 \
  --output-dir training/models/bitext_multiclass_finetuned
```

What this script does:

- uses a local base model cache at `training/models/modernbert-base-zeroshot-v2.0` (downloads automatically if missing)
- creates a timestamped run directory by default (for example `..._20260415T120000Z`)
- trains with early stopping (unless `--no-early-stopping`)
- evaluates on both eval and test
- writes best checkpoint to `<run_dir>/winner/`
- writes metrics to `<run_dir>/metrics.json`
- writes experiment summary JSON to `training/experiments/results/`

## 4) Fine-tune ModernBERT (binary issue/no_issue)

Script: [`training/experiments/src/train_modernbert.py`](../training/experiments/src/train_modernbert.py)

Expected JSONL rows:

```json
{"text": "...", "label": 0}
{"text": "...", "label": 1}
```

Where `0=no_issue`, `1=issue`.

Example:

```bash
python training/experiments/src/train_modernbert.py \
  --train-file training/data/bitext_binary_issue/train.jsonl \
  --eval-file training/data/bitext_binary_issue/eval.jsonl \
  --output-dir training/models/modernbert_finetuned
```

This script writes tokenizer/model artifacts and `metrics.json` to `--output-dir`.

## 5) Serve with BentoML

Service code: [`services/modernbert_bento/service.py`](../services/modernbert_bento/service.py)

The Bento service expects a Hugging Face-compatible exported directory (including `config.json` and model weights) and serves:

- `POST /classify` -> `{ "category": "<label>", "confidence": <float> }`
- `POST /health` -> `{ "status": "ok" }`

Docker Compose mounts the full model directory read-only using `MODERNBERT_MODELS_HOST_DIR` -> `MODERNBERT_MODELS_CONTAINER_DIR` and selects the active `.../winner` checkpoint with `MODERNBERT_MODEL_DIR`.

Run:

```bash
docker compose up --build
```

The backend classifier client calls `CLASSIFIER_BENTOML_URL` (Compose default: `http://modernbert:3000/classify`) with payload `{"text":"..."}`.

## 6) Runtime routing note (LangGraph)

`backend/agent/issue_graph.py` routes as:

- `category == "no_issue"` -> no-issue LLM response branch
- anything else -> validation branch

For that reason, multiclass training with an explicit `no_issue` label (for example `--mode category`) is the current recommended production setup.

---

See [README.md](../README.md) for overall architecture and compose quickstart.
