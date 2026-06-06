# ModernBERT fine-tuning and serving

Canonical guide for BitBot's **multiclass category classifier**: dataset preparation, fine-tuning, evaluation, and BentoML hosting.

**Base model:** `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`

**Recommended flow:**

1. **Phase 1** — build Bitext category splits and run an initial fine-tune.
2. **Phase 2** — generate simulated utterances, convert to the same label space, and **continue** fine-tuning from the Phase 1 `winner/`.
3. **Evaluate** — local Hugging Face checks plus BentoML production-path eval.
4. **Serve** — point Compose at your `winner/` checkpoint and start the stack.

| Topic | Detailed guide |
|-------|----------------|
| Training steps (phase 1 + 2) | [training/README.md](../training/README.md) |
| Evaluation and result interpretation | [testing/README.md](../testing/README.md) |
| Agent routing (category → intent) | [docs/agent.md](agent.md) |

---

## Prerequisites

From the repository root:

```bash
pip install -r training/requirements-train.txt
```

| Requirement | Notes |
|-------------|--------|
| Python 3.11+ | Recommended |
| GPU | Strongly recommended for training |
| Internet | First run downloads Bitext from Hugging Face and caches the base model under `training/models/modernbert-base-zeroshot-v2.0` |
| LLM (phase 2 only) | Ollama or another provider in `.env` for `testing/scripts/generate_category_intent_dataset.py` |

Copy `.env.example` → `.env`. For host-side scripts, also copy `.env.local.example` → `.env.local` (overrides Docker-oriented URLs).

---

## Label space

Production routing expects a closed **13-label** category enum defined in [`training/data/label2id.json`](../training/data/label2id.json):

`ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, `SUBSCRIPTION`, `PRODUCT`, `NO_ISSUE`

- Bitext HF data does not include `PRODUCT` rows; simulator-generated data fills that gap in phase 2.
- Synthetic non-issue samples map to `NO_ISSUE` (label id `12`).
- The LangGraph routes to `no_issue_direct` when category is `no_issue` **or** confidence is below `CATEGORY_CONFIDENCE_THRESHOLD` (default `0.5` in `.env.example`).

---

## Phase 1 — Bitext dataset and initial fine-tune

### Step 1.1 — (Optional) Generate synthetic `no_issue` samples

Script: [`training/scripts/build_is_issue_dataset.py`](../training/scripts/build_is_issue_dataset.py)

Generates JSON with `is_issue: false` rows merged as `NO_ISSUE` during dataset build. Place output under `data/raw/synthetic/no_issue/` (default glob: `no_issue_*.json`).

```bash
python training/scripts/build_is_issue_dataset.py \
  --total-needed 10000 \
  --issue-percent 50 \
  --provider ollama \
  --output data/raw/synthetic/no_issue/generated_is_issue_dataset.json
```

Skip this step with `--bitext-only` in the next command.

### Step 1.2 — Build train / eval / test JSONL

Script: [`training/scripts/create_category_dataset.py`](../training/scripts/create_category_dataset.py)

Loads `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, maps `instruction` → `text`, and (unless `--bitext-only`) merges synthetic `no_issue` rows. Default split: **70% / 15% / 15%** (stratified per label).

```bash
python training/scripts/create_category_dataset.py --mode category \
  --input-dir data/raw/synthetic/no_issue \
  --output-dir training/data/bitext_category
```

**Outputs** in `training/data/bitext_category/`:

| File | Description |
|------|-------------|
| `train.jsonl`, `eval.jsonl`, `test.jsonl` | `{"text": "...", "label": <int>}` |
| `label2id.json` | Copy of canonical mapping |
| `dataset_stats.json` | Counts, class distribution, skip reasons |

**Bitext-only** (no synthetic merge):

```bash
python training/scripts/create_category_dataset.py --mode category --bitext-only \
  --output-dir training/data/bitext_category
```

#### Interpreting `dataset_stats.json`

| Field | What to check |
|-------|----------------|
| `rows_after_dedup` / `duplicates_removed` | Large dedup count may indicate overlapping Bitext and synthetic text |
| `skipped_rows` / `label_missing_in_label2id` | Non-zero → labels outside canonical mapping were dropped |
| `class_distribution_all` | Imbalance is expected; `NO_ISSUE` and `ACCOUNT` are often largest |
| `class_distribution_test` | Holdout support per label — rare labels need simulator data in phase 2 |
| `num_labels` | Should match `training/data/label2id.json` (13) |

**Current local build** (from committed `dataset_stats.json`): 26,872 Bitext + 5,000 synthetic → 29,274 after dedup; split 20,486 / 4,385 / 4,403.

**Modes:** `--mode category` (production default) or `--mode binary` (experimental issue vs `no_issue` only).

### Step 1.3 — Fine-tune on Bitext

Script: [`training/experiments/src/train_multiclass_modernbert.py`](../training/experiments/src/train_multiclass_modernbert.py)

```bash
python training/experiments/src/train_multiclass_modernbert.py \
  --dataset-dir training/data/bitext_category \
  --label2id-file training/data/label2id.json \
  --num-epochs 5 \
  --output-dir training/models/bitext_multiclass_finetuned
```

**Trainer behavior:**

- Caches base model to `training/models/modernbert-base-zeroshot-v2.0` if missing
- Creates a timestamped run directory (e.g. `training/models/bitext_multiclass_finetuned_20260522T092943Z/`)
- Early-stops on eval **macro F1** (`f1_macro`; disable with `--no-early-stopping`)
- Saves best checkpoint to `<run_dir>/winner/` and metrics to `<run_dir>/metrics.json`
- Writes experiment summary under `training/experiments/results/`

**Record the winner path** for phase 2 and serving:

```
training/models/bitext_multiclass_finetuned_<UTC_TIMESTAMP>/winner
```

#### Interpreting training `metrics.json`

| Field | Meaning |
|-------|---------|
| `test_metrics_final.f1_macro` | Primary holdout quality signal |
| `test_metrics_final.accuracy` | Overall correctness; can hide rare-class weakness |
| `confusion_matrix_test` | Row = gold label id, column = predicted id (use `label2id` to decode) |
| `finetuned_winner_dir` | Path to deploy via `MODERNBERT_MODEL_DIR` |

---

## Phase 2 — Simulated data and continue fine-tune

Simulated utterances stress-test persona phrasing and edge cases Bitext does not cover. Reuse the **same** `label2id.json` so label ids stay aligned.

### Step 2.1 — Generate simulated category/intent JSONL

Script: [`testing/scripts/generate_category_intent_dataset.py`](../testing/scripts/generate_category_intent_dataset.py)

```bash
python testing/scripts/generate_category_intent_dataset.py \
  --suite testing/simulator/suites/regression.yaml \
  --max-limit 5000 \
  --output-dir data/raw/simulated
```

Default output: `data/raw/simulated/category_intent_<timestamp>.jsonl`

```json
{"text": "...", "category": "ORDER", "intent": "cancel_order", "seed_id": "...", "persona": "..."}
```

Requires a running LLM (e.g. Ollama at `OLLAMA_BASE_URL`). Use `--save-every` and `--output-file` to resume long runs.

**Alternative:** export user turns from simulator result JSON:

```bash
python testing/scripts/export_simulator_results_jsonl.py \
  --input path/to/simulator_result.json \
  --output data/raw/simulated/from_simulator.jsonl \
  --dedupe
```

### Step 2.2 — Convert simulated JSONL to training splits

```bash
python training/scripts/create_category_dataset.py --mode category \
  --jsonl-input-dir data/raw/simulated \
  --jsonl-glob "category_intent*.jsonl" \
  --label2id-path training/data/label2id.json \
  --output-dir training/data/simulated
```

Rows whose `category` is not in `label2id.json` are dropped (see warnings and `dataset_stats.json`).

### Step 2.3 — Continue fine-tune from the Bitext winner

```bash
python training/experiments/src/train_multiclass_modernbert.py \
  --dataset-dir training/data/simulated \
  --label2id-file training/data/label2id.json \
  --local-base-model-dir training/models/bitext_multiclass_finetuned_<UTC_TIMESTAMP>/winner \
  --output-dir training/models/simulated_multiclass_continue \
  --num-epochs 2 \
  --learning-rate 5e-5
```

Wrapper scripts (edit paths before running):

- [`training/experiments/run_multiclass_continue_finetune.ps1`](../training/experiments/run_multiclass_continue_finetune.ps1)
- [`training/experiments/run_multiclass_continue_finetune.sh`](../training/experiments/run_multiclass_continue_finetune.sh)

After training, set `MODERNBERT_MODEL_DIR` to the new `winner/` if this checkpoint should become production.

**Why phase 2 matters:** a Bitext-only model can score ~99% on Bitext test but ~40% on simulated holdout; continue fine-tuning on simulated splits typically recovers to ~97%+ on the same holdout.

---

## Evaluation

See [testing/README.md](../testing/README.md) for full commands. Two paths:

| Script | Inference | Use when |
|--------|-----------|----------|
| [`testing/scripts/evaluate_modernbert_category.py`](../testing/scripts/evaluate_modernbert_category.py) | Local Hugging Face (`Trainer.predict`) | Fast offline checks |
| [`testing/scripts/evaluate_category_n_intent.py`](../testing/scripts/evaluate_category_n_intent.py) | BentoML `CLASSIFIER_BENTOML_URL` | Matches production serving; optional intent LLM + Postgres |

Each run writes `testing/results/<subdir>/run_YYYYMMDD_HHMMSS.jsonl` with:

1. **`type: "summary"`** — aggregate metrics (start here)
2. **`type: "metadata"`** — input path, model dir, config
3. **`type: "example"`** — per-row gold vs prediction

### Quick eval commands

```bash
# Bitext holdout (local)
python testing/scripts/evaluate_modernbert_category.py \
  --model-dir training/models/<run>/winner \
  -i training/data/bitext_category/test.jsonl \
  -o testing/results/modernbert_category

# Simulated holdout (local)
python testing/scripts/evaluate_modernbert_category.py \
  --model-dir training/models/<run>/winner \
  -i data/raw/simulated/test/category_intent_test.jsonl \
  --label2id-file training/data/label2id.json \
  -o testing/results/modernbert_category

# Bento category-only (production path)
python testing/scripts/evaluate_category_n_intent.py \
  --input-file training/data/bitext_category/test.jsonl \
  --category-only \
  -o testing/results/category_n_intent
```

### Interpreting evaluation results

**Category metrics** (`summary.metrics.category`):

| Field | Meaning |
|-------|---------|
| `accuracy` | Fraction where predicted category equals gold |
| `macro.f1` | Unweighted mean F1 — treats rare classes equally |
| `weighted.f1` | Support-weighted F1 — dominated by frequent classes |
| `per_label[]` | Per-class `support`, `precision`, `recall`, `f1`, `tp`/`fp`/`fn` |

**Decision guide:**

| Signal | Likely cause | Action |
|--------|--------------|--------|
| Low recall, high `fn` | Model under-predicts this label | Add training data or continue fine-tune on simulated rows for that category |
| Low precision, high `fp` | Other utterances misclassified as this label | Inspect confused pairs in `example` rows |
| Bitext strong, simulated weak | Domain gap | Run phase 2 continue fine-tune |
| Bento worse than local HF | Serving checkpoint differs from eval model | Update `MODERNBERT_MODEL_DIR`, rebuild/restart `modernbert` |
| Low `no_issue` recall | Chit-chat routed into issue flows | Add `no_issue` examples or tune `CATEGORY_CONFIDENCE_THRESHOLD` |

**Intent metrics** (Bento eval without `--category-only`): `metrics.intent_on_category_correct` scores intent only when predicted category matches gold — fix category errors first.

**Example observed run** (not a guaranteed benchmark): `testing/results/category_n_intent/run_20260602_091816.jsonl` on 525 simulated test rows — category accuracy ~97.5%, intent-on-correct ~98.0%. Weak spots included `order`↔`delivery` confusion and `get_refund`↔`track_refund` intent mix-ups.

---

## Serve with BentoML

Service: [`services/modernbert_bento/service.py`](../services/modernbert_bento/service.py)

| Piece | Detail |
|-------|--------|
| Endpoints | `POST /classify` → `{category, confidence}`; `POST /health` → `{status: "ok"}` |
| Compose service | `modernbert` on port **3000** |
| Model mount | `${MODERNBERT_MODELS_HOST_DIR:-./training/models}` → `/training/models:ro` |
| Active checkpoint | `MODERNBERT_MODEL_DIR` (container path, e.g. `/training/models/<run>/winner`) |
| Fallback | If unset, newest valid `/training/models/*/winner` by modification time |
| Backend client | `backend/rag/query_classifier.py` → `POST CLASSIFIER_BENTOML_URL` with `{"text":"..."}` |

### Configure and start

1. Set `MODERNBERT_MODEL_DIR` in `.env` to your `winner/` directory (container path):

   ```
   MODERNBERT_MODEL_DIR=/training/models/bitext_multiclass_finetuned_<UTC_TIMESTAMP>/winner
   ```

2. Start the stack:

   ```bash
   docker compose up --build
   ```

3. Verify classifier health:

   ```bash
   curl -s -X POST http://localhost:3000/health
   curl -s -X POST http://localhost:3000/classify -H "Content-Type: application/json" -d "{\"text\":\"Where is my order?\"}"
   ```

4. Smoke eval against Bento:

   ```bash
   python testing/scripts/evaluate_category_n_intent.py --category-only --max-limit 20
   ```

**Troubleshooting:**

| Symptom | Fix |
|---------|-----|
| `modernbert` container fails to start | Ensure `winner/` contains `config.json` + `model.safetensors` (or sharded weights) |
| Wrong categories in production | `MODERNBERT_MODEL_DIR` points at old checkpoint; restart after updating `.env` |
| Eval good locally, bad via Bento | Confirm Bento serves the same `winner/` you evaluated |

---

## Binary fine-tune (experimental)

For issue vs `no_issue` experiments only — **not** the production routing path.

Script: [`training/experiments/src/train_modernbert.py`](../training/experiments/src/train_modernbert.py)

```bash
python training/scripts/create_category_dataset.py --mode binary \
  --input-dir data/raw/synthetic/no_issue \
  --output-dir training/data/bitext_binary

python training/experiments/src/train_modernbert.py \
  --train-file training/data/bitext_binary/train.jsonl \
  --eval-file training/data/bitext_binary/eval.jsonl \
  --output-dir training/models/modernbert_binary_finetuned
```

Labels: `0=no_issue`, `1=issue`.

---

## Runtime routing (LangGraph)

`backend/agent/issue_graph.py` routes as:

- `category == "no_issue"` **or** classifier confidence below `CATEGORY_CONFIDENCE_THRESHOLD` → `no_issue_direct` (chitchat LLM branch)
- otherwise → intent resolution, procedure load, validation, and structured execution

Multiclass training with explicit `NO_ISSUE` (`--mode category`) is the production default.

---

## Related documentation

- [training/README.md](../training/README.md) — phase 1 + 2 training walkthrough
- [testing/README.md](../testing/README.md) — evaluation scripts and metric interpretation
- [docs/agent.md](agent.md) — LangGraph stage contracts
- [README.md](../README.md) — architecture overview and Compose quickstart
