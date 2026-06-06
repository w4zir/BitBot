# ModernBERT evaluation

Step-by-step guide to evaluate a fine-tuned **ModernBERT** category classifier on **Bitext holdout** data, then on **simulated** utterances, and interpret the metrics written under `testing/results/`.

**Canonical reference:** [docs/finetuning-modernbert.md](../docs/finetuning-modernbert.md) (full pipeline, Bento serving, dataset prep).

Two evaluation paths:

| Script | Inference | Use when |
|--------|-----------|----------|
| [`scripts/evaluate_modernbert_category.py`](scripts/evaluate_modernbert_category.py) | Local Hugging Face (`Trainer.predict`) | Fast offline checks; no Bento/Compose required |
| [`scripts/evaluate_category_n_intent.py`](scripts/evaluate_category_n_intent.py) | BentoML `CLASSIFIER_BENTOML_URL` | Matches production serving; optional intent LLM + Postgres |

---

## Prerequisites

From the repository root:

```bash
pip install -r training/requirements-train.txt
```

Copy `.env.example` → `.env` (and `.env.local` for host-side overrides). Important variables:

| Variable | Used by |
|----------|---------|
| `MODERNBERT_MODEL_DIR` | Local eval (`evaluate_modernbert_category.py`) — path to `.../winner` |
| `CLASSIFIER_BENTOML_URL` | Bento eval (`evaluate_category_n_intent.py`) — e.g. `http://localhost:3000/classify` when port-forwarded |
| `CLASSIFIER_BENTOML_TIMEOUT_SECONDS` | HTTP timeout (default `5`) |
| `INTENT_MODEL_PROVIDER`, `INTENT_MODEL`, `OLLAMA_BASE_URL` | Full pipeline eval (intent step; use `VLLM_*` when provider is `vllm`) |
| `POSTGRES_HOST`, … | Intent allowlist lookup via DB |
| `CATEGORY_CONFIDENCE_THRESHOLD` | Agent routes low-confidence predictions to `no_issue_direct` (default `0.5`) |

For Bento eval, start the stack (or at least the `modernbert` service) and point `CLASSIFIER_BENTOML_URL` at a reachable classify endpoint. Confirm `MODERNBERT_MODEL_DIR` in `.env` matches the checkpoint you intend to test.

---

## Supported dataset formats

Both eval scripts accept:

**Training format** (Bitext / simulated splits from `create_category_dataset.py`):

```json
{"text": "where is my order?", "label": 6}
```

Requires sibling or explicit `label2id.json` (`{"ORDER": 6, ...}`).

**Category/intent format** (simulator output):

```json
{"text": "where is my order?", "category": "ORDER", "intent": "track_order"}
```

Category labels are normalized to lowercase for comparison (`ORDER` → `order`).

---

## Phase 1 — Evaluate on Bitext (holdout test set)

Use the `test.jsonl` produced during training (default: `training/data/bitext_category/test.jsonl`).

### Step 1.1 — Local Hugging Face evaluation

Set `MODERNBERT_MODEL_DIR` in `.env` to your fine-tuned `winner/` directory, or pass `--model-dir`:

```bash
python testing/scripts/evaluate_modernbert_category.py \
  --model-dir training/models/bitext_multiclass_finetuned_<UTC_TIMESTAMP>/winner \
  -i training/data/bitext_category/test.jsonl \
  -o testing/results/modernbert_category
```

Default output directory (if `-o` is omitted): `testing/results/modernbert_category/`.

Omit `-i` to default to `<dataset-dir>/test.jsonl` (`--dataset-dir` defaults to `training/data/bitext_category`).

**Useful flags:**

| Flag | Effect |
|------|--------|
| `--max-limit 200` | Smoke test on first N rows |
| `--verbose-debug` | Print per-sample gold vs prediction |
| `--skip-invalid` | Drop bad lines instead of failing |
| `--batch-size 8` | Eval batch size |

### Step 1.2 — BentoML evaluation (production path)

Requires a running classifier at `CLASSIFIER_BENTOML_URL`:

```bash
python testing/scripts/evaluate_category_n_intent.py \
  --input-file training/data/bitext_category/test.jsonl \
  --category-only \
  -o testing/results/category_n_intent
```

`--category-only` skips intent LLM and Postgres — category metrics only.

### Step 1.3 — Read Bitext results

Each run writes `testing/results/<output-dir>/run_YYYYMMDD_HHMMSS.jsonl` (three record types):

1. **`type: "summary"`** — aggregate metrics (start here)
2. **`type: "metadata"`** — input path, model dir, batch size, row counts
3. **`type: "example"`** — one row per sample with gold and predictions

**Category metrics** (in `summary.metrics.category`):

| Field | Meaning |
|-------|---------|
| `accuracy` | Fraction of rows where predicted category equals gold |
| `macro.f1` | Unweighted mean F1 across labels — treats rare classes equally |
| `weighted.f1` | Support-weighted F1 — dominated by frequent classes |
| `per_label[]` | Per-class `support`, `precision`, `recall`, `f1`, `tp`/`fp`/`fn` |

**Example rows** include:

```json
"modernbert": {
  "category": "order",
  "confidence": 0.9978,
  "is_category_correct": true,
  "error": ""
}
```

Use `confidence` to tune `CATEGORY_CONFIDENCE_THRESHOLD` in `.env` (low confidence may route to `no_issue_direct` in the agent graph).

**Sanity checks on Bitext:**

- Macro F1 near weighted F1 → balanced performance across classes
- High `no_issue` recall → fewer false escalations to issue handling
- Large `fn` on a label → model misses that category (needs more data or continue fine-tune)
- Large `fp` on a label → confusion with similar categories (e.g. `feedback` vs `subscription`)

---

## Phase 2 — Evaluate on simulated data

### Step 2.1 — Obtain simulated labeled JSONL

**Generate** new samples (see [training/README.md](../training/README.md#step-21--generate-simulated-categoryintent-jsonl)):

```bash
python testing/scripts/generate_category_intent_dataset.py \
  --suite testing/simulator/suites/regression.yaml \
  --max-limit 1000
```

**Or** build a holdout split from existing simulator JSONL:

```bash
python training/scripts/create_category_dataset.py --mode category \
  --jsonl-input-dir data/raw/simulated \
  --jsonl-glob "category_intent*.jsonl" \
  --label2id-path training/data/bitext_category/label2id.json \
  --output-dir training/data/simulated
```

Then evaluate `training/data/simulated/test.jsonl`.

**Or** evaluate the raw JSONL directly (category/intent format):

```bash
python testing/scripts/evaluate_modernbert_category.py \
  -i data/raw/simulated/category_intent_<timestamp>.jsonl \
  --label2id-file training/data/bitext_category/label2id.json \
  -o testing/results/modernbert_category
```

### Step 2.2 — Local eval on simulated holdout

```bash
python testing/scripts/evaluate_modernbert_category.py \
  --model-dir training/models/simulated_multiclass_continue_<UTC_TIMESTAMP>/winner \
  -i training/data/simulated/test.jsonl \
  --label2id-file training/data/bitext_category/label2id.json \
  -o testing/results/modernbert_category
```

Compare `run_*.jsonl` from phase 1 vs phase 2 on the same metric block — simulated eval usually exposes persona phrasing and edge cases Bitext does not cover. A Bitext-only checkpoint may score ~99% on Bitext test but ~40% on simulated holdout; a phase-2 continue-finetuned checkpoint typically recovers to ~97%+.

### Step 2.3 — Full stack eval (category + intent)

After Bento is serving the checkpoint you want to test:

```bash
python testing/scripts/evaluate_category_n_intent.py \
  --input-file data/raw/simulated/category_intent_<timestamp>.jsonl \
  --max-limit 100 \
  --verbose-debug \
  -o testing/results/category_n_intent
```

Without `--category-only`, intent is evaluated **only when the predicted category matches gold** (`metrics.intent_on_category_correct`). That isolates intent quality from category mistakes.

Requires Postgres seeded with intents per category and a reachable intent LLM.

**Host overrides** when DB/Ollama run on the machine, not inside Compose:

```bash
python testing/scripts/evaluate_category_n_intent.py \
  --input-file data/raw/simulated/category_intent.jsonl \
  --category-only \
  --bentoml-url http://localhost:3000/classify \
  --postgres-host localhost
```

---

## Interpreting results

### Terminal summary

Both scripts print a short summary when the run finishes:

```
Counts: total_examples=4403, category_correct_examples=4086, skipped_invalid_rows=0
Category metrics: accuracy=0.9280, macro_f1=0.9308, weighted_f1=0.9318
Output file: testing/results/modernbert_category/run_20260521_161911.jsonl
```

### Per-label table (`per_label`)

| Signal | Likely cause |
|--------|----------------|
| Low **recall**, high **fn** | Model under-predicts this label; add training data or continue fine-tune on simulated rows for that category |
| Low **precision**, high **fp** | Other utterances misclassified as this label; check confused pairs in `example` rows |
| High **support**, low **f1** | Priority fix — affects most traffic |
| `no_issue` recall drop | Chit-chat routed into issue flows — adjust threshold or add `no_issue` examples |

### Comparing Bitext vs simulated runs

| Observation | Action |
|-------------|--------|
| Bitext test strong, simulated weak | Run phase 2 continue fine-tune on simulated splits; expand simulator suite coverage |
| Both weak on same label | Label definition or DB intent list mismatch; inspect gold labels in JSONL |
| Bento eval worse than local HF | Serving checkpoint differs from `MODERNBERT_MODEL_DIR`; rebuild/restart `modernbert` service |
| Intent metrics only in full eval | Category errors mask intent; fix category first, then re-run without `--category-only` |

### Example observed run (June 2026)

`testing/results/category_n_intent/run_20260602_091816.jsonl` on 525 simulated test rows via Bento (`http://localhost:3000/classify`):

| Metric | Value |
|--------|-------|
| Category accuracy | ~97.5% |
| Category macro F1 | ~0.88 |
| Intent-on-category-correct accuracy | ~98.0% |
| Intent-on-category-correct macro F1 | ~0.87 |

Notable weak spots in that run: `order`↔`delivery` category confusion; `get_refund`↔`track_refund` intent mix-ups. Treat as an example run, not a guaranteed benchmark.

### Debugging individual failures

```bash
python testing/scripts/evaluate_modernbert_category.py \
  -i training/data/simulated/test.jsonl \
  --max-limit 20 \
  --verbose-debug
```

Or inspect `type: "example"` lines in the JSONL where `is_category_correct` is `false`.

### Results directory layout

| Directory | Script |
|-----------|--------|
| `testing/results/modernbert_category/` | `evaluate_modernbert_category.py` (default `-o`) |
| `testing/results/category_n_intent/` | `evaluate_category_n_intent.py` (default `-o`) |

Runs are append-only JSONL files; keep the `metadata` record to reproduce the exact config later.

---

## Quick smoke checklist

```bash
# 1) Local category eval — 50 Bitext test rows
python testing/scripts/evaluate_modernbert_category.py \
  -i training/data/bitext_category/test.jsonl --max-limit 50

# 2) Local category eval — 50 simulated rows
python testing/scripts/evaluate_modernbert_category.py \
  -i training/data/simulated/test.jsonl \
  --label2id-file training/data/bitext_category/label2id.json \
  --max-limit 50

# 3) Bento category-only (stack up)
python testing/scripts/evaluate_category_n_intent.py --category-only --max-limit 20
```

---

## Related documentation

- [training/README.md](../training/README.md) — dataset build and fine-tune workflow
- [docs/finetuning-modernbert.md](../docs/finetuning-modernbert.md) — canonical pipeline (dataset, training, serving, routing)
- [specs/simulator-spec.md](../specs/simulator-spec.md) — simulator architecture
