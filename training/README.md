# ModernBERT fine-tuning

Step-by-step guide to fine-tune **ModernBERT** (`MoritzLaurer/ModernBERT-base-zeroshot-v2.0`) for BitBot category routing. The recommended flow is:

1. **Bitext** — build a multiclass dataset from Hugging Face Bitext (+ optional synthetic `no_issue` rows) and run an initial fine-tune.
2. **Simulated data** — generate persona-driven utterances from the conversation simulator, convert them to the same label space, then **continue** fine-tuning from the Bitext checkpoint.

Production routing expects a **`no_issue`** label (see `backend/agent/issue_graph.py`). Use `--mode category` unless you are experimenting with binary issue detection.

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
| LLM (phase 2 only) | Ollama or another provider configured in `.env` for `testing/scripts/generate_category_intent_dataset.py` |

---

## Phase 1 — Bitext dataset and initial fine-tune

### Step 1.1 — (Optional) Generate synthetic `no_issue` samples

If you want extra non-issue examples beyond Bitext, generate JSON files with `training/scripts/build_is_issue_dataset.py` (rows with `is_issue: false` are merged; issue rows are skipped).

```bash
python training/scripts/build_is_issue_dataset.py \
  --total-needed 10000 \
  --issue-percent 50 \
  --provider ollama \
  --output data/raw/synthetic/no_issue/generated_is_issue_dataset.json
```

Place output under `data/raw/synthetic/no_issue/` (default glob: `no_issue_*.json`). Skip this step with `--bitext-only` in the next command.

### Step 1.2 — Build train / eval / test JSONL

Script: [`scripts/create_category_dataset.py`](scripts/create_category_dataset.py)

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
| `label2id.json` | String label → integer id (keep this for phase 2) |
| `dataset_stats.json` | Counts, class distribution, skip reasons |

Review `dataset_stats.json` for skipped rows and class balance before training.

**Bitext-only** (no synthetic merge):

```bash
python training/scripts/create_category_dataset.py --mode category --bitext-only \
  --output-dir training/data/bitext_category
```

Other modes: `--mode binary` (issue vs `no_issue`), `--mode intent` (HF intent labels). Category mode is the production default.

### Step 1.3 — Fine-tune on Bitext

Script: [`experiments/src/train_multiclass_modernbert.py`](experiments/src/train_multiclass_modernbert.py)

```bash
python training/experiments/src/train_multiclass_modernbert.py \
  --dataset-dir training/data/bitext_category \
  --num-epochs 5 \
  --output-dir training/models/bitext_multiclass_finetuned
```

**What the trainer does:**

- Downloads/caches the base model to `training/models/modernbert-base-zeroshot-v2.0` if missing
- Creates a timestamped run directory (e.g. `training/models/bitext_multiclass_finetuned_20260521T151838Z/`)
- Early-stops on eval **macro F1** (disable with `--no-early-stopping`)
- Saves the best checkpoint to `<run_dir>/winner/` and metrics to `<run_dir>/metrics.json`
- Writes an experiment summary under `training/experiments/results/`

**Record the winner path** — you need it for phase 2 and for serving:

```
training/models/bitext_multiclass_finetuned_<UTC_TIMESTAMP>/winner
```

Set `MODERNBERT_MODEL_DIR` in `.env` to that path when deploying via BentoML.

---

## Phase 2 — Simulated data and continue fine-tune

Simulated utterances stress-test routing with persona + seed diversity. They must use the **same** `label2id.json` as phase 1 so label ids stay aligned.

### Step 2.1 — Generate simulated category/intent JSONL

Script: [`../testing/scripts/generate_category_intent_dataset.py`](../testing/scripts/generate_category_intent_dataset.py)

Uses the conversation simulator (personas, seeds, suite YAML) to produce user messages with gold labels:

```bash
python testing/scripts/generate_category_intent_dataset.py \
  --suite testing/simulator/suites/regression.yaml \
  --max-limit 5000 \
  --output-dir data/raw/simulated
```

Default output: `data/raw/simulated/category_intent_<timestamp>.jsonl` with rows:

```json
{"text": "...", "category": "ORDER", "intent": "cancel_order"}
```

Requires a running LLM (e.g. Ollama at `OLLAMA_BASE_URL`). Tune `--persona`, `--seed`, `--save-every` for checkpoints on long runs.

**Alternative:** export user turns from full simulator result JSON:

```bash
python testing/scripts/export_simulator_results_jsonl.py \
  --input path/to/simulator_result.json \
  --output data/raw/simulated/from_simulator.jsonl \
  --dedupe
```

### Step 2.2 — Convert simulated JSONL to training splits

Reuse the **phase 1** label mapping so ids match the Bitext model:

```bash
python training/scripts/create_category_dataset.py --mode category \
  --jsonl-input-dir data/raw/simulated \
  --jsonl-glob "category_intent*.jsonl" \
  --label2id-path training/data/bitext_category/label2id.json \
  --output-dir training/data/simulated
```

Rows whose `category` is not in `label2id.json` are dropped (see warnings in stderr and `dataset_stats.json`).

For error-focused continue training, some teams use a dedicated folder such as `training/data/simulated_errors` (same command, different `--output-dir`).

### Step 2.3 — Continue fine-tune from the Bitext winner

Point `--local-base-model-dir` at your phase 1 `winner/` directory:

```bash
python training/experiments/src/train_multiclass_modernbert.py \
  --dataset-dir training/data/simulated \
  --label2id-file training/data/bitext_category/label2id.json \
  --local-base-model-dir training/models/bitext_multiclass_finetuned_<UTC_TIMESTAMP>/winner \
  --output-dir training/models/simulated_multiclass_continue \
  --num-epochs 2 \
  --learning-rate 5e-5
```

If you only have `train.jsonl` (no separate eval/test yet), you can reuse `train.jsonl` for eval/test temporarily — see the commented variables in:

- [`experiments/run_multiclass_continue_finetune.sh`](experiments/run_multiclass_continue_finetune.sh)
- [`experiments/run_multiclass_continue_finetune.ps1`](experiments/run_multiclass_continue_finetune.ps1)

Edit paths (especially `LOCAL_BASE_MODEL_DIR` / `$LocalBaseModelDir`) and run the script from the repo root.

After training, update `MODERNBERT_MODEL_DIR` to the new `winner/` if this checkpoint should become production.

---

## Serve the checkpoint (optional)

BentoML service: [`services/modernbert_bento/service.py`](../services/modernbert_bento/service.py)

Docker Compose mounts `training/models` and reads `MODERNBERT_MODEL_DIR` (see `.env.example`). The backend calls `CLASSIFIER_BENTOML_URL` for `POST /classify`.

```bash
docker compose up --build
```

---

## Directory layout

| Path | Purpose |
|------|---------|
| `scripts/` | Dataset builders (`create_category_dataset.py`, `build_is_issue_dataset.py`) |
| `experiments/src/` | Training entrypoints (`train_multiclass_modernbert.py`, `train_modernbert.py`) |
| `experiments/results/` | Experiment summaries from training runs |
| `data/` | Generated datasets (not committed; create locally) |
| `models/` | Base model cache + fine-tuned run directories (`.../winner/`) |
| `requirements-train.txt` | Python dependencies for training |

---

## Related documentation

- [docs/finetuning-modernbert.md](../docs/finetuning-modernbert.md) — extended notes (binary mode, Bento serving)
- [testing/README.md](../testing/README.md) — evaluate checkpoints on Bitext and simulated holdouts
- [scripts/README.md](scripts/README.md) — script index
