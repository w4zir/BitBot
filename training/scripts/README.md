# Training scripts

Scripts for Bitext/simulated dataset creation and synthetic `no_issue` generation.

**Canonical pipeline:** [docs/finetuning-modernbert.md](../../docs/finetuning-modernbert.md)

| Script | Purpose |
|--------|---------|
| [`create_category_dataset.py`](create_category_dataset.py) | HF Bitext and/or simulated JSONL + optional synthetic `no_issue` JSON → stratified `train`/`eval`/`test` JSONL (`--mode category` or `binary`). Writes `dataset_stats.json` and `label2id.json`. |
| [`build_is_issue_dataset.py`](build_is_issue_dataset.py) | LLM-generated synthetic `no_issue` samples (`ollama`, `cerebras`, or `vllm`) |

**Training entrypoints** (not in this directory):

| Script | Purpose |
|--------|---------|
| [`../experiments/src/train_multiclass_modernbert.py`](../experiments/src/train_multiclass_modernbert.py) | Production multiclass fine-tune (phase 1 + continue) |
| [`../experiments/src/train_modernbert.py`](../experiments/src/train_modernbert.py) | Experimental binary issue vs `no_issue` fine-tune |

**Evaluation scripts** (not in this directory):

| Script | Purpose |
|--------|---------|
| [`../../testing/scripts/evaluate_modernbert_category.py`](../../testing/scripts/evaluate_modernbert_category.py) | Local Hugging Face category eval |
| [`../../testing/scripts/evaluate_category_n_intent.py`](../../testing/scripts/evaluate_category_n_intent.py) | BentoML category (+ optional intent) eval |

Install deps: `pip install -r training/requirements-train.txt` (from repo root).
