# Training scripts

Scripts for Bitext dataset creation, binary **issue / no_issue** splits, and ModernBERT fine-tuning.

**Detailed steps:** [../../docs/finetuning-modernbert.md](../../docs/finetuning-modernbert.md)

| Script | Purpose |
|--------|---------|
| `create_bitext_dataset.py` | HF Bitext + optional synthetic `no_issue` JSON → stratified `train`/`eval`/`test` JSONL (`--mode binary`, `category`, or `intent`). Optional `--write-dataset-full` for `dataset_full.json`. |
| `build_bitext_training_dataset.py` | Multiclass stratified train/eval/test JSONL from existing `dataset_full.json` + `label2id.json` |
| `build_binary_issue_training_dataset.py` | Multiclass `dataset_full.json` → binary JSONL (legacy two-step flow) |
| `train_modernbert.py` | Fine-tune `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` |
| `eval_modernbert.py` | Evaluate checkpoint vs JSONL |

Install deps: `pip install -r training/requirements-train.txt` (from repo root).
