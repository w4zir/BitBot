# ModernBERT fine-tuned artifacts (local)

Place your trained model directory here after running `training/scripts/train_modernbert.py`.

Expected layout (Hugging Face format):

- `config.json`
- `tokenizer*` files
- `model.safetensors` or `pytorch_model.bin`

Docker Compose mounts this directory read-only into the `modernbert` Bento service at `/models/modernbert_finetuned`.

This directory is not committed (large files). Generate artifacts locally; see [docs/finetuning-modernbert.md](../../docs/finetuning-modernbert.md).
