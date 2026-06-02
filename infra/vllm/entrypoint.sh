#!/bin/sh
set -eu

model="${BITBOT_VLLM_MODEL:-${VLLM_MODEL:?Set VLLM_MODEL in .env (HuggingFace model id)}}"
served_name="${BITBOT_VLLM_SERVED_NAME:-${VLLM_SERVED_NAME:-$model}}"
max_model_len="${BITBOT_VLLM_MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-512}}"
gpu_memory_utilization="${BITBOT_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.85}}"
max_num_seqs="${BITBOT_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-1}}"
reasoning_parser="${BITBOT_VLLM_REASONING_PARSER:-${VLLM_REASONING_PARSER:-}}"
cache_dir="${HF_HOME:-/root/.cache/huggingface}"
hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

echo "============================================================"
echo "vLLM model preparation"
echo "Model:        $model"
echo "Served name:  $served_name"
echo "HF cache:     $cache_dir"
echo "============================================================"
echo "[vllm] Downloading model files. Existing cached files are reused."

export BITBOT_VLLM_MODEL="$model"
export HF_TOKEN="$hf_token"
export HUGGING_FACE_HUB_TOKEN="$hf_token"

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ["BITBOT_VLLM_MODEL"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

print(f"[vllm] Hugging Face snapshot download started: {model}", flush=True)
path = snapshot_download(repo_id=model, token=token)
print(f"[vllm] Hugging Face snapshot ready: {path}", flush=True)
PY

echo "[vllm] Model download complete: $model"
echo "[vllm] Starting OpenAI-compatible vLLM server..."

set -- "$model" \
  --served-model-name "$served_name" \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --max-num-seqs "$max_num_seqs" \
  --enforce-eager

if [ -n "$reasoning_parser" ]; then
  set -- "$@" --reasoning-parser "$reasoning_parser"
fi

unset VLLM_API_BASE VLLM_MODEL VLLM_SERVED_NAME VLLM_REASONING_PARSER \
  VLLM_TEMPERATURE VLLM_HOST_PORT VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS \
  VLLM_GPU_MEMORY_UTILIZATION BITBOT_VLLM_MODEL BITBOT_VLLM_SERVED_NAME \
  BITBOT_VLLM_MAX_MODEL_LEN BITBOT_VLLM_GPU_MEMORY_UTILIZATION \
  BITBOT_VLLM_MAX_NUM_SEQS BITBOT_VLLM_REASONING_PARSER

exec vllm serve "$@"
