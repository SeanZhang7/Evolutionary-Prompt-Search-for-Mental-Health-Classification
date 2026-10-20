#!/usr/bin/env bash
# Start the locally validated single-GPU Qwen service in prompt-search.
set -eo pipefail

case "${1:-}" in
  ""|--dry-run) ;;
  *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"
model_dir="$project_dir/LLM/Qwen3.5-9B"

# Non-interactive scripts need Conda's shell initialization before activation.
if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  conda_base_dir="$("$CONDA_EXE" info --base)"
elif command -v conda >/dev/null 2>&1; then
  conda_base_dir="$(conda info --base)"
else
  conda_base_dir="$HOME/miniconda3"
fi
if [[ ! -f "$conda_base_dir/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find conda.sh under $conda_base_dir. Check your Conda installation." >&2
  exit 1
fi
source "$conda_base_dir/etc/profile.d/conda.sh"
conda activate prompt-search

if [[ ! -x "$CONDA_PREFIX/bin/vllm" ]]; then
  echo "vLLM is not installed in the prompt-search environment." >&2
  exit 1
fi
if [[ ! -f "$model_dir/config.json" ]]; then
  echo "Cannot find the downloaded model at $model_dir." >&2
  exit 1
fi

server_command=(
  env CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/vllm" serve "$model_dir"
  --served-model-name qwen35-9b
  --host 127.0.0.1
  --port 8000
  --dtype bfloat16
  --tensor-parallel-size 1
  --max-model-len 2048
  --max-num-seqs 1
  --gpu-memory-utilization 0.90
  --language-model-only
  --enforce-eager
  --generation-config vllm
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking": false}'
)

echo "Conda environment: $CONDA_PREFIX"
echo "Model directory: $model_dir"
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' "${server_command[@]}"
  printf '\n'
  exit 0
fi
echo "Starting http://127.0.0.1:8000 on GPU 0; press Ctrl+C to stop."
exec "${server_command[@]}"
