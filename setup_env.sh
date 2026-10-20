#!/usr/bin/env bash
# Create the project's Linux/NVIDIA environment, then check dependencies and CUDA.
set -eo pipefail

dry_run=false
install_existing=false
skip_cuda=false
for option in "$@"; do
  case "$option" in
    --dry-run) dry_run=true ;;
    --install) install_existing=true ;;
    --skip-cuda-check) skip_cuda=true ;;
    -h|--help)
      cat <<'HELP'
Usage: ./setup_env.sh [--dry-run] [--install] [--skip-cuda-check]

Create prompt-search with Python 3.12.15 and install the pinned dependencies.
If the environment exists, check it without installing packages.
  --dry-run          Print the commands without creating, installing, or checking.
  --install          Also install pinned packages into an existing environment.
  --skip-cuda-check  Check packages only (e.g. on a login node without GPU access).
HELP
      exit 0 ;;
    *) echo "Unknown option: $option. Use --help for usage." >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != Linux ]]; then
  echo "This setup targets Linux with an NVIDIA GPU." >&2
  exit 1
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  conda_base_dir="$("$CONDA_EXE" info --base)"
elif command -v conda >/dev/null 2>&1; then
  conda_base_dir="$(conda info --base)"
else
  conda_base_dir="$HOME/miniconda3"
fi
if [[ ! -f "$conda_base_dir/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find conda.sh. Install Conda or set CONDA_EXE to its executable." >&2
  exit 1
fi
source "$conda_base_dir/etc/profile.d/conda.sh"

run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" == false ]]; then
    "$@"
  fi
}

env_exists=false
if conda activate prompt-search >/dev/null 2>&1; then
  env_exists=true
fi
if [[ "$env_exists" == false ]]; then
  run conda create --yes --name prompt-search --no-default-packages python=3.12.15 pip
  run conda activate prompt-search
else
  echo "Using existing environment: $CONDA_PREFIX"
fi

# Activation in a dry run cannot resolve the path of an environment not yet created.
env_python=python
if [[ "$dry_run" == false || "$env_exists" == true ]]; then
  env_python="$CONDA_PREFIX/bin/python"
fi
run "$env_python" -I -B -c 'import sys; assert sys.version_info[:2] == (3, 12), "prompt-search must use Python 3.12"'
if [[ "$env_exists" == false || "$install_existing" == true ]]; then
  # Conda supplies Python/pip; pip installs vLLM and the matching PyTorch build.
  # https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
  run "$env_python" -I -B -m pip --isolated install --upgrade pip --index-url https://pypi.org/simple
  run "$env_python" -I -B -m pip --isolated install \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --requirement "$project_dir/requirements-prompt-search.txt"
else
  echo "Checking only. Use --install to install or repair the pinned dependencies."
fi
run "$env_python" -I -B -m pip --isolated check
check_args=()
if [[ "$skip_cuda" == true ]]; then
  check_args+=(--skip-cuda-check)
fi
run "$env_python" -I -B "$project_dir/scripts/check_prompt_search_env.py" "${check_args[@]}"
if [[ "$dry_run" == true ]]; then
  echo "Dry run complete; no environment or package changes were made."
else
  echo "Environment checks passed. Activate it in your terminal: conda activate prompt-search"
fi
