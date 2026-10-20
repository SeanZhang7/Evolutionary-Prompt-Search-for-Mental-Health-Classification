"""Check pinned package versions, imports, and a small CUDA computation."""

import argparse
from importlib.metadata import version
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cuda-check", action="store_true")
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Expected Python 3.12.")
    requirements = Path(__file__).resolve().parents[1] / "requirements-prompt-search.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, expected = line.strip().split("==")
        actual = version(name)
        print(f"{name}: {actual}")
        if actual != expected:
            raise SystemExit(f"Expected {name}=={expected}; use ./setup_env.sh --install to install the pinned versions.")

    import torch
    import transformers
    import vllm

    print("Package imports: OK")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if torch.version.cuda != "13.2":
        raise SystemExit("Expected the CUDA 13.2 PyTorch build.")
    if args.skip_cuda_check:
        print("CUDA device check skipped; GPU inference has not been verified.")
        return
    if not torch.cuda.is_available():
        raise SystemExit("Packages are installed, but CUDA is unavailable. Check the NVIDIA driver and GPU allocation, then rerun ./setup_env.sh.")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    result = (torch.ones(1, device="cuda") + 1).item()
    if result != 2.0:
        raise SystemExit(f"Unexpected CUDA computation result: {result}")
    print(f"CUDA computation: {result}")


if __name__ == "__main__":
    main()
