# Evolutionary Prompt Search for Mental Health Classification

This project investigates whether evolutionary prompt search improves mental health text classification with frozen LLM weights. Planned comparisons include human-written prompts, independent random search, ordinary EA initialization, and **validity- and reliability-inspired initialization**.

Currently, the pipeline evaluates fixed instructions from a JSON file on DepSign development data. EA and transfer experiments are not yet implemented. The same Qwen3.5-9B checkpoint supports classification and optional candidate generation, with separate greedy and sampling settings. The fixed-file pipeline only classifies; it does not generate prompts.

## Quick start

Prerequisites: Conda, Linux with an NVIDIA GPU and a working driver, model weights in `LLM/Qwen3.5-9B`, and the prepared development data below. Run commands from the project root.

Create the environment:

```bash
./setup_env.sh
```

This creates `prompt-search` with Python 3.12.15 and installs the [pinned core packages](requirements-prompt-search.txt): vLLM 0.32.0, Transformers 5.19.0, and PyTorch 2.13.0 with CUDA 13.2. It checks dependencies, imports, and a small GPU computation. Existing environments are checked without reinstalling; use `--install` to install the pinned packages into one. Use `--dry-run` to preview commands, or `--skip-cuda-check` on a node without GPU access. Model weights are downloaded separately.

Start the model server in one terminal:

```bash
./start_vllm.sh
```

Keep it running. In a second terminal, evaluate every instruction in the prompt file on the fixed 100-post sample:

```bash
conda activate prompt-search
python -B scripts/run_depsign_pipeline_smoke.py \
  --prompts-file prompts/humanPrompt.json \
  --data-file data/processed/depsign/dev_100.tsv
```

Add `--prepare-only` to validate and save inputs without contacting the model server. For the complete cleaned dev set, use `--data-file data/processed/depsign/dev_v1/dev_clean.tsv`; this is also the default when `--data-file` is omitted.

## Prompts and data

Edit [prompts/humanPrompt.json](prompts/humanPrompt.json): each entry in `prompts` has a unique `id` and an `instruction`. The file can contain human-written and model-generated instructions; record their actual provenance. Shared label definitions and output requirements are in [prompts/depsign_task_v1.txt](prompts/depsign_task_v1.txt).

Each prompt evaluates the same posts in the same order. Classification uses temperature 0, seed 0, and thinking disabled. Posts longer than 1,500 tokenizer tokens are truncated; all records remain in the evaluation.

| Data file under `data/processed/depsign/` | Posts | Purpose |
| --- | ---: | --- |
| `dev_v1/dev_clean.tsv` | 4,480 | Complete dev set after removing internal duplicates and conflicting texts |
| `dev_100.tsv` | 100 | Fixed proportional subset for development experiments |

`dev_100.tsv` contains 41 `not_depression`, 51 `moderate`, and 8 `severe` posts, sampled without replacement using seed 1729. Its source, selected IDs, and checksums are recorded in `dev_100.manifest.json`; the sampling code is [sample_depsign_dev.py](scripts/sample_depsign_dev.py). Development scores guide prompt selection; final evaluation requires held-out data.

## Results

Each run creates `data/processed/depsign/pipeline_smoke_runs/<timestamp>/`:

- `summary.csv` / `summary.json`: per-prompt macro-F1, invalid-output rate, counts, and run status.
- `prompts_snapshot.json`: the prompt file used for that run.
- `evaluations/<id>/`: posts, requests, raw responses, predictions, detailed metrics, and configuration metadata.

Invalid outputs and failed requests count as errors and stay in the denominator. Runs use `N × M` classification requests for N prompts and M posts; there is no caching or resume support yet.

## Research scope

The search method draws on **EvoPrompt (Guo et al., 2024)**. **Arnaout et al. (2026), Table 3**, supplies the theoretical inspiration for initialization. Its evaluation framework does not establish that putting these principles into prompts improves classification; testing that conversion is this project's hypothesis.

The planned stability checks cover search-seed variation and prediction consistency under equivalent prompt rewrites. They do not validate the full framework: links to external outcomes, temporal reliability, population reliability, and internal consistency require independent evidence.

DepSign is the current source task. Reddit C-SSRS is planned for transfer across tasks. Dataset B, a second comparable depression dataset, remains undetermined, so transfer across datasets for the same task has not been tested.

The [experiment configuration](configs/experiment_v1.json) and [protocol](docs/experiment_protocol.md) remain drafts. See the [pipeline guide](docs/depsign_pipeline_smoke.md) for detailed usage.

## References

Full citations, including original dataset publications, are in [references.bib](references.bib).

| Source | Role |
| --- | --- |
| [Guo et al., 2024 — EvoPrompt](https://openreview.net/forum?id=ZG3RaNIsO8) · [Code](https://github.com/beeevita/EvoPrompt) | Evolutionary search method |
| [Arnaout et al., 2026](https://aclanthology.org/2026.acl-long.347/) | Validity and reliability framework inspiring initialization |
| [Xu et al., 2024 — Mental-LLM](https://doi.org/10.1145/3643540) | Prompting for mental health prediction |
| [Beck et al., 2024](https://aclanthology.org/2024.eacl-long.159/) | Prompt sensitivity; not specific to mental health |
| [Kayalvizhi S and Thenmozhi D, 2022 — DepSign](https://arxiv.org/abs/2202.03047) · [Published version](https://doi.org/10.1007/978-3-031-16364-7_11) · [Shared task](https://aclanthology.org/2022.ltedi-1.51/) | Dataset construction and annotation |
| [Gaur et al., 2019](https://doi.org/10.1145/3308558.3313698) · [C-SSRS dataset](https://zenodo.org/records/2667859) | User-level suicide-risk data and expert annotation |
| [Milintsevich et al., 2024](https://aclanthology.org/2024.clpsych-1.13/) | Examining what dataset labels measure |
