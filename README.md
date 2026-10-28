# Evolutionary Prompt Search for Mental Health Classification

This project investigates whether evolutionary prompt search improves mental health text classification with frozen LLM weights. Planned comparisons include human-written prompts, independent random search, ordinary EA initialization, and **validity- and reliability-inspired initialization**.

The project supports fixed-file prompt evaluation and configurable GA search on DepSign development data. Transfer experiments are not yet implemented. The same Qwen3.5-9B checkpoint supports classification and candidate generation, with separate greedy and sampling settings. The fixed-file evaluation pipeline only classifies; it does not generate prompts.

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

## Evaluate one prompt

Copy an instruction block into `prompts/BestPrompts/best_prompt.txt`, then run:

```bash
python -B scripts/evaluate_depsign_prompt.py
```

Defaults: that prompt file and `data/processed/depsign/dev_100.tsv`. To select other inputs:

```bash
python -B scripts/evaluate_depsign_prompt.py \
  --prompt-file prompts/BestPrompts/best_prompt.txt \
  --data-file data/processed/depsign/dev_100.tsv
```

The text file contains only the instruction block; classification uses the same fixed task template and parameters as GA. Dataset files must use the existing DepSign TSV schema (`PID`, `Text data`, `Label`) and labels. `--prepare-only` validates inputs without contacting vLLM; `--output-dir` selects a new result directory.

Each evaluation writes `Results/depsign/<timestamp>/`:

- `report.md`: per-class precision, recall, F1, one-vs-rest accuracy, overall accuracy, macro-F1, confusion counts, and directional error rates.
- `performance.json`: exact metrics, error-analysis summaries, the evaluated instruction/template, input hashes, and settings.
- `predictions.csv`: every evaluated post, its label/prediction, actual input text, raw output, and request/format diagnostics.
- `errors.csv`: only incorrect or invalid predictions, for inspection.

Per-class **A** means `(TP + TN) / n` with that class positive and all other outputs negative; it differs from recall. Overall accuracy counts every invalid output/request failure as incorrect. Confusion rates divide by the true class's support, and ties are reported. Invalid outputs are separate from wrong-label confusions. Reports are written once on completion or handled interruption; partial evaluations are marked explicitly. No prompt generation or search occurs.

## Binary DepSign

Add `--label-mode binary` to GA, fixed-file evaluation, or single-prompt evaluation. Binary mode uses `prompts/depsign_task_Bi.txt`; GA and fixed-file evaluation default to `prompts/humanPrompt_Bi.json`. All three runners use the existing `dev_100.tsv` by default in this mode.

```bash
python -B scripts/run_depsign_ga.py --label-mode binary --generations 20 --prepare-only
python -B scripts/run_depsign_ga.py --label-mode binary --generations 20

# Evaluate the fixed binary prompts without search:
python -B scripts/run_depsign_pipeline_smoke.py --label-mode binary

# Evaluate one binary instruction saved from a run:
python -B scripts/evaluate_depsign_prompt.py --label-mode binary --prompt-file path/to/best_prompt.txt
```

Source TSV labels are mapped in memory: `not depression` → `not_depression`, and `moderate` / `severe` → `depression`. The same 100 posts become 41 `not_depression` and 59 `depression`; source files are not rewritten or resampled. Model outputs must match one of the two target labels; `moderate` and `severe` outputs are invalid. Macro-F1 averages the two class F1 scores, including a zero score for an absent class. Binary and three-class macro-F1 measure different tasks and should not be compared as the same metric.

Binary runs retain the existing output files and locations, with `binary_` prefixed to automatic run-directory names and the label mapping recorded in reports. GA still saves the best instruction, performance, and final generation-best-fitness curve; single-prompt evaluation reports two-class P/R/F1/A and directional errors. Search settings and early stopping are unchanged. Omit `--label-mode` (or use `--label-mode three_class`) for the original task. The original `humanPrompt.json` is preserved; binary wording adaptations are recorded in `humanPrompt_Bi.json`.

## GA search

With the model server running, use the existing prompt file as the initial population:

```bash
python -B scripts/run_depsign_ga.py --generations 5 --prepare-only
python -B scripts/run_depsign_ga.py --generations 5
```

Defaults: `prompts/humanPrompt.json`, `data/processed/depsign/dev_100.tsv`, and seed 11. `--generations G` accepts a positive integer and overrides `search.generations_after_initialization` in `configs/experiment_v1.json` (currently 1). G is the maximum number of evolution steps after G0. Other options are `--prompts-file`, `--data-file`, `--seed`, and `--output-dir`. Preparation makes no HTTP requests; it checks all initial classification requests and a sample parent pair. Actual selected pairs are checked before generation. Each command creates a separate run; there is no resume support.

Early stopping uses `search.early_stopping_patience` (default 5). After each completed generation, compare retained-population mean macro-F1 with the previous generation, starting with G0. Five consecutive unchanged comparisons stop the search; any change resets the counter. Use full-precision scores with an absolute tolerance of 1e-12 for floating-point noise, not the four-decimal plot labels. Early stopping is successful completion: the final report records `stopped_early`, `stop_reason`, requested and completed generations, and the last unchanged-generation count. It still saves the best prompt and one final curve. Failed or partially evaluated generations do not count.

Choose parent selection with `search.parent_selection` in `configs/experiment_v1.json` (default `"wheel"`), or override it for one run:

```bash
python -B scripts/run_depsign_ga.py --label-mode binary --generations 20 --parent-selection tour
```

| Mode | Parent selection for each child |
| --- | --- |
| `wheel` | Draw an N-entry pool with replacement, weighted by macro-F1, then sample two distinct pool positions. The positions may contain the same original individual. All-zero fitness uses uniform pool sampling. |
| `random` | Uniformly sample two distinct individuals from the current population, ignoring scores. |
| `tour` | Independently sample two groups of two distinct individuals. Each group's higher-scoring individual becomes a parent; ties favor the first drawn. Groups may overlap and both winners may be the same individual. |

These rules follow [EvoPrompt's GA selection modes](https://github.com/beeevita/EvoPrompt/blob/main/evoluter.py). All modes use the search seed. The old config value `two_stage_roulette_with_replacement` remains an alias for `wheel`, preserving its existing seeded draws. `performance.json` records the effective mode; `--save-details` also records sampled groups/pools and parent IDs in `selection.json`.

Each generation proposes N children, evaluates eligible children, and selects the next N parents by macro-F1 from parents plus children. Survivor ties favor earlier creation. Surviving individuals keep their measured scores.

Set `search.deduplication` to `"exact_text"` (enabled) or `"none"` (disabled). When enabled, initial duplicate instructions are merged in file order; N is the remaining count and must be at least two. A child matching any earlier candidate in this run is discarded **before classification**, including copies and duplicates within the same generation. It consumes its proposal slot but triggers neither evaluation nor replacement generation. Select the best N from existing parents and the remaining children; parents ensure the population stays at N. Comparison strips outer whitespace but preserves case, punctuation, and internal whitespace. This moves deduplication earlier than [EvoPrompt's GA topk mode](https://github.com/beeevita/EvoPrompt/blob/main/evoluter.py#L393-L424), which evaluates children before deduplicating the union. With `"none"`, every child is evaluated separately. No previous run's scores are reused, and the supplied prompt file is not modified.

`search.crossover_probability` and `search.mutation_probability` control the two operation probabilities (currently **0.9** and **0.8**). Each child draws both flags independently. Both enabled: cross then mutate in one Qwen request; crossover only: use its result unchanged; mutation only: mutate the first parent; neither: copy the first parent without a generation request. Single-operation requests return only the relevant field; the program copies crossover-only results into the final instruction. Generation sees reference definitions rather than the full classifier template. Retry attempts keep the same parents and flags. Selection, operation draws, and generation requests use reproducible seeds that include the generation index. Set the generation temperature only in `model.generation.temperature` (currently **0.75**). It must be a finite number greater than zero. Classification remains fixed at temperature 0; other generation settings still follow the configured protocol. Edit the operation instructions in [crossover_and_mutation_prompt.json](prompts/crossover_and_mutation_prompt.json). The outer request template remains [ga_crossover_mutation_v1.txt](prompts/ga_crossover_mutation_v1.txt).

The operator JSON contains a `prompts` list. Each entry has a unique `name`, optional `description`, and three instruction strings: `both`, `crossover_only`, and `mutation_only`. The supplied `baseline` preserves the original rendered instructions. Token-limit placeholders `{{MAX_CROSSOVER_TOKENS}}` and `{{MAX_INSTRUCTION_TOKENS}}` are filled by the program. Keep the existing output fields: both operations require `crossover` and `instruction`; crossover-only requires `crossover`; mutation-only requires `instruction`. Output parsing and length checks still apply.

To compare another version, duplicate the `baseline` entry in the same JSON, assign a new name (for example `v2`), edit its instructions, and select it with `--operator-prompt-name v2`. The default is `baseline`; unknown or duplicate names are rejected. `--operator-prompt-file` can still select a different file, and the earlier single-entry JSON format remains supported. The runner reads the file once at startup and validates the selected entry; `performance.json` records its name, file path, and SHA-256. Detailed mode also saves a snapshot. Setting `search.crossover_probability` to `0` disables crossover; both probabilities at `0` copy the first parent without a generation request.

`--operator-prompt-name v2` enables performance-guided editing (`"feedback": "parent_metrics"`). Each parent record includes only its instruction, macro-F1, and a labeled confusion matrix (true-label rows, prediction columns, including INVALID). Mutation-only receives the first parent; crossover receives both. The feedback comes from existing evaluations on the same development file; it adds no classification requests. V2 asks the generator to combine complementary guidance and choose the direction and extent of mutation from the error patterns, while preserving label definitions. The intermediate crossover has no measured score. Execution probabilities still come from the config; feedback does not change them.

Baseline defaults to `"feedback": "none"` and keeps its original requests. Performance feedback is kept in memory, with its mode recorded in the existing report. Preparation uses explicitly unscored placeholders because parent scores do not exist yet; every actual request is checked with real feedback against the context limit before generation. No parents or metrics are silently truncated. This is a development-feedback search variant, not a claim of improved held-out performance.

By default, a successful run keeps only three files in `data/processed/depsign/ga_runs/<timestamp>/`:

- `best_prompt.txt`: the best instruction block, selected by development macro-F1.
- `performance.json`: its score, invalid-output rate, run status, settings, generation means, skipped invalid child slots, and the most recent generation issue (reason and raw output), if any.
- `best_fitness.png`: one final curve of retained-population maximum macro-F1 from G0 to the last completed generation. `performance.json` also records distinct prompt counts, skipped duplicates, and actual evaluations.

Default GA runs use memory only during search: inputs and tokenizer checksums are read once at startup, and no detailed temporary files are created. Input-file edits apply to the next run. Reports are written and the curve is drawn once at completion, handled failure, or Ctrl+C; interrupted runs include only completed evaluations and generations. Add `--save-details` only when persistent snapshots and raw per-candidate logs are needed for debugging. Preparation alone writes only `performance.json`. Plotting uses Matplotlib, included in the environment requirements.

For N initial prompts after deduplication, M posts, and G evolution steps, `(G + 1) × N × M` is the classification-request upper bound. Early stopping can reduce this budget further. Each skipped duplicate or invalid child saves M classification requests; N=10, M=100, G=5 requires at most 6,000. Reports distinguish this budget from actual evaluations and completed requests. Generation requests depend on operation draws: before retries, the expected number is `G × N × [1 − (1 − p_cross)(1 − p_mutate)]`.

Crossover and final instructions each have a 256-token ceiling; the full JSON output remains limited to 512 tokens, with context checks before inference. After three invalid generation outputs, discard that child slot without replacement and continue. Parents keep the population at N even if all children fail. Reports record skipped slots and per-generation counts; `complete` means the generation limit or early-stopping criterion was reached, not that every proposed child was valid. Transient request errors allow one retry, then stop; other service errors and classification request failures also stop the search. Structural validation does not establish semantic correctness or improved performance.

## Prompts and data

Edit [prompts/humanPrompt.json](prompts/humanPrompt.json): each entry in `prompts` has a unique `id` and an `instruction`. The file can contain human-written and model-generated instructions; record their actual provenance. Shared label definitions and output requirements are in [prompts/depsign_task_v1.txt](prompts/depsign_task_v1.txt).

Each prompt evaluates the same posts in the same order. Classification uses temperature 0, seed 0, and thinking disabled. Posts longer than 1,500 tokenizer tokens are truncated; all records remain in the evaluation.

Shared data loading, request preparation, classification, and scoring live in `scripts/depsign_evaluator.py`. Both evaluation and GA entrypoints use this module; data files are explicit and every supplied record is evaluated.

`scripts/qwen_client.py` handles generation settings and individual Qwen generation requests. It is called by GA, not run separately. Initial prompts always come from the supplied JSON file; only offspring are generated during search.

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

Invalid outputs and failed requests count as errors and stay in the denominator. Each `INVALID` classification prints its raw output, finish reason, request error, and reasoning in the terminal without adding result files or changing label parsing. Runs use `N × M` classification requests for N prompts and M posts; there is no caching or resume support yet.

## Research scope

The search method draws on **EvoPrompt (Guo et al., 2024)**. **Arnaout et al. (2026), Table 3**, supplies the theoretical inspiration for initialization. Its evaluation framework does not establish that putting these principles into prompts improves classification; testing that conversion is this project's hypothesis.

The planned stability checks cover search-seed variation and prediction consistency under equivalent prompt rewrites. They do not validate the full framework: links to external outcomes, temporal reliability, population reliability, and internal consistency require independent evidence.

DepSign is the current source task. Reddit C-SSRS is planned for transfer across tasks. Dataset B, a second comparable depression dataset, remains undetermined, so transfer across datasets for the same task has not been tested.

The [experiment configuration](configs/experiment_v1.json) contains settings used by the current runner. Formal comparisons remain unvalidated; the research scope above describes planned work. See the [pipeline guide](docs/depsign_pipeline_smoke.md) for detailed usage.

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
