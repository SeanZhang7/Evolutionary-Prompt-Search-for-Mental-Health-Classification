# Experiments on declining or flat fitness

The [findings](results/findings.md) include an archived 20-generation example and the [three-seed follow-up curves](results/feedback_curve_g12.png).

All commands run from the repository root in the `prompt-search` environment. The model-backed commands use the existing local Qwen service at `127.0.0.1:8000` with an 8192-token context. Each run is isolated from the project's usual GA output.

## Keep strong component combinations

To evolve task definition, label definitions and instruction together, sort each pool of parents and evaluated offspring by development macro-F1 and retain the top N:

```bash
PROMPT_SEARCH_CONTEXT_TOKENS=8192 python -B experiment/component_search.py \
  --label-mode three_class --survivors elite --seed 11 --generations 4
```

The creative label-definition operator and the other two component operators are loaded from the project configuration. Use `--survivors tournament` for the original survivor rule or `--label-mode binary` for binary DepSign. The command prints the output directory; `ga/best_template.txt` contains the complete winning prompt. On fixed, deterministic development scores, retaining the top N makes both the current best and population mean nondecreasing. Progress beyond a plateau requires a child with a higher score.

The [archived comparison](results/replay.png) covers all four completed DepSign component runs with individual scores, all at seed 11. Across 60 generations, retained best dropped eight times and retained mean dropped 22 times. With the very same parents and evaluated children, retaining the top ten gave a higher best in 12 generations and a higher mean in 59. Only 11 generations discovered a new best. These paired comparisons measure the immediate effect of selection. Later children in the archive were generated from the actual tournament survivors. Regenerate the analysis with `python -B experiment/replay.py`.

## Compare discovery methods

The matched instruction experiment compares GA with tournament survival, GA with top-N survival, textual differential evolution with parent-versus-child survival, and independently sampled proposals with top-N survival. The two GA arms share first-generation proposals. All methods have equal proposal slots and use the same initial five instructions. The [protocol](protocol.md) specifies selection, budgets, development splits and the source paper.

```bash
PROMPT_SEARCH_CONTEXT_TOKENS=8192 python -B experiment/search.py \
  --mode three_class --population 5 --generations 4 --seeds 11 29 47 \
  --search-size 100 --confirmation-size 300 --workers 2 \
  --output experiment/runs/instruction_suite

python -B experiment/report.py --suite experiment/runs/instruction_suite
```

Repeating the search command with the same configuration resumes its private results. Each arm records its retained population and best discovery at every generation. After **all** searches finish, the runner freezes each arm's best search candidate and measures it on 300 other development posts alongside the initial winner. The report plots search curves and paired confirmation changes. Its bootstrap intervals reflect uncertainty across the sampled posts, conditional on these seeds. Counts of actual API calls account for reuse across identical requests. Raw run directories contain local evaluation artifacts and are ignored by Git.

The independent proposal arm adapts the generate-and-score idea in Zhou et al., without APE's demonstration-conditioned instruction generator. The separate confirmation split addresses selection bias from optimizing and reporting on the same sample, as studied by Cawley and Talbot. [Zhou et al., ICLR 2023](https://arxiv.org/abs/2211.01910) · [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)

For the matched four-generation component comparison described in the protocol, use `--output-dir experiment/runs/components_elite_g4` in the component command above. After that run and after freezing the primary suite protocol, score complete combinations using:

```bash
PROMPT_SEARCH_CONTEXT_TOKENS=8192 python -B experiment/confirm_components.py \
  --suite experiment/runs/instruction_suite \
  --elite experiment/runs/components_elite_g4/ga/performance.json \
  --output experiment/runs/component_confirmation \
  --summary experiment/results/component_comparison.json
```

The archived tournament report is needed locally for this last comparison. Its first four generations serve as the reference. The script checks input hashes, original and effective settings, and how well the first common generation reproduces.

## Learn from search mistakes

The supplementary arm uses two calls per valid editing attempt. Qwen first critiques a parent using up to three labeled mistakes from that parent's search predictions. A separate request rewrites the instruction. Only the resulting instruction reaches the classifier. The same five starting prompts, generation count, seeds and classifier budget permit a paired comparison with GA elite. The extra critic, use of labeled error cases and one-parent editing are part of this method. [ProTeGi paper](https://aclanthology.org/2023.emnlp-main.494/)

For a fresh 12-generation run of the feedback method, initialize the shared G0 and run the feedback arm. The first command scores the starting instructions once. The second performs the search. The third plots the actual retained population for every seed.

```bash
export PROMPT_SEARCH_CONTEXT_TOKENS=8192
python -B experiment/search.py --stage init --mode three_class --population 5 \
  --generations 12 --seeds 11 29 47 --search-size 100 --confirmation-size 300 \
  --workers 2 --output experiment/runs/fresh_g0
python -B experiment/feedback.py --stage search --primary experiment/runs/fresh_g0 \
  --output experiment/runs/fresh_feedback
python -B experiment/curve_report.py --suite experiment/runs/fresh_feedback \
  --output experiment/runs/fresh_feedback/plots
```

After the primary suite has scored its initial prompts, start the supplementary search. Once both the supplementary winners and the primary confirmation results exist, run confirmation.

```bash
PROMPT_SEARCH_CONTEXT_TOKENS=8192 python -B experiment/feedback.py --stage search
PROMPT_SEARCH_CONTEXT_TOKENS=8192 python -B experiment/feedback.py --stage confirm
```

To regenerate the primary and full-component comparison plots from completed results:

```bash
python -B experiment/report.py --suite experiment/runs/instruction_suite \
  --component-comparison experiment/results/component_comparison.json
```

## Continue feedback search to G12

We chose the feedback method after inspecting G0–G4, so those four generations are exploratory. The follow-up continues the three saved G4 populations (seeds 11, 29 and 47) to a fixed G12 on the same DepSign `dev_100`. Each seed gets five offspring slots per generation, including after a flat generation. [Follow-up protocol](curve_protocol.md)

```bash
export PROMPT_SEARCH_CONTEXT_TOKENS=8192
python -B experiment/extend_feedback.py --stage prepare
python -B experiment/extend_feedback.py --stage search
python -B experiment/curve_report.py --suite experiment/runs/feedback_g12 --output experiment/results
python -B experiment/extend_feedback.py --stage confirm
```

Preparation verifies the saved scores, parent order and model settings offline. Search resumes in `experiment/runs/feedback_g12/`. The plot and per-generation values go to `experiment/results/feedback_curve_g12.png` and `feedback_curve_g12.json` in the same directory. For each seed, the predeclared late-plateau criterion requires a G9–G12 range of at most **0.005 macro-F1** for both retained best and retained mean, plus at least one unique, valid child evaluated in each of G10, G11 and G12.

Confirmation compares the frozen G12, G4 and initial winners on the same 300 separate development posts. Paired differences and intervals are saved in `experiment/runs/feedback_g12/confirmation.json`.
