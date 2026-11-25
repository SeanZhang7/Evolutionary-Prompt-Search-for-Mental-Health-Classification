# Finding a rising best and mean fitness curve

The curve tracks the best and arithmetic mean macro-F1 of the prompts that actually survive each generation. With fixed scores, keeping the best N from the union of parents and children guarantees that both metrics are nondecreasing. A child must outscore the previous best to move the best curve. Improving on one of the weaker survivors can raise the mean even while the best stays flat.

## An observed 20-generation example

A completed DepSign three-class instruction search used the few-shot task template, the baseline operator, tournament parent selection, population 10 and seed 11. Crossover and mutation probabilities were both 0.8. Generation temperature was 0.8 and classification temperature was zero. The archive specifies how retained scores were calculated and omits the survivor-policy setting, which limits attribution.

| Generation | Retained best | Retained mean |
|---|---:|---:|
| G0 | 0.4395 | 0.3772 |
| G4 | 0.4442 | 0.4308 |
| G8 | 0.4698 | 0.4565 |
| G12 | 0.4779 | 0.4666 |
| G16 | 0.4779 | 0.4680 |
| G20 | 0.4779 | 0.4680 |

The final three generations evaluated seven, seven and five new children. None improved the final population scores. Applying the follow-up's plateau rule retrospectively to all 10 completed DepSign instruction runs in this archive with at least 12 generations, five met it over G9–G12 and eight met it over their last four observations. All used seed 11 across different settings. [20-generation curve and counts of new children](archive_baseline_fewshot_curve.png) · [Example run](../../data/processed/depsign/ga_runs/baseline%20fewshot/performance.json)

## Error-feedback editing with elite retention

On DepSign three-class dev_100, Qwen critiques up to three labeled mistakes made by each parent and rewrites its instruction in a separate request. Each generation evaluates the new unique instructions and retains the five highest-scoring parents or children. All three seeds started with the same five instructions. From G0 to G4, retained mean rose in all 12 seed-by-generation transitions. The number of strict best increases was two for seed 11, three for seed 29 and one for seed 47. The G4 results motivated a fixed extension through G12 to measure late behavior. [G4 curves](feedback_curve_g4.png) · [Follow-up protocol](../curve_protocol.md)

All seeds started at best 0.3745 and mean 0.3596. Across 36 transitions, retained best improved eight times and mean improved 26 times, with zero decreases. Each G10, G11 and G12 produced newly evaluated children for every seed. Two seeds met the predeclared late-plateau criterion of at most 0.005 variation in both metrics over G9–G12.

| Search seed | G12 best | G12 mean | Strict increases in best / mean | G9–G12 best / mean range |
|---|---:|---:|---:|---:|
| 11 | 0.4255 | 0.4166 | 3 / 8 | 0.0000 / 0.0006 |
| 29 | 0.4434 | 0.4279 | 4 / 9 | 0.0160 / 0.0113 |
| 47 | 0.4314 | 0.4178 | 1 / 9 | 0.0000 / 0.0005 |

Seed 11 shows early progress, a later record and a tail plateau. Seed 29 found a new best at G12 after eight flat best transitions. Seed 47 reached its highest score at G1 while its weaker survivors kept improving. Elite retention protects each generation's scores; fresh records depend on what the editor discovers. [All three G0–G12 curves](feedback_curve_g12.png) · [Underlying data](feedback_curve_g12.json) · [Reproduction commands](../README.md)

These curves describe selection on the 100 search posts. Confirmation scores on the additional 300 posts are unavailable for the feedback winners. The local inference endpoint refused the connection when confirmation was attempted, and the completed comparison-arm cache contains none of the three G12 feedback winners. Their performance on those posts remains unknown.

The separate three-component experiment paired survivor policies from identical first-generation proposals. It used ten full prompts at seed 11. Elite retention improved the search curves through G4, while the interval for the paired difference on 300 separate development posts included zero. [Component results](component_findings.md)
