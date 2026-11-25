# Full-prompt component search on DepSign

Elite selection finished G4 with a higher retained best and mean macro-F1. On the same 300 separate development posts, the winner from the historical tournament run scored slightly higher. The paired difference (elite minus tournament) was −0.0062, with a 95% bootstrap interval of [−0.0404, +0.0241].

| Survivor policy | G0 best / mean on dev_100 | G4 best / mean on dev_100 | G0 winner on 300 posts | Best discovered through G4 on 300 posts |
|---|---:|---:|---:|---:|
| Tournament | 0.3943 / 0.3592 | 0.4211 / 0.3835 | 0.3479 | 0.3969 (G3) |
| Elite (top 10) | 0.3943 / 0.3592 | 0.4278 / 0.3971 | 0.3479 | 0.3907 (G4) |

Both runs evaluated identical G0 and G1 candidates with matching scores. Their survivor sets first differed at G1. Each evaluated 48 complete prompts through G4. This comparison evolves the task description, label definitions and instruction together with 10 candidates at seed 11. The instruction experiment evolves one field with five candidates at each of three seeds.

[Fitness curves](component_fitness.png) · [Scores, paired intervals and generation-level audit](component_comparison.json)
