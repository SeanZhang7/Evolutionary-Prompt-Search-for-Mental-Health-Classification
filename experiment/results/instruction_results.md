# DepSign instruction experiments

Search used 100 posts. Confirmation used the same 300 separate development posts for every frozen winner. The table averages over seeds 11, 29, 47. The initial winner scored 0.3745 on search and 0.3700 on confirmation.

| Method | Final retained mean | Best found on search | Confirmation | Change from initial on confirmation [paired 95% interval] |
|---|---:|---:|---:|---:|
| GA tournament | 0.3812 | 0.4014 | 0.3591 | -0.0109 [-0.0369, +0.0158] |
| GA elite | 0.3870 | 0.3970 | 0.3680 | -0.0020 [-0.0271, +0.0234] |
| Textual DE | 0.3837 | 0.4038 | 0.3626 | -0.0075 [-0.0270, +0.0122] |
| Independent | 0.3701 | 0.3915 | 0.3713 | +0.0013 [-0.0167, +0.0184] |

Each confirmation candidate was picked by search macro-F1 over all generations. The intervals resample posts within gold class and condition on the observed seeds. Shared prompts and cached requests reduce actual inference work. The suite used the same number of proposal slots per arm.

[Per-generation curves](live_fitness.png) · [Per-seed results, costs and class F1](live_summary.json)
