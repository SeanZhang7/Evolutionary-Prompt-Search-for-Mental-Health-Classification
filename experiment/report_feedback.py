"""Export search curves, paired confirmation contrasts and role-specific costs without post or prompt text."""

import argparse
from collections import Counter
import json
from math import isclose
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __package__:
    from .report import checked_history, read
else:
    from report import checked_history, read


def build(primary_path, feedback_path, output):
    primary = read(primary_path / "protocol.json")
    feedback = read(feedback_path / "protocol.json")
    confirmation = read(feedback_path / "confirmation.json")
    if (feedback["primary_protocol_hash"] != primary["protocol_hash"]
            or confirmation["protocol_hash"] != feedback["protocol_hash"]
            or confirmation["status"] != "complete"):
        raise ValueError("Feedback confirmation and primary protocol do not match")
    seeds = primary["seeds"]
    generations = list(range(1 + primary["generations"]))
    contrast = confirmation["contrasts"]
    expected = {(ref, seed) for ref in ("G0", "elite") for seed in [*seeds, "mean_over_observed_seeds"]}
    if expected != {(row["baseline"], row["seed"]) for row in contrast} or len(contrast) != len(expected):
        raise ValueError("Incomplete paired feedback comparisons")
    results = {}
    for arm, suite, protocol in (("elite", primary_path, primary), ("feedback", feedback_path, feedback)):
        checked = [checked_history(suite, protocol, arm, seed) for seed in seeds]
        histories = [{**{k: r[k] for k in ("seed", "record_generations", "unique_evaluated", "slots")},
                      **{m: [g[m] for g in r["history"]] for m in ("best", "mean")}} for r in checked]
        role_cost = {}
        for r in checked:
            for role, v in r["cost"].items():
                tally = role_cost.setdefault(role, Counter())
                tally.update(v)
                if v.get("logical_requests", 0) != v.get("actual_requests", 0) + v.get("cached_requests", 0):
                    raise ValueError(f"Inconsistent request accounting for {arm}, seed {r['seed']}, {role}")
        results[arm] = {"per_seed": histories, "search_cost": role_cost,
                        "mean_over_seeds": {m: [fmean(values) for values in zip(*(r[m] for r in histories))]
                                            for m in ("best", "mean")}}
    for seed in seeds:
        found = {r["arm"]: r for r in confirmation["results"] if r["seed"] == seed}
        if set(found) != {"feedback", "elite", "G0"}:
            raise ValueError(f"Incomplete confirmation roster for seed {seed}")
        for ref in ("G0", "elite"):
            row = next(x for x in contrast if x["baseline"] == ref and x["seed"] == seed)
            difference = found["feedback"]["confirmation_f1"] - found[ref]["confirmation_f1"]
            if not isclose(row["delta"], difference, abs_tol=1e-12, rel_tol=0):
                raise ValueError("Paired difference disagrees with reported raw scores")
        for m in ("best", "mean"):
            i = seeds.index(seed)
            if results["feedback"]["per_seed"][i][m][0] != results["elite"]["per_seed"][i][m][0]:
                raise ValueError("Feedback and primary GA have different initial populations")
    data = {"scope": confirmation["scope"], "source": "ProTeGi-inspired textual critique followed by one edit per target.",
            "seeds": seeds, "population": primary["population"], "generations": primary["generations"],
            "protocol_hashes": {"primary": primary["protocol_hash"], "feedback": feedback["protocol_hash"]},
            "methods": results, "confirmation": confirmation["results"], "paired_contrasts": contrast,
            "bootstrap": confirmation["bootstrap"], "confirmation_cost": confirmation["confirmation_cost"],
            "feedback_failed_attempt_cost_by_role": confirmation["failed_attempt_cost_by_role"],
            "accounting": "Actual requests and reported tokens include executed work; cached requests include work reused from other arms or seeds. Logical successful and invalid responses are counted for each consuming arm; see the separate failure ledger."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "feedback_summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55), sharex=True, sharey=True, layout="constrained")
    for ax, field, title in zip(axes, ("best", "mean"), ("Best retained", "Mean retained")):
        for arm, name, color in (("elite", "GA elite", "#0072B2"), ("feedback", "Error feedback", "#AF4800")):
            for row in results[arm]["per_seed"]:
                ax.plot(generations, row[field], color=color, alpha=.22, linewidth=1)
            ax.plot(generations, results[arm]["mean_over_seeds"][field], label=name,
                    color=color, marker="o", markersize=4, linewidth=2.2)
        ax.set(title=title, xlabel="Generation")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Search macro-F1")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"DepSign {primary['mode'].replace('_', '-')} · mean across {len(seeds)} seeds (individual seeds in faint lines)")
    fig.savefig(output / "feedback_fitness.png", dpi=160)
    plt.close(fig)
    diffs = [r for r in contrast if r["seed"] == "mean_over_observed_seeds"]
    for row in diffs:
        lo, hi = row["paired_95pct_interval"]
        print(f"Feedback minus {row['baseline']} on confirmation: {row['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return data


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=root / "runs/instruction_suite")
    parser.add_argument("--feedback", type=Path, default=root / "runs/feedback_suite")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    build(args.primary, args.feedback, args.output)
