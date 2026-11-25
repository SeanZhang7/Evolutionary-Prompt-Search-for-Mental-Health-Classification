"""Audit completed search histories and plot measured fitness and paired confirmation changes."""
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

NAMES = {"tournament": "GA tournament", "elite": "GA elite", "de": "Textual DE", "random": "Independent"}
COLORS = {"tournament": "#586573", "elite": "#0072b2", "de": "#c05d00", "random": "#00816b"}
EPS = 1e-12


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_history(suite, protocol, arm, seed):
    state = read(suite / "arms" / f"{arm}_{seed}.json")
    n, g = protocol["population"], protocol["generations"]
    history = state["generation_metrics"]
    if (state["status"] != "complete" or state["protocol_hash"] != protocol["protocol_hash"]
            or [r["generation"] for r in history] != list(range(g + 1)) or len(state["events"]) != n * g):
        raise ValueError(f"Incomplete or inconsistent run: {arm}/{seed}")
    candidates = {p["id"]: p for p in state["evaluated"]}
    for event in state["events"]:
        if arm in NAMES and event["status"] == "duplicate":
            proposal = read(suite / "cache/generation" / (event["generation_cache_key"] + ".json"))
            if proposal["value"]["instruction"] != candidates[event["duplicate_of"]]["instruction"]:
                raise ValueError(f"Duplicate text disagrees with its generation receipt: {arm}/{seed}/{event['id']}")
    for row in history:
        scores = [candidates[k]["macro_f1"] for k in row["retained"]]
        record = max(p["macro_f1"] for p in candidates.values() if p["generation"] <= row["generation"])
        if len(set(row["retained"])) != n or any(not isclose(a, b, abs_tol=EPS, rel_tol=0) for a, b in
                [(max(scores), row["best"]), (fmean(scores), row["mean"]), (record, row["incumbent"])]):
            raise ValueError(f"Retained scores disagree: {arm}/{seed}/G{row['generation']}")
        gnum = row["generation"]
        if gnum and arm != "tournament":
            previous = history[gnum - 1]["retained"]
            events = [e for e in state["events"] if e["generation"] == gnum]
            if arm == "de":
                expected = list(previous)
                for slot, event in enumerate(events):
                    target = previous[slot]
                    if event["parent_ids"]["target"] != target:
                        raise ValueError(f"DE target mismatched the previous generation: {seed}/{gnum}/{slot}")
                    child = candidates.get(event["id"])
                    if child and child["macro_f1"] > candidates[target]["macro_f1"]:
                        expected[slot] = event["id"]
            else:
                pool = previous + [e["id"] for e in events if e["status"] == "evaluated"]
                expected = sorted(pool, key=lambda k: (-candidates[k]["macro_f1"], candidates[k]["creation_order"]))[:n]
            if expected != row["retained"]:
                raise ValueError(f"Survivors disagree with the replacement policy: {arm}/{seed}/G{gnum}")
    drops = {k: sum(a[k] > b[k] + EPS for a, b in zip(history, history[1:])) for k in ("best", "mean")}
    if arm != "tournament" and any(drops.values()):
        raise ValueError(f"Elitist retention failed: {arm}/{seed}")
    patterns = set()
    previous_best = None
    new_trials = []
    for gnum in range(g + 1):
        cohort = [p for p in candidates.values() if p["generation"] == gnum]
        scores, previously_seen = [], 0
        for candidate in cohort:
            receipt = read(suite/"cache"/"classification"/(candidate["classification_cache_key"] + ".json"))
            rows = receipt["value"]["metrics"]["predictions"]
            if [p["PID"] for p in rows] != protocol["manifest"]["search"]["ids"]:
                raise ValueError("Candidate prediction order does not match search set")
            pattern = tuple(p["prediction"] for p in rows)
            previously_seen += pattern in patterns
            patterns.add(pattern)
            scores.append(candidate["macro_f1"])
        if gnum:
            copies = [e for e in state["events"] if e["generation"] == gnum and e["status"] == "duplicate"]
            new_trials.append({"generation": gnum, "evaluated": len(scores),
                               "mean_trial": fmean(scores) if scores else None,
                               "best_trial": max(scores, default=None),
                               "above_previous_record": sum(v > previous_best + EPS for v in scores),
                               "repeated_prediction_vectors": previously_seen,
                               "duplicate_instructions": len(copies),
                               "copies_of_de_best": sum(e["duplicate_of"] == e["parent_ids"].get("best") for e in copies)
                               if arm == "de" else None})
        previous_best = history[gnum]["incumbent"]
    return {"seed": seed, "arm": arm, "history": history, "drops": drops,
            "distinct_prediction_vectors": len(patterns), "new_trials": new_trials,
            "record_generations": sum(b["incumbent"] > a["incumbent"] + EPS for a, b in zip(history, history[1:])),
            "unique_evaluated": len(candidates), "slots": dict(Counter(e["status"] for e in state["events"])),
            "cost": state["cost"]}


def build_report(suite, out):
    protocol, confirm = read(suite / "protocol.json"), read(suite / "confirmation.json")
    if confirm["protocol_hash"] != protocol["protocol_hash"] or confirm["status"] != "complete":
        raise ValueError("Search and confirmation must use the same completed protocol")
    arms, seeds = protocol["arms"], protocol["seeds"]
    by_arm = {arm: [checked_history(suite, protocol, arm, seed) for seed in seeds] for arm in arms}
    for seed in seeds:
        first_generation = []
        for arm in ("tournament", "elite"):
            events = read(suite / "arms" / f"{arm}_{seed}.json")["events"]
            first_generation.append([
                (e["slot"], e["generation_cache_key"], e["status"], e.get("macro_f1"))
                for e in events if e["generation"] == 1])
        if first_generation[0] != first_generation[1]:
            raise ValueError(f"The two GA survivor policies received different first-generation trials for seed {seed}")
    means, summaries = {}, {}
    for arm, runs in by_arm.items():
        means[arm] = {k: [fmean(points) for points in zip(*([r[k] for r in s["history"]] for s in runs))]
                      for k in ("best", "mean", "incumbent")}
        final = [r for r in confirm["results"] if r["arm"] == arm]
        if sorted(r["seed"] for r in final) != sorted(seeds):
            raise ValueError(f"Missing paired confirmation results for {arm}")
        diffs = {c["baseline"]: c for c in confirm["contrasts"]
                 if c["arm"] == arm and c["seed"] == "mean_over_observed_seeds"}
        summaries[arm] = {
            "final_retained_best": means[arm]["best"][-1], "final_retained_mean": means[arm]["mean"][-1],
            "best_ever_search": means[arm]["incumbent"][-1], "confirmation": fmean(r["confirmation_f1"] for r in final),
            "contrasts": diffs, "record_generations_total": sum(r["record_generations"] for r in runs),
            "unique_evaluated_per_seed": [r["unique_evaluated"] for r in runs],
            "distinct_prediction_vectors_per_seed": [r["distinct_prediction_vectors"] for r in runs],
            "generation_slots": dict(sum((Counter(r["slots"]) for r in runs), Counter())),
            "cost": {role: dict(sum((Counter(r["cost"][role]) for r in runs), Counter()))
                     for role in ("classification", "generation")},
        }
    initial = [r for r in confirm["results"] if r["arm"] == "G0"]
    failures = suite / "failures.jsonl"
    data = {"protocol_hash": protocol["protocol_hash"], "seeds": seeds, "population": protocol["population"],
            "generations": protocol["generations"], "class_counts": {s: protocol["manifest"][s]["class_counts"]
                for s in ("search", "confirmation")}, "scope": confirm["scope"],
            "initial_best_search": fmean(r["search_f1"] for r in initial),
            "initial_best_confirmation": fmean(r["confirmation_f1"] for r in initial),
            "methods": summaries, "per_seed_search": by_arm, "confirmation": confirm,
            "interrupted_request_ledger": [json.loads(line) for line in failures.read_text().splitlines()] if failures.exists() else []}
    out.mkdir(parents=True, exist_ok=True)
    (out/"live_summary.json").write_text(json.dumps(data, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for ax, field, title in zip(axes.flat, ("best", "mean", "incumbent"),
                                ("Best retained", "Mean retained", "Best discovered so far")):
        for arm in arms:
            ax.plot(range(protocol["generations"] + 1), means[arm][field], color=COLORS[arm],
                    marker="o", markersize=4, linewidth=1.7, label=NAMES[arm])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel="Generation", ylabel="Search macro-F1", title=title)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    generations = range(1, protocol["generations"] + 1)
    for arm, runs in by_arm.items():
        trials = list(zip(*(r["new_trials"] for r in runs)))
        means_by_generation, discoveries = [], []
        for items in trials:
            seed_means = [x["mean_trial"] for x in items if x["mean_trial"] is not None]
            means_by_generation.append(fmean(seed_means) if seed_means else float("nan"))
            discoveries.append(sum(x["above_previous_record"] for x in items))
        axes[1,0].plot(generations, means_by_generation, color=COLORS[arm], marker="o", markersize=4)
        axes[1,1].plot(generations, discoveries, color=COLORS[arm], marker="o", markersize=4)
    axes[1,0].set(xlabel="Generation", ylabel="Search macro-F1", title="Mean new trial before selection")
    axes[1,1].set(xlabel="Generation", ylabel="Trials across all seeds", title="New trials beating the previous record")
    for ax in axes[1,:2]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
    axes[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax = axes[1,2]
    ax.axvline(0, color="#999999", ls="--", lw=1)
    for i, arm in enumerate(arms):
        c = summaries[arm]["contrasts"]["G0"]
        lo, hi = c["paired_95pct_interval"]
        ax.plot([lo, hi], [i, i], color=COLORS[arm], linewidth=2.5)
        ax.plot(c["delta"], i, "o", color=COLORS[arm])
    ax.set_yticks(range(len(arms)), [NAMES[a] for a in arms])
    ax.set(xlabel="Paired macro-F1 change from initial winner", title="Disjoint development confirmation")
    fig.suptitle(f"DepSign {protocol['mode'].replace('_','-')} · averages across {len(seeds)} search seeds")
    fig.savefig(out/"live_fitness.png", dpi=160)
    plt.close(fig)
    table = [
        "# DepSign instruction experiments",
        "",
        f"Search used {protocol['search_size']} posts. Confirmation used the same "
        f"{protocol['confirmation_size']} separate development posts for every frozen winner. "
        f"The table averages over seeds {', '.join(map(str, seeds))}. "
        f"The initial winner scored {data['initial_best_search']:.4f} on search "
        f"and {data['initial_best_confirmation']:.4f} on confirmation.",
        "",
        "| Method | Final retained mean | Best found on search | Confirmation | "
        "Change from initial on confirmation [paired 95% interval] |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        s = summaries[arm]
        delta = s["contrasts"]["G0"]
        lo, hi = delta["paired_95pct_interval"]
        table.append(f"| {NAMES[arm]} | {s['final_retained_mean']:.4f} | "
                     f"{s['best_ever_search']:.4f} | {s['confirmation']:.4f} | "
                     f"{delta['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    table += ["", "Each confirmation candidate was picked by search macro-F1 over all generations. "
              "The intervals resample posts within gold class and condition on the observed seeds. "
              "Shared prompts and cached requests reduce actual inference work. "
              "The suite used the same number of proposal slots per arm.",
              "", "[Per-generation curves](live_fitness.png) · [Per-seed results, costs and class F1](live_summary.json)"]
    (out/"instruction_results.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def plot_components(path, output):
    data = read(path)
    if data["status"] != "complete":
        raise ValueError("Cannot plot an incomplete component comparison")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, metric, title in zip(axes, ("best_retained", "mean_retained"), ("Best retained", "Mean retained")):
        for arm, label in (("tournament", "Historical tournament"), ("elite", "Forward top N")):
            values = data["search"][arm]
            ax.plot([r["generation"] for r in values], [r[metric] for r in values],
                    color=COLORS[arm], marker="o", markersize=4, label=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(title=title, xlabel="Generation", ylabel="Search macro-F1")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Full component search · {data['mode'].replace('_','-')} · seed {data['seed']}")
    fig.savefig(output / "component_fitness.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"results")
    parser.add_argument("--component-comparison", type=Path, help="Also plot the completed full-component comparison")
    args = parser.parse_args()
    build_report(args.suite, args.output)
    if args.component_comparison:
        plot_components(args.component_comparison, args.output)
