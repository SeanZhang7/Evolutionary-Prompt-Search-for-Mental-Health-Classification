"""Replay selection on saved DepSign fitness scores; no model or posts are needed."""

import argparse
from collections import defaultdict
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


def drops(scores):
    return sum(a > b + EPS for a, b in zip(scores, scores[1:]))


def describe(report):
    history = report["generation_metrics"]
    best = [g["best_macro_f1"] for g in history]
    means = [g["mean_macro_f1"] for g in history]
    return {
        "mode": report["label_mode"], "seed": report["seed"],
        "survivor_selection": report.get("survivor_selection") or report.get("rules", {}).get("survivor_selection"),
        "generations": len(history) - 1, "evaluations": report["candidate_evaluations"],
        "initial_best": best[0], "final_best": best[-1], "global_best": report["best"]["macro_f1"],
        "initial_mean": means[0], "final_mean": means[-1],
        "observed_best_drops": drops(best), "observed_mean_drops": drops(means),
    }


def top_n(pool, n):
    return sorted(pool, key=lambda c: (-c["macro_f1"], c["creation_order"]))[:n]


def scores(candidates):
    values = [c["macro_f1"] for c in candidates]
    return max(values), fmean(values)


def replay(report):
    """Compare the observed choice with top N using the exact same scored menu each step."""
    n = report["population_size"]
    complete = [c for c in report["population"] if c["status"] == "complete"]
    by_id = {c["id"]: c for c in complete}
    if len(by_id) != len(complete):
        raise ValueError("Duplicate candidate IDs in report")
    by_birth = defaultdict(list)
    for c in complete:
        by_birth[c["generation"]].append(c)
    previous, archive, rows = [], [], []
    for generation, metrics in enumerate(report["generation_metrics"]):
        if metrics["generation"] != generation:
            raise ValueError("Non-consecutive generations in report")
        new = by_birth[generation]
        menu = previous + new
        observed = [by_id[key] for key in metrics["retained_ids"]]
        if len(observed) != n or len(set(metrics["retained_ids"])) != n:
            raise ValueError("Invalid survivor IDs")
        if not set(metrics["retained_ids"]).issubset(c["id"] for c in menu):
            raise ValueError("Invalid candidate pool")
        observed_best, observed_mean = scores(observed)
        if not (isclose(observed_mean, metrics["mean_macro_f1"], abs_tol=EPS, rel_tol=0)
                and isclose(observed_best, metrics["best_macro_f1"], abs_tol=EPS, rel_tol=0)):
            raise ValueError("Reconstructed survivors disagree with recorded fitness")
        local_best, local_mean = scores(top_n(menu, n))
        prev_record = max((c["macro_f1"] for c in archive), default=None)
        archive = top_n(archive + new, n)
        archive_best, archive_mean = scores(archive)
        rows.append({
            "generation": generation, "new_complete_candidates": len(new),
            "observed_best": observed_best, "observed_mean": observed_mean,
            "same_pool_top_n_best": local_best, "same_pool_top_n_mean": local_mean,
            "fixed_stream_top_n_best": archive_best, "fixed_stream_top_n_mean": archive_mean,
            "new_record": prev_record is not None and archive_best > prev_record + EPS,
            "best_new_candidate": max((c["macro_f1"] for c in new), default=None),
        })
        previous = observed
    if len(complete) != sum(row["new_complete_candidates"] for row in rows):
        raise ValueError("Run contains complete candidates outside reported generations")
    if not isclose(rows[-1]["fixed_stream_top_n_best"], report["best"]["macro_f1"], abs_tol=EPS, rel_tol=0):
        raise ValueError("Replayed historical best disagrees with reported global best")
    return rows


def plot(series, reports, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    runs = sorted(series.items())
    fig, axes = plt.subplots(len(runs), 2, figsize=(9, 2.3 * len(runs) + 1), squeeze=False)
    for i, (name, rows) in enumerate(runs):
        mode = reports[name]["mode"].replace("_", "-")
        short_name = Path(name).name.removeprefix(reports[name]["mode"] + "_")
        if short_name[:8].isdigit() and short_name[8:9] == "T":
            short_name = short_name[:8]
        x = [r["generation"] for r in rows]
        for j, metric in enumerate(("best", "mean")):
            ax = axes[i][j]
            for field, label, color, style in (
                ("observed", "Observed tournament", "#15588b", "-"),
                ("same_pool_top_n", "Top N from actual parents + children", "#b44120", "--"),
                ("fixed_stream_top_n", "Top N of all candidates seen", "#287347", ":"),
            ):
                ax.plot(x, [r[f"{field}_{metric}"] for r in rows], label=label,
                        color=color, linestyle=style, linewidth=1.9)
            ax.set_title(f"{mode} · {short_name} · {metric} macro-F1", fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            ax.grid(alpha=.22)
            if i == len(runs) - 1:
                ax.set_xlabel("Generation")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .972))
    fig.suptitle("Selection on recorded DepSign development-set scores", y=.997)
    fig.tight_layout(rect=(0, 0, 1, .927))
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/depsign")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    reports, replays, excluded = {}, {}, {}
    for folder in ("ga_runs", "component_ga_runs"):
        for path in sorted((args.data_root / folder).glob("*/performance.json")):
            raw = path.read_bytes()
            data = json.loads(raw)
            name = str(path.parent.relative_to(args.data_root))
            if data["status"] != "complete" or len(data["generation_metrics"]) < 2:
                excluded[name] = data["status"]
                continue
            reports[name] = {**describe(data), "report_sha256": hashlib.sha256(raw).hexdigest()}
            if "population" in data and "retained_ids" in data["generation_metrics"][0]:
                rows = replay(data)
                replays[name] = rows
                reports[name].update(
                    offspring_slots=data["generation_slots"],
                    valid_unique_offspring=sum(row["new_complete_candidates"] for row in rows[1:]),
                    duplicate_offspring_skipped=data["duplicate_offspring_skipped"],
                    invalid_offspring_skipped=data["invalid_offspring_skipped"],
                    new_record_generations=sum(row["new_record"] for row in rows),
                    missed_best_in_same_pool=sum(row["same_pool_top_n_best"] > row["observed_best"] + EPS for row in rows[1:]),
                    same_pool_top_n_mean_greater=sum(row["same_pool_top_n_mean"] > row["observed_mean"] + EPS for row in rows[1:]),
                    final_fixed_stream_top_n_mean=rows[-1]["fixed_stream_top_n_mean"],
                )
    if not replays:
        raise SystemExit("No complete runs with individual scores and retained IDs found")
    result = {
        "scope": "Complete archived DepSign runs with at least one evolved generation. Fitness is macro-F1 on the search data.",
        "same_pool_top_n": "Top N on the actual preceding survivors plus the candidates actually evaluated in this generation; independent one-step comparisons.",
        "fixed_stream_top_n": "Top N over all actual candidates observed so far. This reuses children of the actual parents even when their parents would have been removed; future generated candidates under live elitist search are unknown.",
        "selection_order": "Descending macro-F1, then ascending creation order.",
        "epsilon_for_counting_changes": EPS,
        "excluded": excluded,
        "replay_totals": {key: sum(reports[name][key] for name in replays) for key in (
            "generations", "evaluations", "new_record_generations", "observed_best_drops",
            "observed_mean_drops", "missed_best_in_same_pool", "same_pool_top_n_mean_greater",
            "offspring_slots", "valid_unique_offspring", "duplicate_offspring_skipped", "invalid_offspring_skipped",
        )},
        "runs": reports, "per_generation_replay": replays,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plot(replays, reports, args.output / "replay.png")
    (args.output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for run in replays:
        r = reports[run]
        print(f"{run}: G0 best={r['initial_best']:.4f}, best found={r['global_best']:.4f}, "
              f"last retained best={r['final_best']:.4f}, best drops={r['observed_best_drops']}, "
              f"mean drops={r['observed_mean_drops']}, record generations={r['new_record_generations']}")


if __name__ == "__main__":
    main()
