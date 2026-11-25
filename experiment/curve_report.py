"""Plot observed survivor best and mean fitness per search seed from a completed suite."""

import argparse
from collections import Counter
import json
from math import ceil, floor, isclose, isfinite
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EPS = 1e-12
PLATEAU_RANGE = 0.005
WINDOWS = [(0, 4), (4, 8), (8, 12)]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def strict_changes(points, metric):
    deltas = [b[metric] - a[metric] for a, b in zip(points, points[1:])]
    gains = sum(x > EPS for x in deltas)
    drops = sum(x < -EPS for x in deltas)
    return {"strict_gains": gains, "regressions": drops, "ties": len(deltas) - gains - drops}


def window_changes(points, start, end):
    if end >= len(points):
        return None
    result = {}
    for metric in ("best", "mean"):
        gain = points[end][metric] - points[start][metric]
        result[metric] = {"gain": gain, "gain_per_generation": gain / (end - start)}
    return result


def late_plateau(points):
    if len(points) < 13:
        return {"status": "insufficient_horizon", "observed": None}
    window = points[9:13]
    ranges = {metric: max(row[metric] for row in window) - min(row[metric] for row in window)
              for metric in ("best", "mean")}
    enough_children = all(row["unique_evaluated_children"] > 0 for row in window[1:])
    result = enough_children and all(span <= PLATEAU_RANGE + EPS for span in ranges.values())
    return {"status": "assessed", "observed": result, "ranges_G9_G12": ranges,
            "at_least_one_unique_child_each_G10_G11_G12": enough_children}


def summarize_arm(path, protocol, arm, seed):
    data = read(path)
    generations = protocol["generations"]
    history = data["generation_metrics"]
    if (data["protocol_hash"] != protocol["protocol_hash"] or data["status"] != "complete"
            or data["arm"] != arm or data["seed"] != seed
            or [row["generation"] for row in history] != list(range(generations + 1))):
        raise ValueError(f"Incomplete or mismatched history: {path}")
    candidates = {item["id"]: item for item in data["evaluated"]}
    unique = {item["instruction"] for item in candidates.values()}
    if (len(candidates) != len(data["evaluated"]) or len(unique) != len(candidates)
            or any(not isfinite(p["macro_f1"]) or not 0 <= p["macro_f1"] <= 1 for p in candidates.values())):
        raise ValueError(f"Invalid scored candidate archive: {path}")
    births = Counter(item["generation"] for item in candidates.values())
    event_births = Counter(event["generation"] for event in data["events"] if event["status"] == "evaluated")
    if (births[0] != protocol["population"] or set(births) - set(range(generations + 1))
            or any(births[g] != event_births[g] for g in range(1, generations + 1))):
        raise ValueError(f"Offspring count disagrees with evaluated candidates: {path}")
    prior = set()
    points = []
    for g, entry in enumerate(history):
        retained = set(entry["retained"])
        eligible = prior | {ident for ident, item in candidates.items() if item["generation"] == g}
        if (len(retained) != len(entry["retained"]) or len(retained) != protocol["population"]
                or not retained <= eligible):
            raise ValueError(f"Invalid survivor roster at G{g}: {path}")
        scores = [candidates[ident]["macro_f1"] for ident in entry["retained"]]
        row = {"generation": g, "best": max(scores), "mean": fmean(scores),
               "unique_evaluated_children": births[g] if g else 0}
        if any(not isclose(row[k], entry[k], abs_tol=EPS, rel_tol=0) for k in ("best", "mean")):
            raise ValueError(f"Reported scores disagree with survivor roster at G{g}: {path}")
        points.append(row)
        prior = retained
    return {"seed": seed, "points": points,
            "changes": {metric: strict_changes(points, metric) for metric in ("best", "mean")},
            "windows": {f"G{start}-G{end}": window_changes(points, start, end) for start, end in WINDOWS},
            "late_plateau": late_plateau(points)}


def plot(result, path):
    seeds = result["per_seed"]
    generations = result["generations"]
    fig, axes = plt.subplots(1, len(seeds), sharex=True, sharey=True,
                             figsize=(max(4.8, 3.25 * len(seeds)), 4.05), squeeze=False)
    values = [p[k] for run in seeds for p in run["points"] for k in ("best", "mean")]
    pad = max(0.005, 0.08 * (max(values) - min(values)))
    limits = (max(0.0, floor((min(values) - pad) * 100) / 100),
              min(1.0, ceil((max(values) + pad) * 100) / 100))
    for ax, run in zip(axes.flat, seeds):
        points = run["points"]
        x = [p["generation"] for p in points]
        for metric, label, color, marker, linestyle in (
            ("best", "Best retained", "#086a96", "o", "-"),
            ("mean", "Mean retained", "#ae4b12", "s", "--")
        ):
            ax.plot(x, [p[metric] for p in points], color=color, marker=marker,
                    markersize=4.7, linewidth=1.8, linestyle=linestyle, label=label)
        if generations >= 12:
            ax.axvspan(9, 12, color="#627f67", alpha=.10)
        ax.set(ylim=limits, xticks=x if generations <= 12 else x[::2],
               title=f"Search seed {run['seed']}", xlabel="Generation")
        ax.grid(axis="y", alpha=.22)
    axes[0,0].set_ylabel("Retained population macro-F1")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.suptitle(f"{result['method_name']} · DepSign {result['label_mode'].replace('_', '-')} · "
                 f"{result['search_size']} development posts", fontsize=13, y=.97)
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, .92), frameon=False)
    if generations < 12:
        footer = f"G0–G{generations} observed. Assessing a late plateau requires G9–G12."
    else:
        footer = "Shading marks G9–G12, the four observed points used to assess the last three transitions."
    fig.text(.5, .021, footer, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .805))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build(suite, arm, output):
    protocol = read(suite / "protocol.json")
    if arm not in protocol["arms"] or not protocol["seeds"]:
        raise ValueError("A predeclared method and at least one seed are required")
    name = {"feedback": "Error-feedback prompt search", "de": "Textual differential evolution",
            "elite": "GA with elite retention", "tournament": "GA with tournament retention",
            "random": "Independent proposals"}[arm]
    rows = [summarize_arm(suite / "arms" / f"{arm}_{seed}.json", protocol, arm, seed) for seed in protocol["seeds"]]
    result = {
        "arm": arm, "method_name": name, "protocol_hash": protocol["protocol_hash"],
        "label_mode": protocol["mode"], "search_size": protocol["search_size"],
        "population": protocol["population"], "generations": protocol["generations"],
        "measured": "Macro-F1 best and arithmetic mean from IDs actually retained in each generation.",
        "slope_definition": "Endpoint change divided by elapsed generations. A null window lacks its final generation.",
        "strict_comparison_epsilon": EPS,
        "plateau_definition": "G9 through G12 span at most 0.005 macro-F1 for both retained best and retained mean; "
                              "at least one unique evaluated child in each of G10, G11, G12. Descriptive search-sample criterion.",
        "per_seed": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{arm}_curve_g{result['generations']}"
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot(result, output / f"{stem}.png")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--arm", default="feedback")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    result = build(args.suite, args.arm, args.output)
    for seed in result["per_seed"]:
        print(f"Seed {seed['seed']}: {seed['changes']}; late plateau {seed['late_plateau']}")
