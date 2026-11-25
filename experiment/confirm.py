"""Evaluate preselected search winners on paired, disjoint development examples."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .evaluation import ClassificationError, Evaluation, ev
else:
    from evaluation import ClassificationError, Evaluation, ev


BOOTSTRAP_SEED = 914
BOOTSTRAP_REPLICATES = 2000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pick(candidates):
    for item in candidates:
        if not math.isfinite(item["macro_f1"]) or not 0 <= item["macro_f1"] <= 1:
            raise ValueError("Invalid development macro-F1 in completed arm")
    return min(candidates, key=lambda p: (-p["macro_f1"], p["creation_order"]))


def frozen_winners(output, protocol):
    if not protocol["seeds"] or not protocol["arms"] or len(set(protocol["arms"])) != len(protocol["arms"]):
        raise ValueError("Missing or invalid predeclared comparison arms")
    winners = []
    for seed in protocol["seeds"]:
        seed_initial = None
        for arm in protocol["arms"]:
            path = output / "arms" / f"{arm}_{seed}.json"
            data = read(path)
            if (data["status"] != "complete" or data["seed"] != seed or data["arm"] != arm
                    or data["protocol_hash"] != protocol["protocol_hash"]
                    or [g["generation"] for g in data["generation_metrics"]] != list(range(protocol["generations"] + 1))):
                raise ValueError(f"Arm does not fully match the planned run: {path}")
            candidates = data["evaluated"]
            n = protocol["population"]
            if len(candidates) < n or len({p["creation_order"] for p in candidates}) != len(candidates):
                raise ValueError(f"Invalid candidate archive in {path}")
            initial = candidates[:n]
            initial_signature = [(p["instruction"], p["macro_f1"]) for p in initial]
            init_best = pick(initial)
            best = pick(candidates)
            if init_best["id"] != data["initial_best"]["id"] or best["id"] != data["best"]["id"]:
                raise ValueError(f"Candidate archive disagrees with reported winner in {path}")
            if seed_initial is None:
                seed_initial = initial_signature
                winners.append({"seed": seed, "arm": "G0", "candidate": init_best["id"],
                                "instruction": init_best["instruction"], "search_f1": init_best["macro_f1"]})
            elif seed_initial != initial_signature:
                raise ValueError(f"Different initial conditions across arms for seed {seed}")
            winners.append({"seed": seed, "arm": arm, "candidate": best["id"],
                            "instruction": best["instruction"], "search_f1": best["macro_f1"]})
    return {"protocol_hash": protocol["protocol_hash"], "selection": "Highest search macro-F1 over all fully evaluated candidates; earliest creation order breaks ties.",
            "winner_records": winners}


def bootstrap_scores(gold, predictions, draws, labels):
    sampled_gold = np.asarray(gold)[draws]
    sampled_pred = np.asarray(predictions)[draws]
    scores = np.zeros(len(draws))
    for label in labels:
        actual, predicted = sampled_gold == label, sampled_pred == label
        denom = actual.sum(axis=1) + predicted.sum(axis=1)
        scores += np.divide(2.0 * (actual & predicted).sum(axis=1), denom,
                            out=np.zeros(len(draws)), where=denom > 0)
    return scores / len(labels)


def summarize(protocol, winners, cache):
    labels = ev.task_spec(protocol["mode"])["labels"]
    ids = protocol["manifest"]["confirmation"]["ids"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary, scores, bootstrap = [], {}, {}
    # Codes keep the bootstrap array small. INVALID gets its own prediction code.
    code = {label: i for i, label in enumerate(labels + ["INVALID"])}
    gold = None
    for key, metrics in cache["evaluations"].items():
        predictions = metrics["predictions"]
        if [p["PID"] for p in predictions] != ids or metrics["request_error_count"]:
            raise ValueError("Incomplete or mismatched confirmation predictions")
        observed = np.array([code[p["gold"]] for p in predictions])
        if gold is None:
            gold = observed
            groups = [np.flatnonzero(gold == k) for k in range(len(labels))]
            draws = np.concatenate([rng.choice(g, size=(BOOTSTRAP_REPLICATES, len(g))) for g in groups], axis=1)
        elif not np.array_equal(observed, gold):
            raise ValueError("Gold labels changed across confirmation candidates")
        pred = [code[p["prediction"]] for p in predictions]
        bootstrap[key] = bootstrap_scores(gold, pred, draws, list(range(len(labels))))
    for item in winners["winner_records"]:
        key = text_key(item["instruction"])
        metrics = cache["evaluations"][key]
        scores[item["seed"], item["arm"]] = (key, metrics["macro_f1"])
        summary.append({"seed": item["seed"], "arm": item["arm"], "candidate": item["candidate"],
                        "instruction_sha256": key, "search_f1": item["search_f1"],
                        "confirmation_f1": metrics["macro_f1"], "per_class": metrics["per_class"],
                        "invalid_count": metrics["invalid_output_count"], "n": metrics["n"]})
    contrasts = []
    for arm in protocol["arms"]:
        for baseline in ["G0"] + (["tournament"] if "tournament" in protocol["arms"] else []):
            if arm == baseline:
                continue
            differences, paired_draws = [], []
            for seed in protocol["seeds"]:
                key, score = scores[seed, arm]
                ref, reference_score = scores[seed, baseline]
                difference = score - reference_score
                rep = bootstrap[key] - bootstrap[ref]
                differences.append(difference)
                paired_draws.append(rep)
                contrasts.append({"arm": arm, "baseline": baseline, "seed": seed, "delta": difference,
                                  "paired_95pct_interval": np.quantile(rep, [0.025, 0.975]).tolist()})
            contrasts.append({"arm": arm, "baseline": baseline, "seed": "mean_over_observed_seeds",
                              "delta": float(np.mean(differences)),
                              "paired_95pct_interval": np.quantile(np.mean(paired_draws, axis=0), [0.025, 0.975]).tolist()})
    usage = Counter()
    for m in cache["evaluations"].values():
        usage.update(m["usage"])
    for failed in cache["failures"]:
        usage.update(failed["usage"])
    return {"status": "complete", "protocol_hash": protocol["protocol_hash"],
            "reference": "See confirmation_winners.json for frozen instructions and confirmation_cache.json for predictions.",
            "scope": protocol["manifest"]["scope"], "results": summary, "contrasts": contrasts,
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "interval": "paired percentile, 95%",
                          "unit": "Post resampled with replacement within gold class, identical draws for all prompts and seeds.",
                          "interpretation": "Conditional on observed search seeds and this development sample's class counts; descriptive intervals, not multiplicity-adjusted."},
            "cost": {"unique_prompts": len(cache["evaluations"]), "winner_slots": len(summary),
                     "failed_attempts": len(cache["failures"]), "reported_tokens": dict(usage),
                     "classification_requests": sum(m["classification_requests"] for m in cache["evaluations"].values())
                                                + sum(f["classification_requests"] for f in cache["failures"])}}


def confirm_suite(output, evaluation, protocol):
    output = Path(output)
    if read(output / "protocol.json") != protocol or evaluation.manifest != protocol["manifest"]:
        raise ValueError("Confirmation data, evaluation configuration or on-disk protocol differ from the search")
    with (output / ".confirmation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Freeze the complete search-selected roster before exposing any confirmation metrics.
        winners = frozen_winners(output, protocol)
        roster = output / "confirmation_winners.json"
        if roster.exists() and read(roster) != winners:
            raise ValueError("The frozen confirmation roster has changed; use a new suite")
        if not roster.exists():
            save(roster, winners)
        cache_path = output / "confirmation_cache.json"
        cache = read(cache_path) if cache_path.exists() else {"protocol_hash": protocol["protocol_hash"],
                                                               "evaluations": {}, "failures": []}
        if cache["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Confirmation cache belongs to a different protocol")
        pending = {text_key(w["instruction"]): w["instruction"] for w in winners["winner_records"]}
        if set(cache["evaluations"]) - pending.keys():
            raise ValueError("Confirmation cache contains an unselected prompt")
        if pending.keys() - cache["evaluations"].keys() and evaluation.preflight() != protocol["server"]:
            raise ValueError("Server identity/version differs from the frozen search protocol")
        for key, text in pending.items():
            if key in cache["evaluations"]:
                continue
            try:
                metrics = evaluation.evaluate(text, split="confirmation")
            except ClassificationError as exc:
                cache["failures"].append({"sha256": key, "error": str(exc),
                                          "classification_requests": exc.classification_requests, "usage": exc.usage})
                save(cache_path, cache)
                raise
            cache["evaluations"][key] = metrics
            save(cache_path, cache)
            print(f"Confirmation {len(cache['evaluations'])}/{len(pending)} unique prompts", flush=True)
        report = summarize(protocol, winners, cache)
        save(output / "confirmation.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args()
    protocol = read(args.suite / "protocol.json")
    evl = Evaluation(protocol["mode"], protocol["search_size"], protocol["confirmation_size"],
                     protocol["split_seed"], protocol["workers"])
    report = confirm_suite(args.suite, evl, protocol)
    print(f"Confirmed {len(report['results'])} winner slots. Results: {args.suite / 'confirmation.json'}")


if __name__ == "__main__":
    main()
