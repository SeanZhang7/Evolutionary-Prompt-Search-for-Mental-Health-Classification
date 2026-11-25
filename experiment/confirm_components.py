"""Paired development confirmation of full component prompts, through a common search generation."""

import argparse
from collections import Counter
from copy import copy, deepcopy
import fcntl
from pathlib import Path

import numpy as np

if __package__:
    from . import confirm
    from .evaluation import ClassificationError, Evaluation, ROOT, ev
else:
    import confirm
    from evaluation import ClassificationError, Evaluation, ROOT, ev
from scripts import component_prompts as cp


HISTORICAL = ROOT / "data/processed/depsign/component_ga_runs/three_class_20261125T012800064686Z/performance.json"
POLICIES = {"tournament": ("mu_plus_lambda_tournament", "first_drawn"),
            "elite": ("mu_plus_lambda_top_mu", "creation_order_parents_then_children")}


def without_policy(report, field):
    base = deepcopy(report[field])
    policy = base["search"] if field == "config" else base
    for key in ("survivor_selection", "tie_break"):
        del policy[key]
    return base


def checked_reports(paths, generation, evaluation):
    reports = {k: confirm.read(path) for k, path in paths.items()}
    for arm, r in reports.items():
        if (r["task"] != "depsign" or r["search_space"] != "components" or r["status"] != "complete"
                or r["label_mode"] != evaluation.label_mode or r["completed_generations"] < generation):
            raise ValueError(f"{arm} is not a completed comparable DepSign component run")
        for settings in [r["rules"], r["config"]["search"]]:
            if (settings["survivor_selection"], settings["tie_break"]) != POLICIES[arm]:
                raise ValueError(f"Unexpected {arm} survivor policy")
        if r["sample_ids"] != evaluation.manifest["search"]["ids"] or r["classification_parameters"] != ev.PARAMETERS:
            raise ValueError(f"{arm} was scored using a different fitness sample or decoding configuration")
        if r["context_tokens"] != ev.CONTEXT_TOKENS or r["model"] != ev.MODEL:
            raise ValueError(f"{arm} inference configuration differs from the confirmation evaluator")
        cp.check_overlap(r["fixed_context"], evaluation.splits["confirmation"])
    a, b = reports["tournament"], reports["elite"]
    for field in ("config", "rules"):
        if without_policy(a, field) != without_policy(b, field):
            raise ValueError(f"Configurations also differ outside survivor settings ({field})")
    same = ("seed", "population_size", "labels", "label_mapping", "sample_ids", "generation_parameters",
            "classification_parameters", "context_tokens", "model", "tokenizer_files_sha256", "fixed_context", "operators")
    for field in same:
        if a[field] != b[field]:
            raise ValueError(f"Compared runs differ in {field}")
    hashes = [{k: v["sha256"] for k, v in r["sources"].items() if k != "config"} for r in (a, b)]
    if hashes[0] != hashes[1] or hashes[0]["evaluation_data"] != evaluation.manifest["dev_100_sha256"]:
        raise ValueError("Non-config input hashes or dev_100 fingerprint differ")
    return reports, hashes[0]


def generation_diagnostics(reports, through):
    comparison = []
    for g in range(through + 1):
        arms = [reports[k] for k in ("tournament", "elite")]
        groups = [{p["id"]: p for p in r["population"] if p["generation"] == g} for r in arms]
        a, b = groups
        common = a.keys() & b.keys()
        equal_prompts = {k for k in common if a[k]["combination_sha256"] == b[k]["combination_sha256"]}
        scored = {k for k in equal_prompts if a[k]["status"] == b[k]["status"] == "complete"}
        identical_scores = {k for k in scored if a[k]["performance"] == b[k]["performance"]}
        same_candidates = (a.keys() == b.keys() and equal_prompts == common
                           and all(a[k]["status"] == b[k]["status"] for k in common))
        history = [next(row for row in r["generation_metrics"] if row["generation"] == g) for r in arms]
        selection = [next((s for s in r["selections"] if s["generation"] == g), None) for r in arms]
        comparison.append({
            "generation": g, "tournament_proposed_valid": len(a), "elite_proposed_valid": len(b),
            "same_slot_and_complete_prompt": len(equal_prompts), "identical_scored_prompts": len(scored),
            "identical_aggregate_metrics_on_same_scored_prompts": len(identical_scores),
            "all_generated_prompts_and_statuses_match": same_candidates,
            "all_scores_on_matching_prompts_agree": identical_scores == scored,
            "parent_pairs_and_seeded_draws_match": selection[0] == selection[1],
            "retained_order_matches": history[0]["retained_ids"] == history[1]["retained_ids"],
            "retained_members_match": set(history[0]["retained_ids"]) == set(history[1]["retained_ids"]),
        })
    return comparison


def freeze(protocol, reports, paths, through, evaluation):
    identities, renderings = [], {}
    for arm, r in reports.items():
        fixed = r["fixed_context"]
        for name, cutoff in [(arm + "_G0", 0), (arm, through)]:
            best = confirm.pick([p for p in r["population"] if p["status"] == "complete" and p["generation"] <= cutoff])
            components = best["components"]
            if best["combination_sha256"] != cp.fingerprint({"components": components, "fixed": fixed}):
                raise ValueError(f"Archived combination fingerprint mismatch: {arm}, {best['id']}")
            instruction = components["instruction"]
            template = cp.render_template(components, fixed)
            full = template.replace("{{INSTRUCTION}}", instruction)
            key = confirm.text_key(full)
            identities.append({"selection": name, "search_f1": best["macro_f1"], "candidate": best["id"],
                               "generation": best["generation"], "creation_order": best["creation_order"],
                               "template_sha256": key, "components": components, "full_template": full})
            renderings[key] = (template, instruction)
    frozen = {"primary_protocol": protocol["protocol_hash"], "through_generation": through,
              "sources": {k: {"path": str(p.resolve()), "sha256": ev.digest(p.read_bytes())} for k, p in paths.items()},
              "confirmation_ids": evaluation.manifest["confirmation"]["ids"], "choices": identities}
    return frozen, renderings


def comparison_report(frozen, reports, hashes, diagnostic, cache, evaluation):
    metrics = cache["evaluations"]
    rows, scores, usage = [], {}, Counter()
    labels = ev.task_spec(evaluation.label_mode)["labels"]
    codes = {label: i for i, label in enumerate(labels + ["INVALID"])}
    samples = evaluation.splits["confirmation"]
    mapping = ev.task_spec(evaluation.label_mode)["label_mapping"]
    ids = [r["PID"] for r in samples]
    gold = np.array([codes[mapping[r["Label"]]] for r in samples])
    rng = np.random.default_rng(confirm.BOOTSTRAP_SEED)
    groups = [np.flatnonzero(gold == k) for k in range(len(labels))]
    draws = np.concatenate([rng.choice(g, size=(confirm.BOOTSTRAP_REPLICATES, len(g))) for g in groups], axis=1)
    distributions = {}
    for digest, m in metrics.items():
        p = m["predictions"]
        if [v["PID"] for v in p] != ids or [codes[v["gold"]] for v in p] != gold.tolist() or m["request_error_count"]:
            raise ValueError("Incomplete or mismatched paired predictions")
        distributions[digest] = confirm.bootstrap_scores(gold, [codes[v["prediction"]] for v in p], draws, range(len(labels)))
        usage.update(m["usage"])
    for w in frozen["choices"]:
        key = w["template_sha256"]
        m = metrics[key]
        scores[w["selection"]] = (m["macro_f1"], distributions[key])
        rows.append({"selection": w["selection"], "winner_generation": w["generation"], "search_f1": w["search_f1"],
                     "confirmation_f1": m["macro_f1"], "n": m["n"], "per_class": m["per_class"],
                     "invalid_count": m["invalid_output_count"], "template_sha256": key})
    contrasts = []
    for arm, baseline in (("elite", "elite_G0"), ("tournament", "tournament_G0"), ("elite", "tournament")):
        x, u = scores[arm]
        y, v = scores[baseline]
        contrasts.append({"arm": arm, "baseline": baseline, "delta": x-y,
                          "paired_95pct_interval": np.quantile(u-v, [0.025, 0.975]).tolist()})
    through = frozen["through_generation"]
    curves, costs = {}, {}
    for arm, r in reports.items():
        complete = [p for p in r["population"] if p["status"] == "complete" and p["generation"] <= through]
        curves[arm] = [{"generation": g["generation"], "best_retained": g["best_macro_f1"], "mean_retained": g["mean_macro_f1"],
                        "best_ever": max(p["macro_f1"] for p in complete if p["generation"] <= g["generation"])}
                       for g in r["generation_metrics"] if g["generation"] <= through]
        attempts = [a for a in r["generation_attempts"] if a["generation"] <= through]
        costs[arm] = {"generations": through, "evaluated_combinations": len(complete),
                      "classification_requests": sum(p["performance"]["n"] for p in complete),
                      "generation_requests": len(attempts),
                      "reported_generation_tokens": sum((a["usage"] or {}).get("total_tokens", 0) for a in attempts)}
    for failure in cache["failures"]:
        usage.update(failure["usage"])
    differing = next((d for d in diagnostic if not all(d[k] for k in
                     ("all_generated_prompts_and_statuses_match", "all_scores_on_matching_prompts_agree",
                      "retained_order_matches", "parent_pairs_and_seeded_draws_match"))), None)
    return {"status": "complete", "seed": reports["elite"]["seed"], "mode": evaluation.label_mode,
            "primary_protocol_hash": frozen["primary_protocol"], "through_generation": through, "search": curves,
            "source_non_config_hashes": hashes, "source_report_sha256": {k: v["sha256"] for k, v in frozen["sources"].items()},
            "search_cost": costs, "generation_agreement": diagnostic, "first_divergent_generation": differing["generation"] if differing else None,
            "service_signatures_match": reports["elite"]["service"] == reports["tournament"]["service"],
            "scope": "Historical single-seed comparison through a common generation. Archives contain aggregate scores and confusion matrices, not individually paired search predictions.",
            "confirmation_scope": evaluation.manifest["scope"], "confirmation": rows, "contrasts": contrasts,
            "bootstrap": {"replicates": confirm.BOOTSTRAP_REPLICATES, "seed": confirm.BOOTSTRAP_SEED,
                          "interval": "Paired percentile bootstrap at post level, conditional on gold class counts and this search seed."},
            "confirmation_cost": {"unique_prompts": len(metrics), "failed_attempts": len(cache["failures"]),
                                  "reported_tokens": dict(usage),
                                  "classification_requests": sum(m["classification_requests"] for m in metrics.values())
                                  + sum(f["classification_requests"] for f in cache["failures"])}}


def compare(paths, evaluation, protocol, through, output, summary_path=None):
    if through < 1 or protocol["manifest"] != evaluation.manifest:
        raise ValueError("Need a positive generation and identical primary evaluation manifest")
    if (ROOT / "experiment/results") in (output.resolve(), *output.resolve().parents):
        raise ValueError("Use a private run directory for full templates, PIDs and predictions")
    reports, hashes = checked_reports(paths, through, evaluation)
    diagnostic = generation_diagnostics(reports, through)
    frozen, renderings = freeze(protocol, reports, paths, through, evaluation)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roster, cache_path = output / "frozen.json", output / "cache.json"
        if roster.exists() and confirm.read(roster) != frozen:
            raise ValueError("Component confirmation roster or source changed; use a new output directory")
        if not roster.exists():
            confirm.save(roster, frozen)
        fingerprint = cp.fingerprint(frozen)
        cache = confirm.read(cache_path) if cache_path.exists() else {"roster_sha256": fingerprint, "evaluations": {}, "failures": []}
        if cache["roster_sha256"] != fingerprint or set(cache["evaluations"]) - renderings.keys():
            raise ValueError("Unexpected cached component result")
        if renderings.keys() - cache["evaluations"].keys() and evaluation.preflight() != protocol["server"]:
            raise ValueError("Current service identity/version differs from primary evaluation")
        for key, (template, instruction) in renderings.items():
            if key in cache["evaluations"]:
                continue
            adapter = copy(evaluation)
            adapter.template = template
            try:
                cache["evaluations"][key] = adapter.evaluate(instruction, "confirmation")
            except ClassificationError as exc:
                cache["failures"].append({"template_sha256": key, "error": str(exc),
                                          "classification_requests": exc.classification_requests, "usage": exc.usage})
                confirm.save(cache_path, cache)
                raise
            confirm.save(cache_path, cache)
            print(f"Confirmed {len(cache['evaluations'])}/{len(renderings)} component combinations", flush=True)
        result = comparison_report(frozen, reports, hashes, diagnostic, cache, evaluation)
        confirm.save(output / "comparison.json", result)
        if summary_path:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            confirm.save(summary_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="Primary search protocol directory")
    parser.add_argument("--elite", type=Path, required=True, help="Completed elite component performance.json")
    parser.add_argument("--tournament", type=Path, default=HISTORICAL)
    parser.add_argument("--through-generation", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True, help="Private directory for resumable evaluations and winners")
    parser.add_argument("--summary", type=Path, help="Optional aggregate-only JSON export")
    args = parser.parse_args()
    p = confirm.read(args.suite / "protocol.json")
    evaluation = Evaluation(p["mode"], p["search_size"], p["confirmation_size"], p["split_seed"], p["workers"])
    compare({"tournament": args.tournament, "elite": args.elite}, evaluation, p, args.through_generation, args.output, args.summary)


if __name__ == "__main__":
    main()
