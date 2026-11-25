"""Continue the three frozen feedback arms from G4 to G12 without replaying earlier generations."""

import argparse
from collections import Counter
import fcntl
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import confirm, feedback, search
from experiment.evaluation import Evaluation, ev


BOUNDARY, HORIZON = 4, 12
RUNS = ROOT / "experiment/runs"


def verify_prefix(state, old, evl, source, receipts):
    n = old["population"]
    records = state["evaluated"]
    if (state["status"] != "complete" or state["protocol_hash"] != old["protocol_hash"]
            or state["arm"] != "feedback" or [g["generation"] for g in state["generation_metrics"]] != list(range(BOUNDARY+1))
            or [p["instruction"] for p in records[:n]] != evl.initial_instructions[:n]
            or len({p["id"] for p in records}) != len(records) or len({p["instruction"] for p in records}) != len(records)):
        raise ValueError(f"Incomplete or inconsistent G4 archive for seed {state['seed']}")
    for p in records:
        if not 0 <= p["generation"] <= BOUNDARY:
            raise ValueError("Unexpected generation in source archive")
        key = search.key([old["protocol_hash"], "classification", p["instruction"]])
        if key != p["classification_cache_key"]:
            raise ValueError("Source classification key does not correspond to the scored instruction")
        path = source / "cache/classification" / (key + ".json")
        receipt = search.read(path)["value"]
        feedback.check_metrics(receipt["metrics"], evl, "search")
        if (receipt["metrics"]["macro_f1"] != p["macro_f1"] or receipt["metrics"]["invalid_output_count"] != p["invalid_output_count"]
                or receipt["requests"] != len(evl.splits["search"]) or receipt["metrics"]["usage"] != receipt["usage"]):
            raise ValueError("Source archive and classification receipt differ")
        receipts[key] = ev.digest(path.read_bytes())
    retained = records[:n]
    for g, summary in enumerate(state["generation_metrics"]):
        if g:
            events = [e for e in state["events"] if e["generation"] == g]
            if [e["parent_id"] for e in events] != [p["id"] for p in retained]:
                raise ValueError("G4 archive parent order mismatch")
            children = [p for p in records if p["generation"] == g]
            if [p["id"] for p in children] != [e["id"] for e in events if e["status"] == "evaluated"]:
                raise ValueError("G4 archive offspring do not match the events")
            retained = search.select_survivors(retained + children, n, mode="mu_plus_lambda_top_mu")
        if ([p["id"] for p in retained] != summary["retained"] or search.leader(retained)["macro_f1"] != summary["best"]
                or not math.isclose(sum(p["macro_f1"] for p in retained)/n, summary["mean"], abs_tol=1e-14)
                or summary["incumbent"] != search.leader([p for p in records if p["generation"] <= g])["macro_f1"]):
            raise ValueError("G4 selection or fitness statistics cannot be reproduced from the archive")
    if state["retained"] != [p["id"] for p in retained] or state["best"] != search.leader(records):
        raise ValueError("Final G4 parent order or best candidate differs")


def prepare(source, primary_dir, output, live):
    primary = search.read(primary_dir / "protocol.json")
    old = search.read(source / "protocol.json")
    if old != feedback.build_protocol(primary) or old["generations"] != BOUNDARY or old["seeds"] != [11, 29, 47]:
        raise ValueError("Need the complete, unchanged three-seed G4 feedback protocol")
    evl = Evaluation(old["mode"], old["search_size"], old["confirmation_size"], old["split_seed"], old["workers"])
    server = evl.preflight() if live else primary["server"]
    if search.protocol_for(SimpleNamespace(**primary), evl, server) != primary:
        raise ValueError("Primary data, model, code, parameters, or server signature differs")
    source_winners = confirm.frozen_winners(source, old)
    if search.read(source / "confirmation_winners.json") != source_winners:
        raise ValueError("Previous search winner freeze disagrees with the source G4 trajectories")
    states, receipts, arm_hashes = {}, {}, {}
    for seed in old["seeds"]:
        path = source / "arms" / f"feedback_{seed}.json"
        state = search.read(path)
        verify_prefix(state, old, evl, source, receipts)
        states[seed] = state
        arm_hashes[str(seed)] = ev.digest(path.read_bytes())
    p = {**old, "generations": HORIZON, "confirmation_comparators": ["G0", "G4"],
         "continuation": {"source_protocol_hash": old["protocol_hash"], "boundary": BOUNDARY,
                          "arm_sha256": arm_hashes, "classification_receipts_sha256": receipts,
                          "script_sha256": ev.digest(Path(__file__).read_bytes()),
                          "scope": "Follow-up after examining the G4 trajectories; fixed eight additional generations."}}
    p.pop("protocol_hash")
    p["protocol_hash"] = search.key(p)
    feedback.frozen(output / "protocol.json", p)
    for key, digest in receipts.items():
        # Inherited candidates still reference their original classifier keys in propose().
        source_bytes = (source / "cache/classification" / (key + ".json")).read_bytes()
        dest = output / "cache/classification" / (key + ".json")
        if ev.digest(source_bytes) != digest:
            raise ValueError("A source classification receipt changed during initialization")
        if dest.exists():
            if ev.digest(dest.read_bytes()) != digest:
                raise ValueError("Copied G4 classification receipt differs from its frozen bytes")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(".tmp")
            temp.write_bytes(source_bytes)
            temp.replace(dest)
    for seed, state in states.items():
        path = output / "arms" / f"feedback_{seed}.json"
        if path.exists():
            saved = search.read(path)
            if saved["protocol_hash"] != p["protocol_hash"] or saved["initial_best"] != state["initial_best"]:
                raise ValueError("Previous continuation belongs to another protocol")
            for field in ["generation_metrics", "evaluated", "events"]:
                if saved[field][:len(state[field])] != state[field]:
                    raise ValueError(f"Continuation modified archived {field}")
            if saved["retained"] != saved["generation_metrics"][-1]["retained"]:
                raise ValueError("Continuation resume has an inconsistent retained order")
        else:
            search.save(path, {**state, "protocol_hash": p["protocol_hash"], "status": "running"})
    return ContinuationSuite(output, evl, p, primary_dir, source), old, source_winners


class ContinuationSuite(feedback.FeedbackSuite):
    def __init__(self, output, evl, protocol, primary_dir, source):
        super().__init__(output, evl, protocol, primary_dir)
        self.source = source

    def inherit_score(self, role, instruction):
        old_key = search.key([self.protocol["continuation"]["source_protocol_hash"], role, instruction])
        new_key = search.key([self.protocol["protocol_hash"], role, instruction])
        # Only archived search scores verified against evaluated source records may be reused.
        if role == "classification" and old_key not in self.protocol["continuation"]["classification_receipts_sha256"]:
            return
        old_path = self.source / "cache" / role / (old_key + ".json")
        new_path = self.output / "cache" / role / (new_key + ".json")
        if old_path.is_file() and not new_path.exists():
            value = search.read(old_path)["value"]
            feedback.check_metrics(value["metrics"], self.evaluation, "search" if role == "classification" else "confirmation")
            if role == "classification" and ev.digest(old_path.read_bytes()) != self.protocol["continuation"]["classification_receipts_sha256"][old_key]:
                raise ValueError("Old classification receipt changed after protocol freeze")
            search.save(new_path, {"origin": ["inherited_g4", old_key], "value": value})

    def score_candidate(self, text, origin, cost):
        self.inherit_score("classification", text)
        return super().score_candidate(text, origin, cost)


def confirm_extension(suite, old_roster):
    p, out = suite.protocol, suite.output
    winners = confirm.frozen_winners(out, p)
    roster = ([{**w, "arm": "G4"} for w in old_roster["winner_records"] if w["arm"] == "feedback"]
              + winners["winner_records"])
    feedback.frozen(out / "confirmation_winners.json", {"protocol_hash": p["protocol_hash"], "winner_records": roster})
    # G4 confirmation, if it has already completed, shares the same 300 PID+gold rows.
    evaluations, cost = {}, {"confirmation": Counter()}
    for w in roster:
        text = w["instruction"]
        digest = confirm.text_key(text)
        if digest in evaluations:
            continue
        suite.inherit_score("confirmation", text)
        def classify():
            metrics = suite.evaluation.evaluate(text, split="confirmation")
            return {"metrics": metrics, "requests": metrics["classification_requests"], "usage": metrics["usage"]}
        value, _ = suite.cached("confirmation", text, ["extension", w["arm"], w["seed"]], classify, cost)
        feedback.check_metrics(value["metrics"], suite.evaluation, "confirmation")
        evaluations[digest] = value["metrics"]
    contrasts, results = [], []
    for ref in ("G0", "G4"):
        pairs = [{**w, "arm": "G0" if w["arm"] == ref else "feedback"} for w in roster if w["arm"] in (ref, "feedback")]
        report = confirm.summarize(p, {"winner_records": pairs}, {"evaluations": evaluations, "failures": []})
        results.extend({**r, "arm": ref if r["arm"] == "G0" else "G12"} for r in report["results"] if ref == "G0" or r["arm"] == "G0")
        contrasts.extend({**r, "arm": "G12", "baseline": ref} for r in report["contrasts"])
    search.save(out / "confirmation.json", {"status": "complete", "protocol_hash": p["protocol_hash"],
        "scope": "Same 300 development posts as previous confirmation; the longer horizon was chosen after G4.",
        "results": results, "contrasts": contrasts, "bootstrap": report["bootstrap"], "confirmation_cost": cost["confirmation"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "search", "confirm"), required=True)
    parser.add_argument("--source", type=Path, default=RUNS / "feedback_suite")
    parser.add_argument("--primary", type=Path, default=RUNS / "instruction_suite")
    parser.add_argument("--output", type=Path, default=RUNS / "feedback_g12")
    args = parser.parse_args(argv)
    source, primary, out = args.source.resolve(), args.primary.resolve(), args.output.resolve()
    if any(out == path or out in path.parents or path in out.parents for path in (source, primary)):
        parser.error("Use a continuation directory separate from both original runs")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (out / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite, old, g4 = prepare(source, primary, out, live=args.stage != "prepare")
        if args.stage == "search":
            for seed in old["seeds"]:
                feedback.search_seed(suite, seed)
            feedback.frozen(out / "search_winners.json", confirm.frozen_winners(out, suite.protocol))
        elif args.stage == "confirm":
            if search.read(out / "search_winners.json") != confirm.frozen_winners(out, suite.protocol):
                raise ValueError("G12 search roster has changed")
            confirm_extension(suite, g4)
        print(f"{args.stage} complete: {out}", flush=True)


if __name__ == "__main__":
    main()
