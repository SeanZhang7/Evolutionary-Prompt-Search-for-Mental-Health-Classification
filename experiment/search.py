"""Run fixed-budget DepSign instruction search with matched seeds and a locked confirmation stage."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from time import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import operators
from experiment.evaluation import Evaluation
from scripts import depsign_evaluator as ev, qwen_client
from scripts.run_ga import select_survivors

ARMS = ("tournament", "elite", "de", "random")
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def key(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def derived_seed(*parts):
    return int(key(parts)[:8], 16) % (2**31)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_for(args, evaluation, server):
    model = ROOT / "LLM/Qwen3.5-9B"
    settings = qwen_client.load_settings(read(ROOT / "configs/experiment_v1.json"))[0]
    if settings["chat_template_kwargs"]["enable_thinking"]:
        raise ValueError("The experiment requires generation thinking disabled")
    protocol = {
        "schema": 1, "task": "depsign", "mode": args.mode, "arms": list(ARMS), "seeds": args.seeds,
        "population": args.population, "generations": args.generations, "search_size": args.search_size,
        "confirmation_size": args.confirmation_size, "split_seed": args.split_seed, "workers": args.workers,
        "manifest": evaluation.manifest, "server": server, "generation_parameters": settings,
        "initial_selection": "First N ordered initial instructions for all methods and all seeds",
        "sampling": "One generation call per target slot, equal generation settings for every arm",
        "duplicate_policy": "Exact stripped text within each arm, skip duplicate without replacement",
        "source_hashes": {name: ev.digest((ROOT / name).read_bytes()) for name in
                          ("experiment/search.py", "experiment/evaluation.py", "experiment/operators.py",
                           "scripts/depsign_evaluator.py", "scripts/qwen_client.py", "scripts/run_ga.py")},
        "model_metadata": {name: ev.digest((model / name).read_bytes()) for name in
                           ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
                            "model.safetensors.index.json")},
        "weight_files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(model.glob("*.safetensors"))},
    }
    protocol["protocol_hash"] = key(protocol)
    return protocol


class Suite:
    def __init__(self, output, evaluation, protocol):
        self.output = output
        self.evaluation = evaluation
        self.protocol = protocol

    def cached(self, role, signature, origin, compute, cost):
        digest = key([self.protocol["protocol_hash"], role, signature])
        path = self.output / "cache" / role / f"{digest}.json"
        if path.exists():
            receipt = read(path)
        else:
            try:
                value = compute()
            except Exception as error:
                with (self.output / "failures.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"origin": origin, "role": role, "key": digest, "time": time(),
                                        "error": str(error), "actual_requests": getattr(error, "classification_requests", 1 if role == "generation" else None),
                                        "reported_usage": getattr(error, "usage", {})}) + "\n")
                raise
            receipt = {"origin": origin, "value": value}
            save(path, receipt)
        value = receipt["value"]
        counts = cost[role]
        counts["logical_requests"] += value["requests"]
        if receipt["origin"] == origin:  # also attribute work recovered after interruption
            counts["actual_requests"] += value["requests"]
            for k in TOKEN_KEYS:
                counts[k] += value["usage"].get(k, 0)
        else:
            counts["cached_requests"] += value["requests"]
        return value, digest

    def score(self, instruction, origin, cost):
        def evaluate():
            result = self.evaluation.evaluate(instruction, "search")
            return {"metrics": result, "usage": result["usage"], "requests": result["classification_requests"]}
        value, digest = self.cached("classification", instruction, origin, evaluate, cost)
        result = value["metrics"]
        return {"macro_f1": result["macro_f1"], "invalid_output_count": result["invalid_output_count"],
                "classification_cache_key": digest}

    def proposal(self, messages, seed, origin, cost):
        settings = self.protocol["generation_parameters"]
        request = {"messages": messages, "model": ev.MODEL, "seed": seed, **settings}
        tokenizer = self.evaluation.tokenizer
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 **settings["chat_template_kwargs"])
        if len(tokenizer.encode(rendered, add_special_tokens=False)) + settings["max_tokens"] > ev.CONTEXT_TOKENS:
            raise ValueError("Generation input plus reserved output exceeds model context")
        def generate():
            result = qwen_client.generate_once(request)
            if result["request_error"]:
                raise RuntimeError(result["request_error"])
            value = {"requests": 1, "usage": result["usage"] or {}, "instruction": None, "invalid": None}
            try:
                if result["finish_reason"] != "stop" or result["reasoning"]:
                    raise ValueError(f"finish={result['finish_reason']}, reasoning={bool(result['reasoning'])}")
                instruction = operators.parse_proposal(result["raw_output"])
                value["instruction"] = ev.validate_instruction(instruction, tokenizer)[0]
            except ValueError as error:
                value["invalid"] = str(error)
            return value
        return self.cached("generation", request, origin, generate, cost)


def leader(people):
    return max(people, key=lambda p: (p["macro_f1"], -p["creation_order"]))


def record(state, population, generation):
    current = leader(population)
    incumbent = leader(state["evaluated"])
    state["best"] = incumbent
    state["retained"] = [p["id"] for p in population]
    state["generation_metrics"].append({"generation": generation, "best": current["macro_f1"],
        "mean": sum(p["macro_f1"] for p in population)/len(population), "incumbent": incumbent["macro_f1"],
        "retained": state["retained"]})


def search_arm(suite, arm, seed, only_init=False):
    p = suite.protocol
    n = p["population"]
    path = suite.output / "arms" / f"{arm}_{seed}.json"
    state = read(path) if path.exists() else {
        "protocol_hash": p["protocol_hash"], "seed": seed, "arm": arm, "status": "running", "evaluated": [],
        "generation_metrics": [], "events": [], "cost": {"generation": {}, "classification": {}}}
    if state["protocol_hash"] != p["protocol_hash"]:
        raise ValueError(f"Protocol mismatch: {path}")
    if state["status"] == "complete":
        return
    cost = {role: Counter(tally) for role, tally in state["cost"].items()}
    state["cost"] = cost
    if not state["generation_metrics"]:
        for i, text in enumerate(suite.evaluation.initial_instructions[:n]):
            identifier = f"g0_{i+1}"
            scores = suite.score(text, [arm, seed, identifier], cost)
            state["evaluated"].append(dict(id=identifier, instruction=text, creation_order=i, generation=0, **scores))
        state["initial_best"] = leader(state["evaluated"])
        record(state, state["evaluated"], 0)
        save(path, state)
        print(f"{arm}/{seed} G0 best {state['best']['macro_f1']:.4f}", flush=True)
    if only_init:
        return
    by_id = {row["id"]: row for row in state["evaluated"]}
    seen = {row["instruction"]: row["id"] for row in state["evaluated"]}
    operator = "ga" if arm in ("elite", "tournament") else arm
    for g in range(len(state["generation_metrics"]), p["generations"] + 1):
        parents = [by_id[identifier] for identifier in state["retained"]]
        proposals = operators.build_generation_proposals(operator, [x["instruction"] for x in parents],
                    [x["macro_f1"] for x in parents], random.Random(derived_seed(seed, g, "parents")))
        survivors = list(parents)
        children = []
        for i, (messages, roles) in enumerate(proposals):
            identifier = f"g{g}_{i+1}"
            request_seed = derived_seed(seed, g, i, "generation")
            origin = [arm, seed, identifier]
            result, digest = suite.proposal(messages, request_seed, origin, cost)
            text = result["instruction"]
            event = {"id": identifier, "generation": g, "seed": request_seed, "slot": i,
                     "parent_ids": {k: parents[v]["id"] for k,v in roles.items()}, "generation_cache_key": digest}
            if result["invalid"]:
                event.update(status="invalid", reason=result["invalid"])
            elif text in seen:
                event.update(status="duplicate", duplicate_of=seen[text])
            else:
                scores = suite.score(text, origin, cost)
                child = dict(id=identifier, instruction=text, creation_order=g*n+i, generation=g, **scores)
                children.append(child)
                state["evaluated"].append(child)
                seen[text] = identifier
                by_id[identifier] = child
                event.update(status="evaluated", macro_f1=child["macro_f1"])
                if arm == "de" and child["macro_f1"] > parents[i]["macro_f1"]:
                    survivors[i] = child
            state["events"].append(event)
        if arm != "de":
            mode = "mu_plus_lambda_tournament" if arm == "tournament" else "mu_plus_lambda_top_mu"
            survivors = select_survivors(parents + children, n, mode=mode, seed=derived_seed(seed,g,"survivors"))
        record(state, survivors, g)
        if g == p["generations"]:
            state["status"] = "complete"
        save(path, state)
        print(f"{arm}/{seed} G{g}: {len(children)}/{n} unique; best={leader(survivors)['macro_f1']:.4f} "
              f"mean={state['generation_metrics'][-1]['mean']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"experiment/results/live")
    parser.add_argument("--mode", choices=("three_class", "binary"), default="three_class")
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,29,47])
    parser.add_argument("--search-size", type=int, default=100)
    parser.add_argument("--confirmation-size", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=20261125)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stage", choices=("init","search","all"), default="all")
    args = parser.parse_args()
    if args.generations < 1 or args.population < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require at least one generation, three individuals, and unique seeds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another runner is writing this suite.\n")
        run(args, output)


def run(args, output):
    evaluation = Evaluation(args.mode, args.search_size, args.confirmation_size, args.split_seed, args.workers)
    if args.population > len(evaluation.initial_instructions) or len(set(evaluation.initial_instructions[:args.population])) != args.population:
        raise ValueError("Need N distinct starting instructions from the fixed input file")
    protocol = protocol_for(args, evaluation, evaluation.preflight())
    path = output / "protocol.json"
    if path.exists():
        if read(path) != protocol:
            raise ValueError("Saved protocol differs; rerun with the original configuration or choose a new directory")
    else:
        save(path, protocol)
    suite = Suite(output, evaluation, protocol)
    print(f"Suite {output}, protocol {protocol['protocol_hash'][:12]}", flush=True)
    for seed in args.seeds:
        for arm in ARMS:
            search_arm(suite, arm, seed, only_init=args.stage == "init")
    if args.stage == "all":
        from experiment.confirm import confirm_suite
        confirm_suite(output, evaluation, protocol)


if __name__ == "__main__":
    main()
