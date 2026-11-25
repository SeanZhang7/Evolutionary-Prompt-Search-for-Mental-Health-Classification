"""Evolve three DepSign prompt components; evaluate and select whole combinations."""

from copy import deepcopy
import json
from pathlib import Path

if __package__:
    from . import run_ga as ga, depsign_evaluator as evaluator, component_prompts as cp
    from .component_report import ComponentReport
else:
    import run_ga as ga
    import depsign_evaluator as evaluator
    import component_prompts as cp
    from component_report import ComponentReport


def counter():
    return {"requests": 0, "responses_with_usage": 0,
            "reported_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def record_cost(cost, result):
    cost["requests"] += 1
    if isinstance(result.get("usage"), dict):
        cost["responses_with_usage"] += 1
        for key in cost["reported_usage"]:
            cost["reported_usage"][key] += result["usage"].get(key, 0) or 0


def individual(identifier, components, fixed, order, **metadata):
    return {"id": identifier, "components": deepcopy(components), "instruction": components["instruction"],
            "combination_sha256": cp.fingerprint({"components": components, "fixed": fixed}),
            "fixed_examples_sha256": fixed["examples_sha256"], "creation_order": order,
            "status": "pending", "semantic_review": "not_assessed", **metadata}


def run_components(output, *, config_file=None, label_mode="three_class", data_file=None,
                   seed=11, generations=None, parent_selection=None, prepare_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"task": "depsign", "search_space": "components", "label_mode": label_mode,
               "purpose": "whole_prompt_component_optimization_on_development_data_not_component_causal_attribution",
               "status": "running", "stage": "prepare", "error": None, "seed": seed,
               "context_tokens": evaluator.CONTEXT_TOKENS, "population": [], "generation_metrics": [],
               "retained_populations": [], "selections": [], "generation_attempts": [], "skipped_generation_slots": [],
               "candidate_evaluations": 0, "candidate_evaluation_attempts": 0, "completed_generations": 0,
               "early_stopping_enabled": False, "stop_reason": None,
               "duplicate_offspring_skipped": 0, "invalid_offspring_skipped": 0,
               "copied_components": {name: 0 for name in cp.COMPONENTS},
               "cost": {"generation": {name: counter() for name in cp.COMPONENTS}, "classification": counter()},
               "sources": {}, "rules": {"deduplication": "exact_full_combination_across_run",
                                           "component_selection": "same_parent_pair_independent_operator_draws"}}
    report = None
    print(f"Component GA output directory: {output}", flush=True)
    try:
        report = ComponentReport(output)  # Plot/similarity dependencies checked before inference.
        spec = evaluator.task_metadata(label_mode)
        summary.update(labels=spec["labels"], label_mapping=spec["label_mapping"])

        def read_source(name, path):
            path = Path(path)
            if not path.is_absolute():
                path = ga.ROOT / path
            path = path.resolve()
            data = path.read_bytes()
            summary["sources"][name] = {"path": str(path), "sha256": evaluator.digest(data)}
            return path, data

        _, config_data = read_source("config", config_file or ga.CONFIG)
        config = cp.read_json(config_data)
        summary["config"] = config
        settings = config["component_search"]
        parameters, attempts, retries = ga.load_settings(config)
        limits = settings["components"]
        if set(limits) != set(cp.COMPONENTS) or set(settings["modes"]) != set(evaluator.LABEL_MODES):
            raise ValueError("Component config requires three components and both label modes")
        for name, limit in limits.items():
            for key in ("max_tokens", "generation_max_tokens"):
                if type(limit[key]) is not int or limit[key] < 1:
                    raise ValueError(f"{name}.{key} must be a positive integer")
        if limits["instruction"]["max_tokens"] > evaluator.MAX_INSTRUCTION_TOKENS:
            raise ValueError("Instruction limit exceeds shared evaluator limit")
        generations = ga.validate_generations(settings["generations_after_initialization"] if generations is None else generations)
        selection_mode = ga.normalize_parent_selection(parent_selection or config["search"]["parent_selection"])
        crossover, mutation = (config["search"][k] for k in ("crossover_probability", "mutation_probability"))
        summary.update(requested_generations=generations, generation_parameters=parameters,
                       classification_parameters=dict(evaluator.PARAMETERS), model=evaluator.MODEL)
        summary["rules"].update(parent_selection=selection_mode, survivor_selection=config["search"]["survivor_selection"],
                                tie_break=config["search"]["tie_break"], crossover_probability=crossover,
                                mutation_probability=mutation, max_proposal_attempts=attempts,
                                max_transport_retries=retries, component_limits=limits)
        tokenizer = evaluator.load_tokenizer()
        model_path = evaluator.ROOT / "LLM/Qwen3.5-9B"
        summary["tokenizer_files_sha256"] = {name: evaluator.digest((model_path / name).read_bytes())
                                             for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")}
        initial = {}
        for mode, mode_paths in settings["modes"].items():
            blobs = {name: read_source(f"{mode}_{name}", mode_paths[name + "_file"])[1] for name in cp.COMPONENTS}
            initial[mode] = cp.load_initial(blobs, mode, tokenizer, limits)
        base, entries, fixed = initial[label_mode]
        summary["fixed_context"] = fixed
        summary["initial_prompt_description"] = cp.read_json(read_source(
            "selected_instruction_source", settings["modes"][label_mode]["instruction_file"])[1]).get("description")
        operators = {name: cp.load_operator(read_source(f"operator_{name}", limits[name]["operator_prompt_file"])[1])
                     for name in cp.COMPONENTS}
        summary["operators"] = operators
        data_path, data = read_source("evaluation_data", Path(data_file).resolve() if data_file else settings["data_file"])
        samples = evaluator.load_data(data_path, data=data)["selected"]
        for _, _, mode_fixed in initial.values():
            cp.check_overlap(mode_fixed, samples)
        manifest_path = data_path.with_suffix(".manifest.json")
        if manifest_path.is_file():
            manifest = cp.read_json(read_source("sampling_manifest", manifest_path)[1])
            if manifest.get("sample_sha256") != evaluator.digest(data):
                raise ValueError("Evaluation data differs from sampling manifest")
            summary["sampling_seed"] = manifest.get("sampling_seed")
        summary["sample_count"] = len(samples)
        summary["sample_ids"] = [r["PID"] for r in samples]
        summary["class_counts"] = {label: sum(spec["label_mapping"][r["Label"]] == label for r in samples)
                                    for label in spec["labels"]}
        seen = {}
        for entry in entries:
            components = {**base, "instruction": entry["instruction"]}
            p = individual(f"parent_{len(summary['population']) + 1:03d}", components, fixed,
                           len(summary["population"]), generation=0, parent_ids=[], source_id=entry["id"])
            if p["combination_sha256"] not in seen:
                summary["population"].append(p)
                seen[p["combination_sha256"]] = p["id"]
        parents = list(summary["population"])
        n = len(parents)
        if n < 2:
            raise ValueError("Require at least two unique initial combinations")
        summary.update(population_size=n, initial_duplicates_removed=len(entries) - n,
                       candidate_evaluation_budget=(generations + 1) * n,
                       planned_classification_requests=(generations + 1) * n * len(samples),
                       generation_slots=generations * n, maximum_component_proposals_before_retries=3 * generations * n)

        def prepare(p):
            candidate = evaluator.prepare_candidate(p["instruction"], samples, tokenizer,
                         task_template=cp.render_template(p["components"], fixed), label_mode=label_mode)
            p["truncated_count"] = sum(r["truncated"] for r in candidate["requests"])
            return candidate

        for p in parents:
            prepare(p)
            p["status"] = "prepared"
        for name in cp.COMPONENTS:
            for ops in ({"crossover": True, "mutation": True}, {"crossover": True, "mutation": False},
                        {"crossover": False, "mutation": True}):
                cp.build_request(name, parents[:2], ops, fixed, operators[name], limits[name], parameters,
                                 tokenizer, prepare_only=True)
        print(f"Initial population: {n}; mode: {label_mode}; parent selection: {selection_mode}; early stopping: disabled", flush=True)
        if prepare_only:
            summary.update(status="prepared", stage="prepared")
        else:
            models = json.loads(evaluator.http_call("/v1/models"))
            version = json.loads(evaluator.http_call("/version"))
            identity = ga.pipeline.model_identity({"server_models": models, "model": evaluator.MODEL})
            if identity.get("max_model_len") is not None and identity["max_model_len"] < evaluator.CONTEXT_TOKENS:
                raise ValueError("Server context is smaller than configured context")
            summary["service"] = {"identity": identity, "version": version}

            def evaluate(p):
                summary["stage"] = f"evaluation:{p['id']}"
                candidate = prepare(p)
                p["status"] = "running"
                summary["candidate_evaluation_attempts"] += 1
                partial = []

                def on_result(index, total, result):
                    record_cost(summary["cost"]["classification"], result)
                    partial.append({key: result.get(key) for key in ("gold", "prediction", "request_error", "finish_reason")})
                    partial[-1]["reasoning"] = bool(result.get("reasoning"))
                    evaluator.print_prediction(index, total, result)

                print(f"Evaluating {p['id']} ({len(samples)} posts)", flush=True)
                try:
                    result = evaluator.evaluate_candidate(candidate, on_result=on_result)
                except (Exception, KeyboardInterrupt):
                    if partial:
                        p["performance"] = evaluator.calculate_metrics(partial, label_mode)
                        p["performance"]["partial"] = True
                    raise
                p["performance"] = result["metrics"]
                p["macro_f1"] = result["metrics"]["macro_f1"]
                if result["metrics"]["request_error_count"]:
                    p["status"] = "request_error"
                    raise RuntimeError(f"{p['id']} has classification request errors; search stopped")
                p["status"] = "complete"
                summary["candidate_evaluations"] += 1
                print(f"{p['id']}: macro-F1={p['macro_f1']:.4f}", flush=True)

            def retain(generation, population):
                metrics = ga.generation_metrics(generation, population)
                metrics["unique_prompt_count"] = len({p["combination_sha256"] for p in population})
                metrics["retained_ids"] = [p["id"] for p in population]
                summary["generation_metrics"].append(metrics)
                summary["retained_populations"].append({"generation": generation, "ids": metrics["retained_ids"]})
                summary["last_retained_population_ids"] = metrics["retained_ids"]
                summary["completed_generations"] = generation

            for p in parents:
                evaluate(p)
            retain(0, parents)
            for generation in range(1, generations + 1):
                selection = ga.select_parents(parents, cp.derived_seed(seed, generation, "parents"), selection_mode)
                draws = cp.operation_draws(seed, generation, n, crossover, mutation)
                selection.update(generation=generation, component_operations=draws)
                summary["selections"].append(selection)
                lookup = {p["id"]: p for p in parents}
                children = []
                for slot, (ids, operations) in enumerate(zip(selection["parent_ids"], draws["choices"]), 1):
                    identifier = f"g{generation:03d}_child_{slot:03d}"
                    chosen = [lookup[i] for i in ids]
                    values, edits = {}, {}
                    try:
                        for name in cp.COMPONENTS:
                            ops = operations[name]
                            summary["stage"] = f"generation:{identifier}:{name}"
                            if not any(ops.values()):
                                values[name] = deepcopy(chosen[0]["components"][name])
                                edits[name] = {"copied": True}
                                summary["copied_components"][name] += 1
                                continue
                            request, tokens = cp.build_request(name, chosen, ops, fixed, operators[name], limits[name], parameters, tokenizer)

                            def record(event):
                                record_cost(summary["cost"]["generation"][name], event)
                                metadata = {key: event.get(key) for key in ("generation", "slot", "proposal_attempt", "transport_retry",
                                            "operations", "outcome", "rejection_reason", "request_error", "finish_reason", "usage")}
                                metadata.update(component=name, seed=event["request"]["seed"], child_id=identifier)
                                summary["generation_attempts"].append(metadata)
                                reason = event.get("rejection_reason") or event.get("request_error")
                                print(f"{identifier} {name}, attempt {event['proposal_attempt']}, retry {event['transport_retry']}: "
                                      f"{event['outcome']}" + (f" — {reason}" if reason else ""), flush=True)

                            edit = ga.generate_child(request, tokenizer, seed=cp.derived_seed(seed, name, "generation"),
                                slot=slot, generation=generation, attempts=attempts, retries=retries, record=record, operations=ops,
                                parse_response=lambda content: cp.parse_response(content, name, ops, fixed, tokenizer, limits[name]["max_tokens"]))
                            values[name] = edit["value"]
                            edits[name] = {**edit, "rendered_prompt_tokens": tokens, "copied": False}
                    except ga.InvalidChildProposalsExhausted as exc:
                        summary["invalid_offspring_skipped"] += 1
                        summary["skipped_generation_slots"].append({"generation": generation, "slot": slot, "id": identifier,
                            "failed_component": name, "parent_ids": ids, "component_operations": operations, "reason": str(exc)})
                        print(f"{identifier}: skipped whole child after invalid {name}", flush=True)
                        continue
                    child = individual(identifier, values, fixed, len(summary["population"]), generation=generation,
                                       parent_ids=ids, component_operations=operations, component_edits=edits)
                    summary["population"].append(child)
                    if child["combination_sha256"] in seen:
                        child.update(status="skipped_duplicate", duplicate_of=seen[child["combination_sha256"]])
                        summary["duplicate_offspring_skipped"] += 1
                        print(f"{identifier}: duplicate complete combination; skipped classification", flush=True)
                        continue
                    seen[child["combination_sha256"]] = identifier
                    evaluate(child)
                    children.append(child)
                summary["stage"] = f"selection:{generation}"
                survivor_seed = cp.derived_seed(seed, generation, "survivors")
                # Whole-combination deduplication has already run, so equal instruction strings stay eligible.
                parents = ga.select_survivors(parents + children, n, "none",
                                             mode=config["search"]["survivor_selection"], seed=survivor_seed)
                retain(generation, parents)
                summary["generation_metrics"][-1]["survivor_selection_seed"] = survivor_seed
                print(f"G{generation}: retained {n} combinations; evaluated {len(children)} new children", flush=True)
            summary.update(status="complete", stage="complete", stop_reason="generation_limit",
                           next_population_ids=[p["id"] for p in parents])
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                       error=f"{type(exc).__name__}: {exc}")
        for p in summary["population"]:
            if p["status"] == "running":
                p["status"] = summary["status"]
    if report is not None:
        report(summary)
    else:
        evaluator.save_json(output / "performance.json", summary)
    print(f"GA status: {summary['status']}; evaluated {summary['candidate_evaluations']} individuals", flush=True)
    if summary["error"]:
        print(f"Error at {summary['stage']}: {summary['error']}", flush=True)
    print(f"Results: {output}", flush=True)
    return summary
