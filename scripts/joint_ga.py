"""Joint binary prompt search: one instruction, two task-specific evaluations."""

from collections import Counter
import json
from pathlib import Path

if __package__:
    from . import run_ga as ga
    from . import depsign_evaluator as evaluator
    from . import multiobjective as mo
    from .joint_report import write_report
    from .prompt_similarity import PromptSimilarityReport
else:
    import run_ga as ga
    import depsign_evaluator as evaluator
    import multiobjective as mo
    from joint_report import write_report
    from prompt_similarity import PromptSimilarityReport


def generation_metrics(generation, population):
    return {"generation": generation, "population_size": len(population),
            "unique_prompt_count": len(ga.unique_instructions(population)),
            "tasks": {task: {"best_macro_f1": max(p["fitness"][task] for p in population),
                              "mean_macro_f1": sum(p["fitness"][task] for p in population) / len(population)}
                      for task in mo.TASKS},
            "pareto_count": len(mo.pareto_front(population)),
            "retained_ids": [p["id"] for p in population]}


def new_individual(identifier, instruction, order, **metadata):
    return {"id": identifier, "instruction": instruction, "creation_order": order,
            "instruction_sha256": evaluator.digest(instruction.encode()), "status": "pending",
            "fitness": {}, "performance": {}, "semantic_review": "not_assessed", **metadata}


def cost_counter():
    return {"requests": 0, "responses_with_usage": 0, "elapsed_seconds": 0.0,
            "reported_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def add_cost(counter, result):
    counter["requests"] += 1
    counter["elapsed_seconds"] += result.get("elapsed_seconds", 0)
    if isinstance(result.get("usage"), dict):
        counter["responses_with_usage"] += 1
        for key in counter["reported_usage"]:
            counter["reported_usage"][key] += result["usage"].get(key, 0) or 0


def run_joint(output, *, config_file=None, prompts_file=None, seed=11, generations=None,
              prepare_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "task": "joint", "label_mode": "binary", "status": "running", "stage": "prepare", "error": None,
        "purpose": "joint_task_development_optimization_not_unseen_task_transfer_or_test",
        "seed": seed, "context_tokens": evaluator.CONTEXT_TOKENS,
        "population": [], "generation_metrics": [], "retained_populations": [], "selections": [],
        "candidate_evaluations": 0, "candidate_evaluation_attempts": 0,
        "task_evaluations": {t: 0 for t in mo.TASKS}, "completed_generations": 0,
        "requested_generations": generations, "early_stopping_enabled": False, "stop_reason": None,
        "generation_requests": 0, "proposal_attempts": 0, "transport_retries": 0,
        "copied_offspring": 0, "duplicate_offspring_skipped": 0, "invalid_offspring_skipped": 0,
        "invalid_generation_outputs": 0, "skipped_generation_slots": [], "generation_attempts": [],
        "cost": {"generation": cost_counter(), "classification": {t: cost_counter() for t in mo.TASKS}},
        "sources": {}, "tasks": {}, "children_per_request": 1,
        "rules": {"objectives": list(mo.TASKS), "direction": "maximize_macro_f1_per_task",
                  "parent_selection": "nsga2_rank_crowding_tournament", "survivor_selection": "nsga2",
                  "deduplication": "exact_text", "deduplication_scope": "all_prior_candidates_in_run",
                  "crowding": "frontwise_range_normalized; constant_objective_contributes_zero",
                  "survivor_tie_break": "creation_order", "parent_tie_break": "seeded_random",
                  "operator_feedback": "none", "generator_task_context": "none"},
    }
    similarity = None
    print(f"Joint GA output directory: {output}", flush=True)
    try:
        # Dependency checks precede all service calls, including prepare-only runs.
        similarity = PromptSimilarityReport(output)

        def read_source(name, path):
            path = Path(path)
            if not path.is_absolute():
                path = ga.ROOT / path
            path = path.resolve()
            data = path.read_bytes()
            summary["sources"][name] = {"path": str(path), "sha256": evaluator.digest(data)}
            return path, data

        _, config_data = read_source("config", config_file or ga.CONFIG)
        config = json.loads(config_data)
        summary["config"] = config
        settings = config["joint_search"]
        if set(settings["tasks"]) != set(mo.TASKS):
            raise ValueError("joint_search.tasks must contain exactly depsign and dreaddit")
        generations = ga.validate_generations(settings["generations_after_initialization"]
                                               if generations is None else generations)
        summary["requested_generations"] = generations
        parameters, attempts, retries = ga.load_settings(config)
        crossover, mutation = (config["search"][key] for key in ("crossover_probability", "mutation_probability"))
        summary["rules"].update(crossover_probability=crossover, mutation_probability=mutation,
                                max_proposal_attempts=attempts, max_transport_retries=retries,
                                max_instruction_tokens=evaluator.MAX_INSTRUCTION_TOKENS)
        _, prompts_data = read_source("initial_prompts", Path(prompts_file).resolve() if prompts_file else settings["prompts_file"])
        summary["initial_prompt_description"] = json.loads(prompts_data).get("description")
        entries = ga.pipeline.load_prompts(prompts_data)
        summary["input_prompt_count"] = len(entries)
        entries = ga.unique_instructions(entries)
        summary["initial_duplicates_removed"] = summary["input_prompt_count"] - len(entries)
        n = len(entries)
        if n < 2:
            raise ValueError("Joint search requires at least two unique initial instructions")
        summary.update(population_size=n, candidate_evaluation_budget=(generations + 1) * n,
                       offspring_per_generation=n, budget_interpretation="upper_bound_before_skips")
        _, template_data = read_source("generation_template", settings["generation_template_file"])
        template = template_data.decode("utf-8")
        _, operator_data = read_source("operator_prompt", settings["operator_prompt_file"])
        operator = ga.load_operator_prompts(operator_data, settings.get("operator_prompt_name", "baseline"))
        if operator.get("feedback", "none") != "none" or operator.get("children_per_request", 1) != 1:
            raise ValueError("Joint operators require feedback=none and children_per_request=1")
        if "generation_max_tokens" in operator:
            parameters = {**parameters, "max_tokens": operator["generation_max_tokens"]}
        summary.update(operator_prompt=operator, generation_template=template,
                       generation_parameters=parameters, classification_parameters=dict(evaluator.PARAMETERS),
                       model=evaluator.MODEL)
        tokenizer = evaluator.load_tokenizer()
        model_path = evaluator.ROOT / "LLM/Qwen3.5-9B"
        summary["tokenizer_files_sha256"] = {
            name: evaluator.digest((model_path / name).read_bytes())
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")}
        tasks = {}
        for task in mo.TASKS:
            paths = settings["tasks"][task]
            data_path, data = read_source(f"{task}_data", paths["data_file"])
            template_path, task_data = read_source(f"{task}_template", paths["task_template_file"])
            task_template = task_data.decode("utf-8")
            evaluator.task_background(task_template)  # Required Instruction heading, never sent to generator.
            spec = evaluator.task_metadata("binary", dataset=task, task_template_file=template_path)
            loaded = evaluator.load_data(data_path, data=data, dataset=task)
            samples = loaded["selected"]
            tasks[task] = (samples, task_template)
            summary["tasks"][task] = {**spec, "data_file": str(data_path), "sample_count": len(samples),
                                      "sample_ids": [r["PID"] for r in samples],
                                      "class_counts": dict(Counter(spec["label_mapping"][r["Label"]] for r in samples))}
            manifest_path = data_path.with_suffix(".manifest.json")
            if manifest_path.is_file():
                _, manifest_data = read_source(f"{task}_sampling_manifest", manifest_path)
                manifest = json.loads(manifest_data)
                if manifest.get("sample_sha256") != evaluator.digest(data):
                    raise ValueError(f"{task} sample differs from its sampling manifest")
                summary["tasks"][task]["sampling_seed"] = manifest.get("sampling_seed")
        summary["planned_classification_requests"] = (summary["candidate_evaluation_budget"]
                                                       * sum(len(samples) for samples, _ in tasks.values()))
        summary["sources"]["runner"] = {"path": str(Path(__file__)),
                                          "sha256": evaluator.digest(Path(__file__).read_bytes())}
        for index, entry in enumerate(entries):
            instruction, _ = evaluator.validate_instruction(entry["instruction"], tokenizer)
            summary["population"].append(new_individual(
                f"parent_{index + 1:03d}", instruction, index, source_id=entry["id"],
                generation=0, parent_ids=[], provenance="provided_file_see_initial_prompt_description"))
        parents = list(summary["population"])
        seen = {p["instruction"]: p["id"] for p in parents}

        def prepare(item):
            return {task: evaluator.prepare_candidate(item["instruction"], samples, tokenizer,
                                                      task_template=task_template, label_mode="binary", dataset=task)
                    for task, (samples, task_template) in tasks.items()}

        # Validate every initial classification request, without storing N copies of data.
        for item in parents:
            prepared = prepare(item)
            item["truncated_counts"] = {task: sum(r["truncated"] for r in candidate["requests"])
                                        for task, candidate in prepared.items()}
            item["status"] = "prepared"
        del prepared
        # Validate all operator branches even when their current probability is zero.
        for operations in ({"crossover": True, "mutation": True},
                           {"crossover": True, "mutation": False},
                           {"crossover": False, "mutation": True}):
            ga.build_request(None, template, parents[:2], parameters, tokenizer, operations, operator)
        print(f"Initial population: {n}; NSGA-II; binary tasks: depsign + dreaddit; early stopping: disabled", flush=True)
        if prepare_only:
            summary.update(status="prepared", stage="prepared")
        else:
            models = json.loads(evaluator.http_call("/v1/models"))
            version = json.loads(evaluator.http_call("/version"))
            identity = ga.pipeline.model_identity({"server_models": models, "model": evaluator.MODEL})
            if identity.get("max_model_len") is not None and identity["max_model_len"] < evaluator.CONTEXT_TOKENS:
                raise ValueError("Server context is smaller than the configured context")
            summary["service"] = {"identity": identity, "version": version}

            def evaluate(item):
                summary["stage"] = f"evaluation:{item['id']}"
                candidates = prepare(item)  # Check both task budgets before sending either task.
                item["status"] = "running"
                summary["candidate_evaluation_attempts"] += 1
                item["truncated_counts"] = {task: sum(r["truncated"] for r in candidate["requests"])
                                            for task, candidate in candidates.items()}
                for task, candidate in candidates.items():
                    summary["stage"] = f"evaluation:{item['id']}:{task}"
                    partial = []

                    def on_result(index, total, result):
                        add_cost(summary["cost"]["classification"][task], result)
                        # Enough to recover partial metrics, without retaining responses or posts.
                        partial.append({key: result.get(key) for key in
                                        ("gold", "prediction", "request_error", "finish_reason")})
                        partial[-1]["reasoning"] = bool(result.get("reasoning"))
                        evaluator.print_prediction(index, total, result)

                    print(f"Evaluating {item['id']} on {task} ({len(candidate['requests'])} posts)", flush=True)
                    try:
                        evaluation = evaluator.evaluate_candidate(candidate, on_result=on_result)
                    except (Exception, KeyboardInterrupt):
                        if partial:
                            item["performance"][task] = evaluator.calculate_metrics(partial, "binary", dataset=task)
                            item["performance"][task]["partial"] = True
                        raise
                    metrics = evaluation["metrics"]
                    item["performance"][task] = metrics
                    summary["task_evaluations"][task] += 1
                    if metrics["request_error_count"]:
                        item["status"] = "request_error"
                        raise RuntimeError(f"{item['id']} on {task} has request errors; partial candidate excluded from selection")
                    item["fitness"][task] = metrics["macro_f1"]
                    print(f"{item['id']} {task}: macro-F1={metrics['macro_f1']:.4f}; "
                          f"invalid_output_rate={metrics['invalid_output_rate']:.4f}", flush=True)
                    del evaluation
                item["status"] = "complete"
                summary["candidate_evaluations"] += 1

            def retain(generation, population):
                summary["generation_metrics"].append(generation_metrics(generation, population))
                ids = [p["id"] for p in population]
                summary["retained_populations"].append({"generation": generation, "ids": ids})
                summary["last_retained_population_ids"] = ids
                summary["completed_generations"] = generation

            def record(event):
                add_cost(summary["cost"]["generation"], event)
                summary["generation_requests"] += 1
                summary["proposal_attempts"] += event["transport_retry"] == 0
                summary["transport_retries"] += event["transport_retry"] > 0
                summary["invalid_generation_outputs"] += event["outcome"] == "invalid"
                summary["generation_attempts"].append({key: event.get(key) for key in (
                    "generation", "slot", "operations", "proposal_attempt", "transport_retry", "outcome",
                    "rejection_reason", "request_error", "finish_reason", "usage")})
                summary["generation_attempts"][-1]["seed"] = event["request"]["seed"]
                reason = event.get("rejection_reason") or event.get("request_error")
                print(f"G{event['generation']} child {event['slot']}, attempt {event['proposal_attempt']}, "
                      f"retry {event['transport_retry']}: {event['outcome']}" + (f" — {reason}" if reason else ""), flush=True)

            for item in parents:
                evaluate(item)
            retain(0, parents)
            for generation in range(1, generations + 1):
                selection = mo.select_parents(parents, ga.selection_seed(seed, generation))
                operators = ga.select_operators(seed, generation, n, crossover, mutation)
                selection.update(generation=generation, operator_selection=operators)
                summary["selections"].append(selection)
                lookup = {p["id"]: p for p in parents}
                children = []
                for slot, (ids, operations) in enumerate(zip(selection["parent_ids"], operators["choices"]), 1):
                    identifier = f"g{generation:03d}_child_{slot:03d}"
                    summary["stage"] = f"generation:{identifier}"
                    chosen = [lookup[i] for i in ids]
                    if any(operations.values()):
                        request, tokens = ga.build_request(None, template, chosen, parameters, tokenizer,
                                                           operations, operator)
                        try:
                            child = ga.generate_child(request, tokenizer, seed=seed, slot=slot, generation=generation,
                                                      attempts=attempts, retries=retries, record=record, operations=operations)
                        except ga.InvalidChildProposalsExhausted as exc:
                            summary["invalid_offspring_skipped"] += 1
                            summary["skipped_generation_slots"].append({"id": identifier, "generation": generation,
                                                                       "slot": slot, "parent_ids": ids,
                                                                       "operations": operations, "reason": str(exc)})
                            print(f"{identifier}: invalid proposals exhausted; skipped", flush=True)
                            continue
                        provenance = "qwen_ga_offspring"
                    else:
                        child = {"instruction": chosen[0]["instruction"], "crossover": None,
                                 "seed": None, "proposal_attempt": 0}
                        tokens, provenance = 0, "ga_copy_first_parent"
                        summary["copied_offspring"] += 1
                    item = new_individual(identifier, child.pop("instruction"), len(summary["population"]),
                                          generation=generation, parent_ids=ids, operations=operations,
                                          provenance=provenance, rendered_prompt_tokens=tokens, **child)
                    summary["population"].append(item)
                    if item["instruction"] in seen:
                        item.update(status="skipped_duplicate", duplicate_of=seen[item["instruction"]])
                        summary["duplicate_offspring_skipped"] += 1
                        print(f"{identifier}: duplicate of {item['duplicate_of']}; skipped classification", flush=True)
                        continue
                    seen[item["instruction"]] = identifier
                    children.append(item)
                    evaluate(item)
                summary["stage"] = f"selection:{generation}"
                parents = mo.select_survivors(parents + children, n)
                retain(generation, parents)
                print(f"G{generation}: retained {n}; evaluated {len(children)} new instructions", flush=True)
            summary.update(status="complete", stage="complete", stop_reason="generation_limit",
                           next_population_ids=[p["id"] for p in parents])
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                       error=f"{type(exc).__name__}: {exc}")
        for item in summary["population"]:
            if item["status"] == "running":
                item["status"] = summary["status"]
    write_report(output, summary)
    if similarity is not None and summary["status"] != "prepared":
        similarity(summary)
    print(f"GA status: {summary['status']}; evaluated {summary['candidate_evaluations']} individuals", flush=True)
    if summary["error"]:
        print(f"Error at {summary['stage']}: {summary['error']}", flush=True)
    print(f"Results: {output}", flush=True)
    return summary
