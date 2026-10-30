"""Run configurable GA generations on fixed dev data with optional text deduplication.

Two-stage roulette selection and joint crossover/mutation follow EvoPrompt:
https://github.com/beeevita/EvoPrompt/tree/94caff336555df99acc5c338c7930c40cc550ad9
JSON operators, macro-F1, pre-evaluation deduplication and failure policies are adaptations.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re

if __package__:
    from . import depsign_evaluator as evaluator
    from . import qwen_client
    from . import run_depsign_pipeline_smoke as pipeline
    from .ga_report import CompactReport
    from .prompt_similarity import PromptSimilarityReport
else:
    import depsign_evaluator as evaluator
    import qwen_client
    import run_depsign_pipeline_smoke as pipeline
    from ga_report import CompactReport
    from prompt_similarity import PromptSimilarityReport

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment_v1.json"
TEMPLATE = ROOT / "prompts/ga_crossover_mutation_v1.txt"
DEFAULT_OPERATOR_PROMPT = ROOT / "prompts/crossover_and_mutation_prompt.json"
DEFAULT_DATA = ROOT / "data/processed/depsign/dev_100.tsv"
PURPOSE = "ga_development_pilot_not_held_out_test"
PARENT_SELECTION_MODES = ("wheel", "random", "tour")
RULES = {
    "population_size": "from_prompt_file",
    "offspring_per_generation": "population_size",
    "zero_fitness_selection": "uniform",
    "operator_order": ["crossover", "mutation"],
    "operator_requests_per_proposal": 1,
    "survivor_selection": "mu_plus_lambda_top_mu",
    "tie_break": "creation_order_parents_then_children",
    "candidate_evaluation_budget": "(generations_after_initialization + 1) * population_size",
    "reuse_evaluation_cache": False,
    "max_proposal_attempts_per_slot": 3, "max_transport_retries_per_request": 1,
    "exhausted_slot_action": "skip_invalid_child_and_continue",
    "max_instruction_tokens": evaluator.MAX_INSTRUCTION_TOKENS,
    "max_crossover_tokens": evaluator.MAX_INSTRUCTION_TOKENS,
    "searchable_component": "instruction_block_only",
}


def normalize_parent_selection(mode):
    if mode == "two_stage_roulette_with_replacement":
        return "wheel"
    if mode not in PARENT_SELECTION_MODES:
        raise ValueError("parent_selection must be wheel, random, or tour")
    return mode


def load_settings(config):
    parameters, attempts, retries = qwen_client.load_settings(config)
    search = config["search"]
    normalize_parent_selection(search.get("parent_selection"))
    patience = search.get("early_stopping_patience")
    if type(patience) is not int or patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")
    if search.get("deduplication") not in ("none", "exact_text"):
        raise ValueError("deduplication must be none or exact_text")
    for key in ("crossover_probability", "mutation_probability"):
        probability = search[key]
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{key} must be a number between 0 and 1")
    for key, expected in RULES.items():
        if search.get(key) != expected:
            raise ValueError(f"Unsupported GA setting: {key}")
    expected = {**evaluator.PARAMETERS, "temperature": parameters["temperature"], "top_p": 0.8,
                "top_k": 20, "max_tokens": parameters["max_tokens"],
                "chat_template_kwargs": parameters["chat_template_kwargs"]}
    expected.pop("seed")
    if parameters != expected:
        raise ValueError("GA generation parameters other than temperature, max_tokens and thinking differ from the configured protocol")
    classification = dict(config["model"]["classification"])
    if classification.pop("decoding") != "greedy" or classification != evaluator.PARAMETERS:
        raise ValueError("Configured classifier differs from the shared evaluator")
    if config["evaluation"]["fitness"] != "macro_f1":
        raise ValueError("GA requires macro-F1 fitness")
    return parameters, attempts, retries


def request_seed(seed, slot, attempt, generation=1):
    key = json.dumps([seed, "ga", generation, slot, attempt, "crossover_mutation"],
                     separators=(",", ":")).encode()
    return int(evaluator.digest(key)[:8], 16) % (2 ** 31)


def selection_seed(seed, generation):
    if generation == 1:
        return seed
    key = json.dumps([seed, "ga", generation, "parent_selection"], separators=(",", ":")).encode()
    return int(evaluator.digest(key)[:8], 16) % (2 ** 31)


def select_operators(seed, generation, count, crossover_probability, mutation_probability):
    key = json.dumps([seed, "ga", generation, "operator_selection"], separators=(",", ":")).encode()
    derived_seed = int(evaluator.digest(key)[:8], 16) % (2 ** 31)
    rng = random.Random(derived_seed)
    choices = [{"crossover": rng.random() < crossover_probability,
                "mutation": rng.random() < mutation_probability} for _ in range(count)]
    return {"seed": derived_seed, "choices": choices}


def validate_generations(value):
    if type(value) is not int or value < 1:
        raise ValueError("generations must be a positive integer (G0 initialization is not included)")
    return value


def positive_integer(value):
    try:
        return validate_generations(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def select_parents(population, seed, mode="wheel"):
    """Select N parent pairs using EvoPrompt's wheel, random, or size-two tour rules."""
    mode = normalize_parent_selection(mode)
    n = len(population)
    scores = [p["macro_f1"] for p in population]
    if n < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Require at least two individuals with finite fitness in [0, 1]")
    rng = random.Random(seed)
    weights = None
    tournaments = []
    if mode == "wheel":
        total = sum(scores)
        weights = [s / total for s in scores] if total else [1 / n] * n
        # Preserve the original random-call order for existing wheel seeds.
        pool = rng.choices(range(n), weights=weights, k=n)
        pairs = [rng.sample(range(n), 2) for _ in range(n)]
    else:
        pool = list(range(n))
        if mode == "random":
            pairs = [rng.sample(range(n), 2) for _ in range(n)]
        else:
            pairs = []
            for _ in range(n):
                groups = [rng.sample(range(n), 2), rng.sample(range(n), 2)]
                tournaments.append(groups)
                # max keeps the first drawn competitor on a score tie.
                pairs.append([max(group, key=lambda i: scores[i]) for group in groups])
    result = {"mode": mode, "seed": seed, "weights": weights, "pool_indices": pool,
            "pool_ids": [population[i]["id"] for i in pool], "pair_positions": pairs,
            "parent_ids": [[population[pool[i]]["id"] for i in pair] for pair in pairs]}
    if mode == "tour":
        result.update(tournament_size=2, tournament_tie_break="first_drawn",
                      tournament_groups=tournaments,
                      tournament_group_ids=[[[population[i]["id"] for i in group] for group in groups]
                                            for groups in tournaments])
    return result


def unique_instructions(population):
    """Keep the first occurrence of each instruction, preserving input order."""
    seen = set()
    unique = []
    for item in population:
        key = item["instruction"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def select_survivors(population, count, deduplication="none"):
    ranked = sorted(population, key=lambda p: (-p["macro_f1"], p["creation_order"]))
    if deduplication == "exact_text":
        ranked = unique_instructions(ranked)
        if len(ranked) < count:
            raise ValueError("Fewer unique survivor instructions than population size")
    elif deduplication != "none":
        raise ValueError("deduplication must be none or exact_text")
    return ranked[:count]


def load_operator_prompts(data, name=None):
    prompts = json.loads(data.decode("utf-8"), object_pairs_hook=pipeline.unique_object)
    if isinstance(prompts, dict) and "prompts" in prompts:
        if set(prompts) != {"prompts"} or not isinstance(prompts["prompts"], list) or not prompts["prompts"]:
            raise ValueError("Operator prompt collection must contain a nonempty prompts list")
        by_name = {}
        for item in prompts["prompts"]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                raise ValueError("Every operator prompt entry requires a nonempty name")
            if item["name"] in by_name:
                raise ValueError(f"Duplicate operator prompt name: {item['name']}")
            by_name[item["name"]] = item
        selected = name if name is not None else "baseline"
        if selected not in by_name:
            raise ValueError(f"Unknown operator prompt name {selected!r}; available: {', '.join(by_name)}")
        prompts = by_name[selected]
    elif name is not None and isinstance(prompts, dict) and prompts.get("name") != name:
        raise ValueError(f"Operator prompt file does not contain name {name!r}")
    required = {"name", "both", "crossover_only", "mutation_only"}
    optional = {"description", "feedback", "children_per_request", "generation_max_tokens"}
    if not isinstance(prompts, dict) or not required <= prompts.keys() or prompts.keys() - required - optional:
        raise ValueError("Operator prompt file requires name, both, crossover_only, mutation_only; description is optional")
    if any(not isinstance(prompts[key], str) or not prompts[key].strip() for key in required):
        raise ValueError("Operator prompt fields must be nonempty strings")
    if "description" in prompts and not isinstance(prompts["description"], str):
        raise ValueError("Operator prompt description must be a string")
    if prompts.get("feedback", "none") not in ("none", "parent_metrics"):
        raise ValueError("Operator feedback must be none or parent_metrics")
    count = prompts.get("children_per_request", 1)
    if type(count) is not int or count not in (1, 2):
        raise ValueError("Operator children_per_request must be 1 or 2")
    if "generation_max_tokens" in prompts:
        budget = prompts["generation_max_tokens"]
        if type(budget) is not int or budget < 1:
            raise ValueError("Operator generation_max_tokens must be a positive integer")
    allowed = {"{{MAX_CROSSOVER_TOKENS}}", "{{MAX_INSTRUCTION_TOKENS}}"}
    for key in ("both", "crossover_only", "mutation_only"):
        markers = set(re.findall(r"\{\{.*?\}\}", prompts[key], re.DOTALL))
        if markers - allowed:
            raise ValueError(f"Unsupported placeholder in operator prompt {key}: {sorted(markers - allowed)}")
    return prompts


def operator_contract(operations, prompts=None):
    if not any(operations.values()):
        raise ValueError("No generation request is needed when both operators are disabled")
    if prompts is None:
        prompts = load_operator_prompts(DEFAULT_OPERATOR_PROMPT.read_bytes())
    key = ("both" if operations["crossover"] and operations["mutation"] else
           "crossover_only" if operations["crossover"] else "mutation_only")
    return (prompts[key].replace("{{MAX_CROSSOVER_TOKENS}}", str(RULES["max_crossover_tokens"]))
            .replace("{{MAX_INSTRUCTION_TOKENS}}", str(RULES["max_instruction_tokens"])))


def parent_feedback(parents, operations, *, allow_unscored=False):
    # Mutation-only edits the first parent; the other parent is not evidence for it.
    selected = parents if operations["crossover"] else parents[:1]
    records = []
    for parent in selected:
        record = {"instruction": parent["instruction"]}
        if "performance" not in parent:
            if not allow_unscored:
                raise ValueError(f"Missing measured performance for parent {parent['id']}")
            record["performance"] = None
            record["feedback_status"] = "not_evaluated_preparation_placeholder_only"
        else:
            record.update({key: parent["performance"][key] for key in ("macro_f1", "confusion_matrix")})
        records.append(record)
    return records


def build_request(task, template, parents, parameters, tokenizer, operations=None, operator_prompts=None,
                  *, allow_unscored=False):
    operations = operations if operations is not None else {"crossover": True, "mutation": True}
    if task.count("{{INSTRUCTION}}") != 1 or task.count("{{POST_JSON}}") != 1:
        raise ValueError("Unexpected fixed task template")
    sections = re.split(r"\r?\nInstruction[ \t]*\r?\n", task)
    if len(sections) != 2 or any(marker in sections[0] for marker in ("{{INSTRUCTION}}", "{{POST_JSON}}")):
        raise ValueError("Cannot isolate task definitions from fixed task template")
    if operator_prompts is None:
        operator_prompts = load_operator_prompts(DEFAULT_OPERATOR_PROMPT.read_bytes())
    contract = operator_contract(operations, operator_prompts)
    parent_input = [p["instruction"] for p in parents]
    if operator_prompts.get("feedback", "none") == "parent_metrics":
        parent_input = parent_feedback(parents, operations, allow_unscored=allow_unscored)
        contract = (
            "Parent records include development evaluation feedback. Confusion-matrix rows are true labels; "
            "columns are predicted labels; cells are counts; INVALID means an invalid classification output. "
            "Each row sum is the number of true examples of that class. Metrics belong to the instruction in the same record, "
            "not to the untested crossover or child. Aggregate errors suggest hypotheses, not causes. "
            "Do not copy metrics into the instruction. " + contract)
    replacements = {"{{TASK_SPEC_JSON}}": json.dumps(sections[0].strip(), ensure_ascii=False),
                    "{{PARENTS_JSON}}": json.dumps(parent_input, ensure_ascii=False, separators=(",", ":") if operator_prompts.get("feedback") == "parent_metrics" else None),
                    "{{OPERATIONS_JSON}}": json.dumps(operations),
                    "{{OUTPUT_CONTRACT}}": contract,
                    "{{CHILD_FIELD_RULE}}": (
                        "Within each child object, every requested instruction field must be a nonempty string containing ONLY the short edited instruction block."
                        if operator_prompts.get("children_per_request", 1) == 2 else
                        "Every requested field must be a nonempty string containing ONLY the short edited instruction block.")}
    if any(template.count(key) != 1 for key in replacements):
        raise ValueError("Unexpected GA template placeholders")
    # One substitution pass: placeholders inside supplied parent text are data.
    prompt = re.sub(r"\{\{(?:TASK_SPEC_JSON|PARENTS_JSON|OPERATIONS_JSON|OUTPUT_CONTRACT|CHILD_FIELD_RULE)\}\}",
                    lambda match: replacements[match.group()], template)
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=parameters["chat_template_kwargs"]["enable_thinking"])
    tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    if tokens + parameters["max_tokens"] > evaluator.CONTEXT_TOKENS:
        raise ValueError(f"GA prompt plus output exceeds context budget: {tokens} + {parameters['max_tokens']} > {evaluator.CONTEXT_TOKENS}; "
                         "set PROMPT_SEARCH_CONTEXT_TOKENS consistently for the client and vLLM server")
    return {"model": evaluator.MODEL, "messages": messages, **parameters}, tokens


def parse_child(content, tokenizer, operations=None):
    operations = operations if operations is not None else {"crossover": True, "mutation": True}
    if not isinstance(content, str):
        raise ValueError("Generation content must be text")
    obj = json.loads(content, object_pairs_hook=pipeline.unique_object)
    required = {key for key, enabled in (("crossover", operations["crossover"]),
                                        ("instruction", operations["mutation"])) if enabled}
    if not required or not isinstance(obj, dict) or set(obj) != required:
        raise ValueError(f"Expected exactly these fields for selected operations: {', '.join(sorted(required))}")
    if operations["crossover"]:
        crossover, cross_tokens = evaluator.validate_instruction(obj["crossover"], tokenizer)
    else:
        crossover, cross_tokens = None, 0
    if operations["mutation"]:
        instruction, tokens = evaluator.validate_instruction(obj["instruction"], tokenizer)
    else:
        instruction, tokens = crossover, cross_tokens
    return {"crossover": crossover, "crossover_tokens": cross_tokens,
            "instruction": instruction, "instruction_tokens": tokens}


class InvalidChildProposalsExhausted(RuntimeError):
    """Only invalid model outputs exhausted this slot; service errors are separate."""


def parse_offspring(content, tokenizer, operations, children_per_request):
    """Validate a whole response atomically; never retain one half of an invalid pair."""
    if children_per_request == 1:
        return parse_child(content, tokenizer, operations)
    if children_per_request != 2:
        raise ValueError("Expected one or two children per request")
    if not isinstance(content, str):
        raise ValueError("Generation content must be text")
    obj = json.loads(content, object_pairs_hook=pipeline.unique_object)
    if (not isinstance(obj, dict) or set(obj) != {"children"}
            or not isinstance(obj["children"], list) or len(obj["children"]) != 2):
        raise ValueError("Expected exactly a children array containing two child objects")
    return {"children": [parse_child(json.dumps(child, ensure_ascii=False), tokenizer, operations)
                         for child in obj["children"]]}


def generate_child(request, tokenizer, *, seed, slot, attempts, retries, record, generation=1, operations=None,
                   children_per_request=1):
    operations = operations if operations is not None else {"crossover": True, "mutation": True}
    for attempt in range(1, attempts + 1):
        payload = {**request, "seed": request_seed(seed, slot, attempt, generation)}
        for retry in range(retries + 1):
            event = {"generation": generation, "slot": slot, "operations": operations, "proposal_attempt": attempt,
                     "transport_retry": retry, "request": payload}
            try:
                event.update(qwen_client.generate_once(payload))
            except (Exception, KeyboardInterrupt) as exc:
                event.update(outcome="interrupted" if isinstance(exc, KeyboardInterrupt) else "request_error",
                             request_error=f"{type(exc).__name__}: {exc}")
                record(event)
                raise
            event["outcome"] = "request_error"
            if event["request_error"] is not None:
                record(event)
                if event["retryable"] and retry < retries:
                    continue
                raise RuntimeError(f"Generation request failed: {event['request_error']}")
            try:
                if event["finish_reason"] != "stop":
                    raise ValueError(f"Expected finish_reason=stop; got {event['finish_reason']!r}")
                if event["reasoning"] and not request["chat_template_kwargs"]["enable_thinking"]:
                    raise ValueError("Nonempty reasoning despite thinking disabled")
                child = parse_offspring(event["raw_output"], tokenizer, operations, children_per_request)
                event.update(outcome="accepted", **child)
            except ValueError as exc:
                event.update(outcome="invalid", rejection_reason=str(exc))
            record(event)
            if event["outcome"] == "accepted":
                return {**child, "seed": payload["seed"], "proposal_attempt": attempt}
            break
    raise InvalidChildProposalsExhausted(
        f"Child slot {slot} exhausted {attempts} proposals; last rejection: {event['rejection_reason']}")


def individual(identifier, instruction, order, **metadata):
    return {"id": identifier, "instruction": instruction, "creation_order": order,
            "instruction_sha256": evaluator.digest(instruction.encode("utf-8")),
            "instruction_file": f"instructions/{identifier}.txt",
            "evaluation_directory": f"evaluations/{identifier}",
            "semantic_review": "not_assessed", "status": "pending", **metadata}


def generation_metrics(generation, population):
    return {"generation": generation, "population_size": len(population),
            "unique_prompt_count": len(unique_instructions(population)),
            "mean_macro_f1": sum(item["macro_f1"] for item in population) / len(population),
            "best_macro_f1": max(item["macro_f1"] for item in population)}


def _run_search(output, *, prompts_file=None, data_file=DEFAULT_DATA,
                     seed=11, generations=None, prepare_only=False, save_details=True, label_mode="three_class",
                     parent_selection=None, operator_prompt_file=DEFAULT_OPERATOR_PROMPT, operator_prompt_name=None,
                     task_template_file=None, no_early_stopping=False):
    """Shared search loop; compact mode performs no intermediate file writes."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "stage": "prepare", "purpose": PURPOSE, "seed": seed,
               "population": [], "results": [], "generation_metrics": [], "candidate_evaluations": 0,
               "requested_generations": generations, "completed_generations": 0,
               "stopped_early": False, "stop_reason": None, "unchanged_mean_generations": 0,
               "early_stopping_enabled": not no_early_stopping,
               "selections": [], "retained_populations": [],
               "generation_requests": 0, "proposal_attempts": 0, "transport_retries": 0,
               "copied_offspring": 0,
               "duplicate_offspring_skipped": 0,
               "invalid_offspring_skipped": 0, "skipped_generation_slots": [],
               "invalid_generation_outputs": 0, "generation_request_errors": 0,
               "logged_classification_requests": 0, "error": None,
               "semantic_review": "not_assessed", "rules": dict(RULES),
               "method_reference": "guo2024connecting", "script_sha256": evaluator.digest(Path(__file__).read_bytes())}
    print(f"GA output directory: {output}", flush=True)
    try:
        summary.update(evaluator.task_metadata(label_mode, task_template_file=task_template_file),
                       context_tokens=evaluator.CONTEXT_TOKENS)
        spec = evaluator.task_spec(label_mode, task_template_file=task_template_file)
        paths = {"prompts_snapshot.json": Path(prompts_file or pipeline.default_prompts(label_mode)).resolve(),
                 "data_snapshot.tsv": Path(data_file).resolve(), "config_snapshot.json": CONFIG,
                 "task_template.txt": spec["task_template_path"], "generation_template.txt": TEMPLATE,
                 "operator_prompts.json": Path(operator_prompt_file).resolve()}
        frozen = {name: path.read_bytes() for name, path in paths.items()}
        if save_details:
            for name, data in frozen.items():
                (output / name).write_bytes(data)
        summary["sources"] = {name: {"path": str(paths[name]), "sha256": evaluator.digest(data)}
                              for name, data in frozen.items()}
        operator_prompts = load_operator_prompts(frozen["operator_prompts.json"], operator_prompt_name)
        summary["operator_prompt"] = {**summary["sources"]["operator_prompts.json"], "name": operator_prompts["name"]}
        feedback_mode = operator_prompts.get("feedback", "none")
        summary["operator_feedback"] = feedback_mode
        config = json.loads(frozen["config_snapshot.json"])
        generations = validate_generations(config["search"]["generations_after_initialization"]
                                           if generations is None else generations)
        summary["requested_generations"] = generations
        summary["rules"]["generations_after_initialization"] = generations
        parameters, attempts, retries = load_settings(config)
        children_per_request = operator_prompts.get("children_per_request", 1)
        if "generation_max_tokens" in operator_prompts:
            parameters = {**parameters, "max_tokens": operator_prompts["generation_max_tokens"]}
        summary.update(children_per_request=children_per_request, invalid_generation_groups=0)
        if children_per_request == 2:
            summary["rules"].update(offspring_per_generation="2 * population_size",
                                    candidate_evaluation_budget="(2 * generations_after_initialization + 1) * population_size")
        selection_mode = normalize_parent_selection(
            config["search"]["parent_selection"] if parent_selection is None else parent_selection)
        summary["rules"]["parent_selection"] = selection_mode
        if selection_mode == "tour":
            summary["rules"].update(tournament_size=2, tournament_tie_break="first_drawn")
        print(f"Parent selection: {selection_mode}", flush=True)
        patience = config["search"]["early_stopping_patience"]
        summary["rules"]["early_stopping_patience"] = patience
        if no_early_stopping:
            print("Early stopping: disabled for this run", flush=True)
        crossover_probability = config["search"]["crossover_probability"]
        mutation_probability = config["search"]["mutation_probability"]
        deduplication = config["search"]["deduplication"]
        summary["rules"].update(crossover_probability=crossover_probability,
                                mutation_probability=mutation_probability, deduplication=deduplication)
        entries = pipeline.load_prompts(frozen["prompts_snapshot.json"])
        summary["input_prompt_count"] = len(entries)
        if deduplication == "exact_text":
            entries = unique_instructions(entries)
        summary["initial_duplicates_removed"] = summary["input_prompt_count"] - len(entries)
        n = len(entries)
        if n < 2:
            raise ValueError("GA requires at least two initial prompts after applying deduplication")
        print(f"Initial population: {n}; deduplication={deduplication}; "
              f"initial duplicates removed={summary['initial_duplicates_removed']}", flush=True)
        summary.update(population_size=n, offspring_count=generations * n * children_per_request,
                       offspring_per_generation=n * children_per_request, generation_groups_per_generation=n,
                       candidate_evaluation_budget=(generations * children_per_request + 1) * n,
                       budget_interpretation="upper_bound_before_duplicate_skips",
                       deduplication_stage="before_classification",
                       deduplication_scope="all_prior_candidates_in_this_run",
                       generation_proposal_slots=generations * n,
                       generation_parameters=parameters, classification_parameters=evaluator.PARAMETERS)
        tokenizer = evaluator.load_tokenizer()
        task, template = (frozen[key].decode("utf-8") for key in ("task_template.txt", "generation_template.txt"))
        if save_details:
            (output / "instructions").mkdir()
        prepared_runs = {}
        common_inputs = None
        common_evaluation = None
        if not save_details:
            dataset = evaluator.load_data(paths["data_snapshot.tsv"], data=frozen["data_snapshot.tsv"])
            samples = dataset["selected"]
            model_path = evaluator.ROOT / "LLM/Qwen3.5-9B"
            common_inputs = {
                "sample_count": len(samples), "source_sha256": evaluator.digest(dataset["data"]),
                "context_tokens": evaluator.CONTEXT_TOKENS,
                **evaluator.task_metadata(label_mode, task_template_file=task_template_file),
                "class_counts": {label: sum(spec["label_mapping"][r["Label"]] == label for r in samples)
                                 for label in spec["labels"]},
                "evaluation_scope": dataset["scope"],
                "task_template_sha256": evaluator.digest(frozen["task_template.txt"]),
                "tokenizer_files_sha256": {name: evaluator.digest((model_path / name).read_bytes())
                                           for name in ("tokenizer.json", "tokenizer_config.json",
                                                        "chat_template.jinja", "config.json")},
            }

        def check_sources():
            # Compact mode uses the startup bytes throughout, not mutable disk inputs.
            if not save_details:
                return
            if any(path.read_bytes() != frozen[name] for name, path in paths.items()):
                raise ValueError("An input/configuration file changed during this run")

        def prepare(item):
            nonlocal common_inputs
            if not save_details:
                prepared_runs[item["id"]] = evaluator.prepare_candidate(
                    item["instruction"], samples, tokenizer, task_template=task, label_mode=label_mode)
                item["status"] = "prepared"
                return
            check_sources()
            (output / item["instruction_file"]).write_text(item["instruction"], encoding="utf-8")
            prepared = evaluator.prepare_run(output / item["instruction_file"],
                                              output / item["evaluation_directory"],
                                              tokenizer=tokenizer, data_file=paths["data_snapshot.tsv"], label_mode=label_mode,
                                              task_template_file=spec["task_template_path"])
            signature = pipeline.input_signature(prepared["manifest"])
            if common_inputs is not None and common_inputs != signature:
                raise ValueError("Candidate evaluation inputs changed")
            common_inputs = signature
            prepared["manifest"].update(purpose=PURPOSE, individual_id=item["id"],
                                        instruction_provenance=item["provenance"])
            evaluator.save_json(prepared["output"] / "manifest.json", prepared["manifest"])
            prepared_runs[item["id"]] = prepared
            item["status"] = "prepared"

        def service_check():
            if not save_details and "service" in summary:
                return
            models = json.loads(evaluator.http_call("/v1/models"))
            version = json.loads(evaluator.http_call("/version"))
            identity = pipeline.model_identity({"server_models": models, "model": evaluator.MODEL})
            signature = {"identity": identity, "version": version}
            if "service" in summary and summary["service"] != signature:
                raise ValueError("Served model identity or version changed")
            if identity.get("max_model_len") is not None and identity["max_model_len"] < evaluator.CONTEXT_TOKENS:
                raise ValueError("Server context is smaller than the configured context")
            summary["service"] = signature

        def evaluate(item):
            nonlocal common_evaluation
            summary["stage"] = f"evaluation:{item['id']}"
            check_sources()
            service_check()
            item["status"] = "running"
            prepared = prepared_runs[item["id"]]
            if save_details:
                evaluation = evaluator.evaluate_run(prepared)
                metrics = evaluation["metrics"]
                row, signature = pipeline.collect_evaluation(output, item, prepared)
                if common_evaluation is not None and signature != common_evaluation:
                    raise ValueError("Evaluation conditions changed")
                common_evaluation = signature
            else:
                def record_result(index, total, result):
                    summary["logged_classification_requests"] += 1
                    evaluator.print_prediction(index, total, result)

                print(f"Evaluating {item['id']} ({len(prepared['requests'])} posts)", flush=True)
                evaluation = evaluator.evaluate_candidate(prepared, on_result=record_result)
                metrics = evaluation["metrics"]
                row = {"prompt_id": item["id"],
                       **{key: metrics[key] for key in ("n", "macro_f1", "invalid_output_rate", "request_error_count",
                                                       "nonempty_reasoning_count", "length_limited_output_count")}}
                print(f"{item['id']}: macro-F1={row['macro_f1']:.4f}; "
                      f"invalid_output_rate={row['invalid_output_rate']:.4f}", flush=True)
            summary["results"].append(row)
            summary["candidate_evaluations"] += 1
            item.update(macro_f1=row["macro_f1"], status="complete_with_request_errors" if row["request_error_count"] else "complete")
            if row["request_error_count"]:
                raise RuntimeError(f"{item['id']} has request errors; score retained")
            if feedback_mode == "parent_metrics":
                item["performance"] = {key: metrics[key] for key in (
                    "macro_f1", "confusion_matrix")}
            # Retained individuals keep their measured score, not all request payloads in memory.
            prepared_runs.pop(item["id"])

        for index, entry in enumerate(entries):
            item = individual(f"parent_{index + 1:03d}", entry["instruction"], index,
                              source_id=entry["id"], provenance="provided_file_authorship_unspecified",
                              parent_ids=[], generation=0)
            summary["population"].append(item)
            prepare(item)
        parents = list(summary["population"])
        seen_instructions = {p["instruction"]: p["id"] for p in parents}
        summary.update(samples_per_prompt=common_inputs["sample_count"],
                       planned_classification_requests=summary["candidate_evaluation_budget"] * common_inputs["sample_count"],
                       evaluation_inputs=common_inputs)
        # Validate the operator before any HTTP; actual selected pairs are checked again below.
        if crossover_probability > 0 or mutation_probability > 0:
            build_request(task, template, parents[:2], parameters, tokenizer,
                          {"crossover": crossover_probability > 0, "mutation": mutation_probability > 0},
                          operator_prompts=operator_prompts, allow_unscored=True)
        if feedback_mode == "parent_metrics":
            summary["feedback_preparation"] = "Unscored placeholders only; complete feedback requests checked before each generation call."
        if prepare_only:
            summary.update(status="prepared", stage="prepared")
        else:
            for item in parents:
                evaluate(item)
            summary["generation_metrics"].append(generation_metrics(0, parents))
            previous_mean = summary["generation_metrics"][-1]["mean_macro_f1"]
            summary["retained_populations"].append({"generation": 0, "ids": [p["id"] for p in parents]})

            def record(event):
                event.update(generation_group=f"g{event['generation']:03d}_group_{event['slot']:03d}",
                             children_per_request=children_per_request)
                if save_details:
                    with (output / "generation_attempts.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                if event["outcome"] == "copied":
                    summary["copied_offspring"] += event.get("offspring_count", 1)
                else:
                    summary["generation_requests"] += 1
                    summary["proposal_attempts"] += event["transport_retry"] == 0
                    summary["transport_retries"] += event["transport_retry"] > 0
                    summary["invalid_generation_outputs"] += event["outcome"] == "invalid"
                    summary["generation_request_errors"] += event.get("request_error") is not None
                if event["outcome"] in ("invalid", "request_error", "interrupted"):
                    summary["last_generation_issue"] = {key: event.get(key) for key in (
                        "generation", "slot", "proposal_attempt", "transport_retry", "operations", "outcome",
                        "rejection_reason", "request_error", "finish_reason", "reasoning", "raw_output", "usage")}
                reason = event.get("rejection_reason") or event.get("request_error")
                unit = "group" if children_per_request == 2 else "child"
                print(f"G{event['generation']} {unit} {event['slot']}, attempt {event['proposal_attempt']}, "
                      f"retry {event['transport_retry']}: {event['outcome']}" + (f" — {reason}" if reason else ""), flush=True)

            for generation in range(1, generations + 1):
                selection = select_parents(parents, selection_seed(seed, generation), selection_mode)
                operators = select_operators(seed, generation, n, crossover_probability, mutation_probability)
                selection.update(generation=generation, operator_selection=operators)
                summary["selection"] = selection
                summary["selections"].append(selection)
                if save_details:
                    evaluator.save_json(output / "selection.json", {"generations": summary["selections"]})
                parent_lookup = {p["id"]: p for p in parents}
                children = []
                skipped_this_generation = 0
                invalid_this_generation = 0
                for slot, (parent_ids, operations) in enumerate(zip(selection["parent_ids"], operators["choices"]), start=1):
                    prefix = "" if generation == 1 else f"g{generation:03d}_"
                    identifiers = [f"{prefix}child_{(slot - 1) * children_per_request + index:03d}"
                                   for index in range(1, children_per_request + 1)]
                    group_id = f"g{generation:03d}_group_{slot:03d}"
                    summary["stage"] = f"generation:{group_id if children_per_request == 2 else identifiers[0]}"
                    check_sources()
                    chosen = [parent_lookup[i] for i in parent_ids]
                    if any(operations.values()):
                        service_check()
                        request, tokens = build_request(task, template, chosen, parameters, tokenizer, operations, operator_prompts=operator_prompts)
                        try:
                            child = generate_child(request, tokenizer, seed=seed, slot=slot, generation=generation,
                                                   attempts=attempts, retries=retries, record=record, operations=operations,
                                                   children_per_request=children_per_request)
                        except InvalidChildProposalsExhausted as exc:
                            summary["invalid_offspring_skipped"] += children_per_request
                            summary["invalid_generation_groups"] += 1
                            invalid_this_generation += children_per_request
                            for index, identifier in enumerate(identifiers, start=1):
                                summary["skipped_generation_slots"].append({
                                    "generation": generation, "slot": slot, "id": identifier,
                                    "generation_group": group_id, "sibling_index": index,
                                    "parent_ids": parent_ids, "operations": operations,
                                    "proposal_attempts": attempts, "reason": str(exc)})
                            print(f"G{generation} group {slot}: skipped {children_per_request} child slot(s) "
                                  f"after {attempts} invalid proposals; "
                                  "continuing search", flush=True)
                            continue
                        proposals = ([child] if children_per_request == 1 else
                                     [{**entry, "seed": child["seed"], "proposal_attempt": child["proposal_attempt"]}
                                      for entry in child["children"]])
                        provenance = "qwen_ga_offspring"
                    else:
                        instruction, instruction_tokens = evaluator.validate_instruction(chosen[0]["instruction"], tokenizer)
                        child = {"instruction": instruction, "instruction_tokens": instruction_tokens,
                                 "crossover": None, "crossover_tokens": 0, "seed": None, "proposal_attempt": 0}
                        proposals = [dict(child) for _ in range(children_per_request)]
                        tokens, provenance = 0, "ga_copy_first_parent"
                        record({"generation": generation, "slot": slot, "operations": operations,
                                "proposal_attempt": 0, "transport_retry": 0, "outcome": "copied", "request": None,
                                "offspring_count": children_per_request})
                    for index, (identifier, child) in enumerate(zip(identifiers, proposals), start=1):
                        item = individual(identifier, child.pop("instruction"), len(summary["population"]),
                                          parent_ids=parent_ids, provenance=provenance, generation=generation,
                                          generation_group=group_id, sibling_index=index,
                                          operations=operations, rendered_prompt_tokens=tokens, **child)
                        summary["population"].append(item)
                        if deduplication == "exact_text" and item["instruction"] in seen_instructions:
                            item.update(status="skipped_duplicate", duplicate_of=seen_instructions[item["instruction"]])
                            summary["duplicate_offspring_skipped"] += 1
                            skipped_this_generation += 1
                            print(f"{identifier}: duplicate of {item['duplicate_of']}; skipped classification", flush=True)
                            continue
                        seen_instructions[item["instruction"]] = item["id"]
                        children.append(item)
                        prepare(item)
                for item in children:
                    evaluate(item)
                check_sources()
                combined = parents + children
                parents = select_survivors(combined, n, deduplication)
                metrics = generation_metrics(generation, parents)
                metrics["duplicate_offspring_skipped"] = skipped_this_generation
                metrics["invalid_offspring_skipped"] = invalid_this_generation
                metrics["offspring_evaluated"] = len(children)
                # Ignore only floating-point accumulation noise, not plot rounding.
                unchanged = math.isclose(metrics["mean_macro_f1"], previous_mean, rel_tol=0.0, abs_tol=1e-12)
                summary["unchanged_mean_generations"] = summary["unchanged_mean_generations"] + 1 if unchanged else 0
                metrics["unchanged_mean_generations"] = summary["unchanged_mean_generations"]
                previous_mean = metrics["mean_macro_f1"]
                summary["generation_metrics"].append(metrics)
                print(f"G{generation} selection: skipped {skipped_this_generation} duplicate children; "
                      f"skipped {invalid_this_generation} invalid children; "
                      f"retained {n} individuals, {metrics['unique_prompt_count']} distinct prompts", flush=True)
                summary["retained_populations"].append({"generation": generation, "ids": [p["id"] for p in parents]})
                summary["completed_generations"] = generation
                summary["stage"] = f"completed_generation:{generation}"
                if not no_early_stopping and summary["unchanged_mean_generations"] >= patience and generation < generations:
                    summary.update(stopped_early=True, stop_reason="mean_macro_f1_unchanged")
                    print(f"Early stopping at G{generation}: retained-population mean macro-F1 "
                          f"unchanged for {patience} consecutive generations", flush=True)
                    break
            survivors = parents
            summary["next_population_ids"] = [p["id"] for p in survivors]
            if save_details:
                pending_path = output / ".next_population.json.tmp"
                evaluator.save_json(pending_path, {
                    "description": f"GA survivors after {summary['completed_generations']} generations; development selection, semantic review pending.",
                    "prompts": [{"id": p["id"], "instruction": p["instruction"]} for p in survivors]})
                pending_path.replace(output / "next_population.json")
            summary.update(status="complete", stage="complete")
            if summary["stop_reason"] is None:
                summary["stop_reason"] = "generation_limit"
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                       error=f"{type(exc).__name__}: {exc}")
    finally:
        if summary["status"] not in {"complete", "prepared"}:
            if save_details:
                (output / "next_population.json").unlink(missing_ok=True)
            summary.pop("next_population_ids", None)
            for item in summary["population"]:
                if item["status"] == "running":
                    item["status"] = summary["status"]
        if save_details:
            summary["logged_classification_requests"] = sum(
                len(p.read_text(encoding="utf-8").splitlines())
                for p in (output / "evaluations").glob("*/predictions.jsonl"))
            pipeline.save_summary(output, summary)
    print(f"GA status: {summary['status']}; evaluated {summary['candidate_evaluations']} individuals", flush=True)
    if summary["error"]:
        print(f"Error at {summary['stage']}: {summary['error']}", flush=True)
    return summary


def run_ga(output, *, prompts_file=None, data_file=DEFAULT_DATA,
           seed=11, generations=None, prepare_only=False, save_details=False, label_mode="three_class",
           parent_selection=None, operator_prompt_file=DEFAULT_OPERATOR_PROMPT, operator_prompt_name=None,
           task_template_file=None, no_early_stopping=False):
    """Write the compact report once on exit; default search uses memory only."""
    if generations is not None:
        validate_generations(generations)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    report = CompactReport(output)
    similarity = PromptSimilarityReport(output)
    arguments = dict(prompts_file=prompts_file, data_file=data_file, seed=seed, generations=generations,
                     prepare_only=prepare_only, save_details=save_details, label_mode=label_mode,
                     parent_selection=parent_selection, operator_prompt_file=operator_prompt_file,
                     operator_prompt_name=operator_prompt_name, task_template_file=task_template_file,
                     no_early_stopping=no_early_stopping)
    result = _run_search(output, **arguments)
    report(result)
    if result["status"] != "prepared":
        similarity(result)
    print(f"Results: {output}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-mode", choices=evaluator.LABEL_MODES, default="three_class")
    parser.add_argument("--task-template-file", type=Path,
                        help="Override the fixed classification task template (must match the label mode)")
    parser.add_argument("--prompts-file", type=Path, help="Default: humanPrompt_Bi.json for binary; humanPrompt.json otherwise")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA, help="Dev TSV; default: fixed dev_100.tsv")
    parser.add_argument("--operator-prompt-file", type=Path, default=DEFAULT_OPERATOR_PROMPT,
                        help="JSON file with both, crossover_only, and mutation_only instructions")
    parser.add_argument("--operator-prompt-name", help="Select a named operator prompt in the JSON collection (default: baseline)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--parent-selection", choices=PARENT_SELECTION_MODES,
                        help="Override search.parent_selection in the config (default: wheel)")
    parser.add_argument("--generations", type=positive_integer,
                        help="Positive number of evolution steps after G0; defaults to search.generations_after_initialization")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable plateau early stopping for this run; errors and interrupts still stop the search")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing output is never overwritten")
    parser.add_argument("--prepare-only", action="store_true", help="Validate inputs and requests without HTTP calls")
    parser.add_argument("--save-details", action="store_true", help="Also keep snapshots, raw responses and all individual evaluations")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.label_mode == "binary":
        stamp = "binary_" + stamp
    output = args.output_dir or ROOT / "data/processed/depsign/ga_runs" / stamp
    result = run_ga(output, prompts_file=args.prompts_file, data_file=args.data_file,
                    seed=args.seed, generations=args.generations, prepare_only=args.prepare_only,
                    save_details=args.save_details, label_mode=args.label_mode, parent_selection=args.parent_selection,
                    operator_prompt_file=args.operator_prompt_file, operator_prompt_name=args.operator_prompt_name,
                    task_template_file=args.task_template_file, no_early_stopping=args.no_early_stopping)
    if result["status"] not in {"complete", "prepared"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
