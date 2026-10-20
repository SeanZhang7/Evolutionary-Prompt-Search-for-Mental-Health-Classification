"""Generate two ordinary instruction candidates using the existing Qwen service.

This is a generation smoke test, with no dataset access or candidate scoring.
LLM-generated initialization follows the EvoPrompt approach (Guo et al., 2024);
the generation template and validation choices here belong to this project.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import unicodedata
from urllib.error import HTTPError, URLError

if __package__:
    from . import depsign_evaluator as evaluator
else:
    import depsign_evaluator as evaluator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment_v1.json"
GENERATION_TEMPLATE_PATH = ROOT / "prompts/generate_ordinary_v1.txt"
CONDITION = "ordinary_generation_smoke"
OPERATOR = "initialization"
PURPOSE = "candidate_generation_smoke_only_not_method_evaluation"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request_seed(search_seed, slot, attempt):
    key = json.dumps([search_seed, CONDITION, slot, attempt, OPERATOR],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int(digest(key)[:8], 16) % (2 ** 31)


def instruction_id(instruction):
    normalized = " ".join(unicodedata.normalize("NFC", instruction).split())
    return digest(normalized.encode("utf-8"))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_instruction(content, tokenizer):
    if not isinstance(content, str):
        raise ValueError("Generation content must be text")
    value = json.loads(content, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != {"instruction"}:
        raise ValueError("Expected one JSON object with only an instruction field")
    return evaluator.validate_instruction(value["instruction"], tokenizer)


def load_settings(config):
    model = config["model"]
    if model["checkpoint"] != "Qwen/Qwen3.5-9B" or not model["same_checkpoint_for_both_roles"]:
        raise ValueError("This runner requires the same Qwen3.5-9B checkpoint for both roles")
    parameters = dict(model["generation"])
    if parameters.pop("decoding") != "sampling":
        raise ValueError("Generation must use sampling")
    if parameters.pop("seed_policy") != "sha256_search_seed_condition_slot_attempt_operator":
        raise ValueError("Unsupported generation seed policy")
    expected = set(evaluator.PARAMETERS) - {"seed"}
    if set(parameters) != expected or parameters["temperature"] <= 0:
        raise ValueError("Unexpected generation parameters")
    if parameters["chat_template_kwargs"] != {"enable_thinking": False}:
        raise ValueError("This generation smoke test requires thinking disabled")
    search = config["search"]
    if search["max_instruction_tokens"] != evaluator.MAX_INSTRUCTION_TOKENS:
        raise ValueError("Generator and evaluator instruction limits differ")
    attempts = search["max_proposal_attempts_per_slot"]
    retries = search["max_transport_retries_per_request"]
    if type(attempts) is not int or attempts < 1 or type(retries) is not int or retries < 0:
        raise ValueError("Invalid attempt or transport retry limits")
    return parameters, attempts, retries


def build_generation_request(task_template, generation_template, parameters, tokenizer):
    if task_template.count("{{INSTRUCTION}}") != 1 or task_template.count("{{POST_JSON}}") != 1:
        raise ValueError("Unexpected fixed task template")
    for marker in ("{{TASK_TEMPLATE_JSON}}", "{{MAX_INSTRUCTION_TOKENS}}"):
        if generation_template.count(marker) != 1:
            raise ValueError(f"Expected exactly one {marker}")
    prompt = generation_template.replace("{{MAX_INSTRUCTION_TOKENS}}", str(evaluator.MAX_INSTRUCTION_TOKENS))
    prompt = prompt.replace("{{TASK_TEMPLATE_JSON}}", json.dumps(task_template, ensure_ascii=False))
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    if tokens + parameters["max_tokens"] > evaluator.CONTEXT_TOKENS:
        raise ValueError(f"Generation prompt plus output exceeds context budget: {tokens} + {parameters['max_tokens']}")
    return {"model": evaluator.MODEL, "messages": messages, **parameters}, tokens


def generate_once(request):
    result = {
        "raw_http_response": None, "raw_output": None, "request_error": None,
        "reasoning": None, "finish_reason": None, "usage": None, "retryable": False,
    }
    started = time.monotonic()
    try:
        result["raw_http_response"] = evaluator.http_call("/v1/chat/completions", request)
        response = json.loads(result["raw_http_response"])
        choice = response["choices"][0]
        message = choice["message"]
        result.update(raw_output=message.get("content"),
                      reasoning=message.get("reasoning") or message.get("reasoning_content"),
                      finish_reason=choice.get("finish_reason"), usage=response.get("usage"))
    except HTTPError as exc:
        result.update(raw_http_response=exc.read().decode("utf-8", errors="replace"),
                      request_error=f"HTTP {exc.code}", retryable=exc.code == 429 or 500 <= exc.code < 600)
    except (URLError, TimeoutError, OSError) as exc:
        result.update(request_error=f"{type(exc).__name__}: {exc}", retryable=True)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        result["request_error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def generate_candidates(base_request, tokenizer, *, count, seed, max_attempts, max_retries,
                        on_attempt=None, on_candidate=None):
    """Independent proposals; prior outputs and scores never enter later prompts."""
    if count < 1 or max_attempts < 1 or max_retries < 0:
        raise ValueError("Invalid generation budget")
    candidates, events, seen = [], [], set()
    for slot in range(1, count + 1):
        accepted = None
        for attempt in range(1, max_attempts + 1):
            request = {**base_request, "seed": request_seed(seed, slot, attempt)}
            for retry in range(max_retries + 1):
                event = {"slot": slot, "proposal_attempt": attempt, "transport_retry": retry,
                         "request": request, **generate_once(request)}
                event["outcome"] = "request_error"
                if event["request_error"] is None:
                    try:
                        if event["finish_reason"] != "stop":
                            raise ValueError(f"Unexpected finish_reason: {event['finish_reason']}")
                        if event["reasoning"]:
                            raise ValueError("Nonempty reasoning despite thinking disabled")
                        instruction, tokens = parse_instruction(event["raw_output"], tokenizer)
                        identity = instruction_id(instruction)
                        event.update(instruction=instruction, instruction_id=identity, instruction_tokens=tokens)
                        if identity in seen:
                            event["outcome"] = "duplicate"
                        else:
                            event["outcome"] = "accepted"
                            accepted = {"slot": slot, "proposal_attempt": attempt, "seed": request["seed"],
                                        "instruction": instruction, "instruction_id": identity,
                                        "instruction_sha256": digest(instruction.encode("utf-8")),
                                        "instruction_tokens": tokens, "provenance": "qwen_generated_ordinary",
                                        "semantic_review": "pending", "purpose": PURPOSE}
                    except ValueError as exc:
                        event.update(outcome="invalid", rejection_reason=str(exc))
                events.append(event)
                if on_attempt is not None:
                    on_attempt(event)
                if accepted is not None or not event["retryable"] or retry == max_retries:
                    break
            if accepted is not None:
                seen.add(accepted["instruction_id"])
                candidates.append(accepted)
                if on_candidate is not None:
                    on_candidate(accepted)
                break
        if accepted is None:
            break  # Exhausted slot: stop rather than silently exceeding the cap.
    outcomes = Counter(event["outcome"] for event in events)
    summary = {
        "status": "complete" if len(candidates) == count else "incomplete", "purpose": PURPOSE,
        "requested_candidates": count, "accepted_candidates": len(candidates),
        "failed_slot": None if len(candidates) == count else len(candidates) + 1,
        "proposal_attempts": len({(e["slot"], e["proposal_attempt"]) for e in events}),
        "actual_generation_requests": len(events),
        "transport_retries": sum(e["transport_retry"] > 0 for e in events),
        "duplicate_count": outcomes["duplicate"], "invalid_count": outcomes["invalid"],
        "request_error_count": outcomes["request_error"],
        "candidate_evaluations": 0, "classification_requests": 0, "dataset_records_read": 0,
        "nonempty_reasoning_count": sum(bool(e["reasoning"]) for e in events),
        "length_limited_output_count": sum(e["finish_reason"] == "length" for e in events),
        "reported_usage": {key: sum(e["usage"].get(key, 0) or 0 for e in events if isinstance(e["usage"], dict))
                           for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "responses_with_usage": sum(isinstance(e["usage"], dict) for e in events),
        "elapsed_request_seconds": sum(e["elapsed_seconds"] for e in events),
        "semantic_review": "pending",
    }
    return {"candidates": candidates, "attempts": events, "summary": summary}


def positive_integer(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=positive_integer, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prepare-only", action="store_true", help="Check and save the generation request without contacting vLLM")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    parameters, max_attempts, max_retries = load_settings(config)
    model_path = ROOT / "LLM/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    task_template = evaluator.TASK_TEMPLATE_PATH.read_text(encoding="utf-8")
    generation_template = GENERATION_TEMPLATE_PATH.read_text(encoding="utf-8")
    base_request, prompt_tokens = build_generation_request(task_template, generation_template, parameters, tokenizer)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir.resolve() if args.output_dir else ROOT / "data/processed/depsign/generation_runs" / stamp
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config_snapshot.json", config)
    save_json(output / "base_request.json", base_request)
    (output / "task_template.txt").write_text(task_template, encoding="utf-8")
    (output / "generation_template.txt").write_text(generation_template, encoding="utf-8")
    manifest = {
        "status": "prepared", "purpose": PURPOSE, "requested_candidates": args.count,
        "condition": CONDITION, "operator": OPERATOR, "search_seed": args.seed,
        "slot_and_attempt_numbering": "one_based", "seed_policy": config["model"]["generation"]["seed_policy"],
        "base_url": evaluator.BASE_URL, "model": evaluator.MODEL, "parameters": parameters,
        "context_tokens": evaluator.CONTEXT_TOKENS, "rendered_prompt_tokens": prompt_tokens,
        "max_instruction_tokens": evaluator.MAX_INSTRUCTION_TOKENS,
        "max_proposal_attempts_per_slot": max_attempts, "max_transport_retries_per_request": max_retries,
        "config_sha256": digest(config_bytes), "script_sha256": digest(Path(__file__).read_bytes()),
        "evaluator_sha256": digest(Path(evaluator.__file__).read_bytes()),
        "task_template_sha256": digest(task_template.encode("utf-8")),
        "generation_template_sha256": digest(generation_template.encode("utf-8")),
        "deduplication": "sha256_nfc_whitespace_collapsed_case_preserved",
        "instruction_whitespace_policy": "strip outer whitespace only; save exact text used",
        "generation_constraints": "structural checks only; semantic review pending",
        "dataset_records_read": 0, "candidate_evaluations": 0,
        "client_versions": {name: version(name) for name in ("transformers", "torch", "vllm")},
        "tokenizer_files_sha256": {name: digest((model_path / name).read_bytes())
                                   for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")},
    }
    save_json(output / "manifest.json", manifest)
    print(f"Output directory: {output}", flush=True)
    print(f"Prepared generation request: {prompt_tokens} prompt tokens + {parameters['max_tokens']} maximum output tokens", flush=True)
    if args.prepare_only:
        return
    try:
        models = json.loads(evaluator.http_call("/v1/models"))
        if evaluator.MODEL not in {item["id"] for item in models["data"]}:
            raise ValueError(f"Server does not expose {evaluator.MODEL}")
        server_version = json.loads(evaluator.http_call("/version"))
    except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
        manifest.update(status="preflight_failed", error=f"{type(exc).__name__}: {exc}")
        save_json(output / "manifest.json", manifest)
        raise SystemExit(f"Service preflight failed: {exc}") from exc
    manifest.update(status="running", server_models=models, server_version=server_version)
    save_json(output / "manifest.json", manifest)
    with (output / "attempts.jsonl").open("w", encoding="utf-8") as attempts_file, \
            (output / "candidates.jsonl").open("w", encoding="utf-8") as candidates_file:
        def record_attempt(event):
            attempts_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            attempts_file.flush()
            print(f"Slot {event['slot']}, attempt {event['proposal_attempt']}, retry {event['transport_retry']}: {event['outcome']}", flush=True)

        def record_candidate(candidate):
            path = output / f"candidate_{candidate['slot']:02d}.txt"
            path.write_text(candidate["instruction"], encoding="utf-8")
            candidates_file.write(json.dumps({**candidate, "file": path.name}, ensure_ascii=False) + "\n")
            candidates_file.flush()
            print(f"Saved candidate: {path}\n{candidate['instruction']}", flush=True)

        result = generate_candidates(base_request, tokenizer, count=args.count, seed=args.seed,
                                     max_attempts=max_attempts, max_retries=max_retries,
                                     on_attempt=record_attempt, on_candidate=record_candidate)
    save_json(output / "summary.json", result["summary"])
    manifest["status"] = result["summary"]["status"]
    save_json(output / "manifest.json", manifest)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if result["summary"]["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
