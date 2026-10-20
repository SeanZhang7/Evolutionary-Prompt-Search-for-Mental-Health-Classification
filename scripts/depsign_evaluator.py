"""Shared DepSign candidate preparation, classification, and scoring.

Used by the nine-record smoke runner and fixed-file development-set evaluation.
The separation of candidate evaluation from search follows EvoPrompt's design:
https://github.com/beeevita/EvoPrompt/blob/main/evaluator.py
This module uses the project's own request, parsing, and scoring policies.
"""

import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TASK_TEMPLATE_PATH = ROOT / "prompts/depsign_task_v1.txt"
LABEL_MAP = {"not depression": "not_depression", "moderate": "moderate", "severe": "severe"}
LABELS = list(LABEL_MAP.values())
BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen35-9b"
CONTEXT_TOKENS = 2048
TEXT_TOKENS = 1500
MAX_INSTRUCTION_TOKENS = 256
PARAMETERS = {
    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "max_tokens": 16, "seed": 0,
    "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}


def parse_label(content):
    if isinstance(content, str) and content.strip() in LABELS:
        return content.strip()
    return "INVALID"


def calculate_metrics(records):
    if not records:
        raise ValueError("Cannot score zero records")
    matrix = [[0] * (len(LABELS) + 1) for _ in LABELS]
    columns = LABELS + ["INVALID"]
    for record in records:
        matrix[LABELS.index(record["gold"])][columns.index(record["prediction"])] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp  # Includes INVALID predictions.
        per_class[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": sum(matrix[i]),
        }
    invalid = sum(r["prediction"] == "INVALID" for r in records)
    return {
        "n": len(records), "macro_f1": sum(v["f1"] for v in per_class.values()) / len(LABELS),
        "per_class": per_class,
        "invalid_output_count": invalid, "invalid_output_rate": invalid / len(records),
        "valid_output_coverage": 1 - invalid / len(records),
        "request_error_count": sum(r.get("request_error") is not None for r in records),
        "nonempty_reasoning_count": sum(bool(r.get("reasoning")) for r in records),
        "length_limited_output_count": sum(r.get("finish_reason") == "length" for r in records),
        "confusion_matrix": {"rows": LABELS, "columns": columns, "values": matrix},
        "invalid_policy": "All records retained; failed/invalid predictions count as errors; macro over three real labels only",
        "purpose": "candidate_evaluation_scope_defined_by_runner",
    }


def build_request(text, template, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=False)
    truncated = len(ids) > TEXT_TOKENS
    used = tokenizer.decode(ids[:TEXT_TOKENS], skip_special_tokens=False) if truncated else text
    prompt = template.replace("{{POST_JSON}}", json.dumps(used, ensure_ascii=False))
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    count = len(tokenizer.encode(rendered, add_special_tokens=False))
    if count + PARAMETERS["max_tokens"] > CONTEXT_TOKENS:
        raise ValueError(f"Prompt exceeds fixed context budget: {count}")
    return {"model": MODEL, "messages": messages, **PARAMETERS}, {
        "original_text_tokens": len(ids), "input_text_used": used,
        "truncated": truncated, "rendered_prompt_tokens": count,
    }


def validate_instruction(instruction, tokenizer):
    """Apply shared structural checks; this does not verify semantic compliance."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Candidate instruction must be nonempty text")
    instruction = instruction.strip()
    if any(marker in instruction for marker in ("{{INSTRUCTION}}", "{{POST_JSON}}")):
        raise ValueError("Candidate instruction contains a reserved template placeholder")
    instruction_tokens = len(tokenizer.encode(instruction, add_special_tokens=False))
    if instruction_tokens > MAX_INSTRUCTION_TOKENS:
        raise ValueError(f"Candidate instruction exceeds {MAX_INSTRUCTION_TOKENS} tokens: {instruction_tokens}")
    return instruction, instruction_tokens


def prepare_candidate(instruction, samples, tokenizer):
    """Prepare every request before inference; only the instruction slot changes.

    Samples use the cleaned TSV schema: PID, Text data, Label. Outer instruction
    whitespace is stripped once; the returned instruction is the exact text used.
    No model requests are made by this function.
    """
    instruction, instruction_tokens = validate_instruction(instruction, tokenizer)
    if not samples or len({row["PID"] for row in samples}) != len(samples):
        raise ValueError("Samples must be nonempty and have unique PIDs")
    if any(row["Label"] not in LABEL_MAP for row in samples):
        raise ValueError("Unexpected sample label")
    task_template = TASK_TEMPLATE_PATH.read_text(encoding="utf-8")
    if task_template.count("{{INSTRUCTION}}") != 1 or task_template.count("{{POST_JSON}}") != 1:
        raise ValueError("Expected exactly one instruction and one post placeholder")
    template = task_template.replace("{{INSTRUCTION}}", instruction)
    prepared = []
    for row in samples:
        request, details = build_request(row["Text data"], template, tokenizer)
        prepared.append({"PID": row["PID"], "gold": LABEL_MAP[row["Label"]], "request": request, **details})
    return {
        "instruction": instruction, "instruction_tokens": instruction_tokens,
        "task_template": task_template, "template": template, "requests": prepared,
    }


def http_call(path, payload=None):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(BASE_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180 if payload is not None else 10) as response:
        return response.read().decode("utf-8")


def classify(prepared):
    result = {
        "PID": prepared["PID"], "gold": prepared["gold"], "prediction": "INVALID",
        "raw_http_response": None, "raw_output": None, "request_error": None,
        "reasoning": None, "finish_reason": None, "usage": None,
    }
    started = time.monotonic()
    try:
        result["raw_http_response"] = http_call("/v1/chat/completions", prepared["request"])
        response = json.loads(result["raw_http_response"])
        choice = response["choices"][0]
        message = choice["message"]
        result.update(
            raw_output=message.get("content"),
            reasoning=message.get("reasoning") or message.get("reasoning_content"),
            finish_reason=choice.get("finish_reason"), usage=response.get("usage"),
        )
        result["prediction"] = parse_label(result["raw_output"])
    except HTTPError as exc:
        result["raw_http_response"] = exc.read().decode("utf-8", errors="replace")
        result["request_error"] = f"HTTP {exc.code}"
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        result["request_error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def evaluate_candidate(candidate, *, on_result=None):
    """Classify a prepare_candidate result and return metrics plus raw predictions.

    on_result(index, total, result) can persist each response immediately. Failed
    requests stay in the score denominator. Service preflight belongs to the runner.
    """
    prepared = candidate["requests"]
    if not prepared:
        raise ValueError("Cannot evaluate zero samples")
    results = []
    for index, item in enumerate(prepared, start=1):
        result = classify(item)
        results.append(result)
        if on_result is not None:
            on_result(index, len(prepared), result)
    return {"metrics": calculate_metrics(results), "predictions": results}
