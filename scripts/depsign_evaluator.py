"""Shared DepSign candidate preparation, classification, and scoring.

Used by GA, fixed-file development evaluation, and data preparation.
The separation of candidate evaluation from search follows EvoPrompt's design:
https://github.com/beeevita/EvoPrompt/blob/main/evaluator.py
This module uses the project's own request, parsing, and scoring policies.
"""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DEV = Path("data/processed/depsign/dev_v1/dev_clean.tsv")
DEV_PURPOSE = "development_evaluation_not_held_out_test"
TASK_TEMPLATE_PATH = ROOT / "prompts/depsign_task_v1.txt"
LABEL_MAP = {"not depression": "not_depression", "moderate": "moderate", "severe": "severe"}
LABELS = list(LABEL_MAP.values())
LABEL_MODES = ("three_class", "binary")


def task_spec(label_mode="three_class", *, task_template_file=None):
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unknown label mode: {label_mode}")
    mapping = (dict(LABEL_MAP) if label_mode == "three_class" else
               {"not depression": "not_depression", "moderate": "depression", "severe": "depression"})
    template_path = (TASK_TEMPLATE_PATH if label_mode == "three_class" else ROOT / "prompts/depsign_task_Bi.txt")
    if task_template_file is not None:
        template_path = Path(task_template_file).resolve()
    if label_mode != "three_class" and template_path == (ROOT / "prompts/depsign_task_v2.txt").resolve():
        raise ValueError("depsign_task_v2.txt requires --label-mode three_class")
    return {"label_mode": label_mode, "label_mapping": mapping,
            "labels": list(dict.fromkeys(mapping.values())),
            "task_template_path": template_path}


def task_metadata(label_mode="three_class", *, task_template_file=None):
    spec = task_spec(label_mode, task_template_file=task_template_file)
    return {**spec, "task_template_path": str(spec["task_template_path"])}


BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen35-9b"
try:
    CONTEXT_TOKENS = int(os.environ.get("PROMPT_SEARCH_CONTEXT_TOKENS", "2048"))
except ValueError as exc:
    raise ValueError("PROMPT_SEARCH_CONTEXT_TOKENS must be a positive integer") from exc
if CONTEXT_TOKENS < 1:
    raise ValueError("PROMPT_SEARCH_CONTEXT_TOKENS must be a positive integer")
TEXT_TOKENS = 1500
MAX_INSTRUCTION_TOKENS = 256
PARAMETERS = {
    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "max_tokens": 16, "seed": 0,
    "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}


def parse_label(content, label_mode="three_class"):
    if isinstance(content, str) and content.strip() in task_spec(label_mode)["labels"]:
        return content.strip()
    return "INVALID"


def calculate_metrics(records, label_mode="three_class"):
    labels = task_spec(label_mode)["labels"]
    if not records:
        raise ValueError("Cannot score zero records")
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    columns = labels + ["INVALID"]
    for record in records:
        matrix[labels.index(record["gold"])][columns.index(record["prediction"])] += 1
    per_class = {}
    for i, label in enumerate(labels):
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
        "n": len(records), "macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels),
        "per_class": per_class,
        "invalid_output_count": invalid, "invalid_output_rate": invalid / len(records),
        "valid_output_coverage": 1 - invalid / len(records),
        "request_error_count": sum(r.get("request_error") is not None for r in records),
        "nonempty_reasoning_count": sum(bool(r.get("reasoning")) for r in records),
        "length_limited_output_count": sum(r.get("finish_reason") == "length" for r in records),
        "confusion_matrix": {"rows": labels, "columns": columns, "values": matrix},
        "invalid_policy": f"All records retained; failed/invalid predictions count as errors; macro over {len(labels)} real labels only",
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
        raise ValueError(f"Prompt plus output exceeds context budget: {count} + {PARAMETERS['max_tokens']} > {CONTEXT_TOKENS}; "
                         "set PROMPT_SEARCH_CONTEXT_TOKENS consistently for the client and vLLM server")
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


def prepare_candidate(instruction, samples, tokenizer, *, task_template=None, label_mode="three_class"):
    """Prepare every request before inference; only the instruction slot changes.

    Samples use the cleaned TSV schema: PID, Text data, Label. Outer instruction
    whitespace is stripped once; the returned instruction is the exact text used.
    No model requests are made by this function.
    """
    spec = task_spec(label_mode)
    instruction, instruction_tokens = validate_instruction(instruction, tokenizer)
    if not samples or len({row["PID"] for row in samples}) != len(samples):
        raise ValueError("Samples must be nonempty and have unique PIDs")
    if any(row["Label"] not in LABEL_MAP for row in samples):
        raise ValueError("Unexpected sample label")
    if task_template is None:
        task_template = spec["task_template_path"].read_text(encoding="utf-8")
    if task_template.count("{{INSTRUCTION}}") != 1 or task_template.count("{{POST_JSON}}") != 1:
        raise ValueError("Expected exactly one instruction and one post placeholder")
    template = task_template.replace("{{INSTRUCTION}}", instruction)
    prepared = []
    for row in samples:
        request, details = build_request(row["Text data"], template, tokenizer)
        prepared.append({"PID": row["PID"], "gold": spec["label_mapping"][row["Label"]], "label_mode": label_mode, "request": request, **details})
    return {
        "label_mode": label_mode, "instruction": instruction, "instruction_tokens": instruction_tokens,
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
        result["prediction"] = parse_label(result["raw_output"], prepared.get("label_mode", "three_class"))
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
    return {"metrics": calculate_metrics(results, candidate.get("label_mode", "three_class")), "predictions": results}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT / "LLM/Qwen3.5-9B", local_files_only=True)


def load_data(data_file, *, data=None):
    """Validate an explicit dev TSV and use every record in file order."""
    cleaned_dev = (ROOT / CLEAN_DEV).resolve()
    source = Path(data_file).resolve()
    if data is None:
        data = source.read_bytes()
    is_cleaned_dev = source == cleaned_dev
    if is_cleaned_dev:
        cleaning = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
        if digest(data) != cleaning["output_sha256"][source.name]:
            raise ValueError("Source differs from cleaned dev manifest")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t", strict=True)
    if reader.fieldnames != ["PID", "Text data", "Label"]:
        raise ValueError("Unexpected source schema")
    rows = list(reader)
    if not rows:
        raise ValueError("Data file must contain at least one record")
    for index, row in enumerate(rows, start=1):
        if (set(row) != set(reader.fieldnames)
                or any(not isinstance(row[key], str) or not row[key].strip() for key in reader.fieldnames)):
            raise ValueError(f"Malformed or empty fields in data record {index}")
        if row["Label"] not in LABEL_MAP:
            raise ValueError(f"Unexpected label in data record {index}: {row['Label']!r}")
    if len({row["PID"] for row in rows}) != len(rows) or len({row["Text data"] for row in rows}) != len(rows):
        raise ValueError("Data file must have unique PIDs and texts; clean duplicates/conflicts first")
    selected = rows
    scope = "full_cleaned_dev" if is_cleaned_dev else "provided_file_all_rows"
    return {
        "source": source, "data": data, "rows": rows, "selected": selected,
        "scope": scope, "is_cleaned_dev": is_cleaned_dev,
    }


def prepare_run(instruction_path, output=None, *, data_file, tokenizer=None, label_mode="three_class",
                task_template_file=None):
    """Save one fixed instruction and all selected requests without model calls."""
    dataset = load_data(data_file)
    source, data, rows, selected = (dataset[key] for key in ("source", "data", "rows", "selected"))
    instruction_path = Path(instruction_path).resolve()
    instruction_data = instruction_path.read_bytes()
    model_path = ROOT / "LLM/Qwen3.5-9B"
    if tokenizer is None:
        tokenizer = load_tokenizer()
    spec = task_spec(label_mode, task_template_file=task_template_file)
    candidate = prepare_candidate(instruction_data.decode("utf-8"), selected, tokenizer, label_mode=label_mode,
                                  task_template=spec["task_template_path"].read_text(encoding="utf-8"))
    prepared = candidate["requests"]
    template = candidate["template"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output).resolve() if output is not None else ROOT / "data/processed/depsign/evaluation_runs" / stamp
    output.mkdir(parents=True, exist_ok=False)
    with (output / "sample.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["PID", "Text data", "Label"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    (output / "prompt.txt").write_text(template, encoding="utf-8")
    (output / "instruction.txt").write_text(candidate["instruction"], encoding="utf-8")
    (output / "task_template.txt").write_text(candidate["task_template"], encoding="utf-8")
    (output / "requests.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in prepared), encoding="utf-8"
    )
    source_indices = {row["PID"]: i for i, row in enumerate(rows, start=1)}
    counts = Counter(item["gold"] for item in prepared)
    manifest = {
        "status": "prepared", "purpose": DEV_PURPOSE,
        "evaluation_scope": dataset["scope"],
        "source": str(source), "source_sha256": digest(data),
        "source_record_count": len(rows), "sample_count": len(selected),
        "cleaning_manifest_verified": dataset["is_cleaned_dev"],
        "sample_sha256": digest((output / "sample.tsv").read_bytes()),
        "prompt_sha256": digest(template.encode("utf-8")), "prompt_reference": "kayalvizhi2022dataset, section 3.2; project paraphrase",
        "instruction_source": str(instruction_path), "instruction_source_sha256": digest(instruction_data),
        "instruction_sha256": digest(candidate["instruction"].encode("utf-8")),
        "instruction_tokens": candidate["instruction_tokens"],
        "instruction_whitespace_policy": "strip outer whitespace only; archive exact text used in instruction.txt",
        "instruction_provenance": "user_supplied_file_authorship_unspecified",
        "task_template_source": str(spec["task_template_path"]),
        "task_template_sha256": digest(candidate["task_template"].encode("utf-8")),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "evaluator_sha256": digest(Path(__file__).read_bytes()),
        "label_mode": label_mode, "label_mapping": spec["label_mapping"], "sampling_seed": None,
        "sampling": "all rows in source file order; no sampling or shuffle",
        "selected_records": [{"PID": r["PID"], "source_record": source_indices[r["PID"]], "raw_label": r["Label"]} for r in selected],
        "class_counts": {label: counts[label] for label in spec["labels"]},
        "base_url": BASE_URL, "model": MODEL, "parameters": PARAMETERS,
        "context_tokens": CONTEXT_TOKENS, "text_token_limit": TEXT_TOKENS,
        "max_instruction_tokens": MAX_INSTRUCTION_TOKENS,
        "truncation": "Keep first 1500 tokenizer tokens only when longer; preserve shorter text verbatim",
        "truncated_count": sum(item["truncated"] for item in prepared),
        "client_versions": {name: version(name) for name in ["transformers", "torch", "vllm"]},
        "tokenizer_files_sha256": {name: digest((model_path / name).read_bytes()) for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"]},
        "transport_retries": 0,
        "train_or_test_accessed": False if dataset["is_cleaned_dev"] else None,
    }
    save_json(output / "manifest.json", manifest)
    print(f"Output directory: {output}", flush=True)
    print(f"Instruction file: {instruction_path} ({candidate['instruction_tokens']} tokens)", flush=True)
    print(f"Prepared {len(selected)} records ({dataset['scope']}); classes: {manifest['class_counts']}; truncated: {manifest['truncated_count']}", flush=True)
    return {"output": output, "manifest": manifest, "candidate": candidate}


def print_prediction(index, total, result):
    """Print progress and invalid-output diagnostics without writing files."""
    print(f"{index}/{total} {result['PID']}: gold={result['gold']}, prediction={result['prediction']}", flush=True)
    if result["prediction"] == "INVALID":
        diagnostic = {key: result.get(key) for key in (
            "raw_output", "finish_reason", "request_error", "reasoning")}
        print("  INVALID details: " + json.dumps(diagnostic, ensure_ascii=False), flush=True)


def evaluate_run(prepared_run):
    """Run previously prepared requests and preserve all responses and failures."""
    output = prepared_run["output"]
    manifest = prepared_run["manifest"]
    candidate = prepared_run["candidate"]
    # Check service before classifying. An unavailable service is not a measured result.
    models = json.loads(http_call("/v1/models"))
    if MODEL not in {item["id"] for item in models["data"]}:
        raise ValueError(f"Server does not expose {MODEL}")
    manifest.update(status="running", server_models=models, server_version=json.loads(http_call("/version")))
    save_json(output / "manifest.json", manifest)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        def record_result(index, total, result):
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print_prediction(index, total, result)

        evaluation = evaluate_candidate(candidate, on_result=record_result)
    metrics = evaluation["metrics"]
    metrics["purpose"] = manifest["purpose"]
    metrics["evaluation_scope"] = manifest["evaluation_scope"]
    save_json(output / "metrics.json", metrics)
    manifest["status"] = "complete_with_request_errors" if metrics["request_error_count"] else "complete"
    save_json(output / "manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return evaluation
