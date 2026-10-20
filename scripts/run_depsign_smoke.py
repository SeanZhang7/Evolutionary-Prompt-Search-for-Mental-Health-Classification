"""Evaluate one instruction on nine dev samples, or all rows of --data-file."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import random

if __package__:
    from . import depsign_evaluator as evaluator
else:
    import depsign_evaluator as evaluator

ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = evaluator.LABEL_MAP
DEFAULT_INSTRUCTION = ROOT / "prompts/depsign_smoke_instruction_v1.txt"
SAMPLING_SEED = 1729
CLEAN_DEV = Path("data/processed/depsign/dev_v1/dev_clean.tsv")
SMOKE_PURPOSE = "pipeline_smoke_test_only_not_method_evaluation"
DEV_PURPOSE = "development_evaluation_not_held_out_test"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_rows(rows):
    if len({r["PID"] for r in rows}) != len(rows) or len({r["Text data"] for r in rows}) != len(rows):
        raise ValueError("Cleaned dev must have unique PIDs and texts")
    if set(r["Label"] for r in rows) != set(LABEL_MAP):
        raise ValueError("Unexpected source labels")
    rng = random.Random(SAMPLING_SEED)
    selected = []
    for raw_label in LABEL_MAP:
        selected.extend(rng.sample([r for r in rows if r["Label"] == raw_label], 3))
    rng.shuffle(selected)
    return selected


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT / "LLM/Qwen3.5-9B", local_files_only=True)


def load_data(data_file=None):
    """Use all supplied rows in file order; no argument keeps the nine-row smoke."""
    cleaned_dev = (ROOT / CLEAN_DEV).resolve()
    source = Path(data_file).resolve() if data_file is not None else cleaned_dev
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
    smoke = data_file is None
    selected = sample_rows(rows) if smoke else rows
    scope = "fixed_nine_dev_smoke" if smoke else ("full_cleaned_dev" if is_cleaned_dev else "provided_file_all_rows")
    return {
        "source": source, "data": data, "rows": rows, "selected": selected,
        "scope": scope, "smoke": smoke, "is_cleaned_dev": is_cleaned_dev,
    }


def prepare_run(instruction_path, output=None, *, tokenizer=None, data_file=None):
    """Save one fixed instruction and all selected requests without model calls."""
    dataset = load_data(data_file)
    source, data, rows, selected = (dataset[key] for key in ("source", "data", "rows", "selected"))
    instruction_path = Path(instruction_path).resolve()
    instruction_data = instruction_path.read_bytes()
    model_path = ROOT / "LLM/Qwen3.5-9B"
    if tokenizer is None:
        tokenizer = load_tokenizer()
    candidate = evaluator.prepare_candidate(instruction_data.decode("utf-8"), selected, tokenizer)
    prepared = candidate["requests"]
    template = candidate["template"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output).resolve() if output is not None else ROOT / "data/processed/depsign/smoke_runs" / stamp
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
        "status": "prepared", "purpose": SMOKE_PURPOSE if dataset["smoke"] else DEV_PURPOSE,
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
        "instruction_provenance": "assistant_written_smoke_only" if instruction_path == DEFAULT_INSTRUCTION.resolve() else "user_supplied_file_authorship_unspecified",
        "task_template_source": str(evaluator.TASK_TEMPLATE_PATH.relative_to(ROOT)),
        "task_template_sha256": digest(candidate["task_template"].encode("utf-8")),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "evaluator_sha256": digest(Path(evaluator.__file__).read_bytes()),
        "label_mapping": LABEL_MAP, "sampling_seed": SAMPLING_SEED if dataset["smoke"] else None,
        "sampling": ("3 per class in label_mapping order, random.Random.sample in source order, then shuffle"
                     if dataset["smoke"] else "all rows in source file order; no sampling or shuffle"),
        "selected_records": [{"PID": r["PID"], "source_record": source_indices[r["PID"]], "raw_label": r["Label"]} for r in selected],
        "class_counts": {label: counts[label] for label in LABEL_MAP.values()},
        "base_url": evaluator.BASE_URL, "model": evaluator.MODEL, "parameters": evaluator.PARAMETERS,
        "context_tokens": evaluator.CONTEXT_TOKENS, "text_token_limit": evaluator.TEXT_TOKENS,
        "max_instruction_tokens": evaluator.MAX_INSTRUCTION_TOKENS,
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


def evaluate_run(prepared_run):
    """Run previously prepared requests and preserve all responses and failures."""
    output = prepared_run["output"]
    manifest = prepared_run["manifest"]
    candidate = prepared_run["candidate"]
    # Check service before classifying. An unavailable service is not a measured result.
    models = json.loads(evaluator.http_call("/v1/models"))
    if evaluator.MODEL not in {item["id"] for item in models["data"]}:
        raise ValueError(f"Server does not expose {evaluator.MODEL}")
    manifest.update(status="running", server_models=models, server_version=json.loads(evaluator.http_call("/version")))
    save_json(output / "manifest.json", manifest)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        def record_result(index, total, result):
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index}/{total} {result['PID']}: gold={result['gold']}, prediction={result['prediction']}", flush=True)

        evaluation = evaluator.evaluate_candidate(candidate, on_result=record_result)
    metrics = evaluation["metrics"]
    metrics["purpose"] = manifest["purpose"]
    metrics["evaluation_scope"] = manifest["evaluation_scope"]
    save_json(output / "metrics.json", metrics)
    manifest["status"] = "complete_with_request_errors" if metrics["request_error_count"] else "complete"
    save_json(output / "manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and check inputs without contacting vLLM")
    parser.add_argument("--instruction-file", type=Path, default=DEFAULT_INSTRUCTION,
                        help="UTF-8 file containing only a candidate instruction; default: assistant-written smoke instruction")
    parser.add_argument("--data-file", type=Path,
                        help="Evaluate ALL rows of this dev TSV (PID, Text data, Label); omitted: fixed nine-row smoke")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    prepared = prepare_run(args.instruction_file, args.output_dir, data_file=args.data_file)
    if args.prepare_only:
        return
    result = evaluate_run(prepared)
    if result["metrics"]["request_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
