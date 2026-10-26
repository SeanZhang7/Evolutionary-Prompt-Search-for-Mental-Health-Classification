"""Evaluate all fixed instructions in one JSON file on the same dev TSV.

No prompt generation is performed. Task definitions, classification parameters,
and data remain fixed. The default uses the entire cleaned development set.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

if __package__:
    from . import depsign_evaluator as classifier
else:
    import depsign_evaluator as classifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "prompts/humanPrompt.json"
PURPOSE = "fixed_prompts_development_evaluation_not_held_out_test"
COLUMNS = ["prompt_id", "instruction_file", "instruction_sha256", "evaluation_directory",
           "n", "macro_f1", "invalid_output_rate", "request_error_count",
           "nonempty_reasoning_count", "length_limited_output_count", "truncated_count",
           "sample_sha256", "semantic_review"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_prompts(data):
    """Read every entry in file order. Identical text under distinct IDs is retained."""
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict) or "prompts" not in value or set(value) - {"prompts", "description"}:
        raise ValueError('Expected a JSON object containing "prompts" and an optional "description"')
    if "description" in value and not isinstance(value["description"], str):
        raise ValueError("description must be a string")
    entries = value["prompts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("prompts must be a nonempty list")
    prompts, seen_ids = [], set()
    for index, item in enumerate(entries, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "instruction"}:
            raise ValueError(f"Prompt {index} must contain exactly id and instruction")
        identifier = item["id"]
        if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identifier) is None:
            raise ValueError(f"Prompt {index}: id must be 1-64 letters, digits, underscores or hyphens, starting with a letter or digit")
        if identifier in seen_ids:
            raise ValueError(f"Duplicate prompt id: {identifier}")
        instruction = item["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Prompt {identifier}: instruction must be nonempty text")
        seen_ids.add(identifier)
        used = instruction.strip()
        prompts.append({
            "id": identifier, "instruction": used,
            "source_instruction_sha256": classifier.digest(instruction.encode("utf-8")),
            "instruction_sha256": classifier.digest(used.encode("utf-8")),
            "instruction_file": f"instructions/{identifier}.txt",
            "evaluation_directory": f"evaluations/{identifier}",
            "semantic_review": "not_assessed",
        })
    return prompts


def save_summary(output, summary):
    classifier.save_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(summary["results"])


def model_identity(manifest):
    matches = [item for item in manifest["server_models"]["data"] if item["id"] == manifest["model"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching served model")
    return {key: matches[0][key] for key in ("id", "root", "parent", "max_model_len", "owned_by")}


def input_signature(manifest):
    keys = ("source_sha256", "sample_sha256", "selected_records", "label_mapping", "class_counts",
            "source_record_count", "sample_count", "evaluation_scope", "sampling", "sampling_seed",
            "cleaning_manifest_verified",
            "task_template_sha256", "parameters", "base_url", "model", "tokenizer_files_sha256",
            "context_tokens", "text_token_limit", "truncation", "evaluator_sha256",
            "script_sha256", "client_versions")
    return {key: manifest[key] for key in keys}


def evaluation_signature(manifest):
    # Response-specific timestamps and permission IDs are not model identity.
    return {**input_signature(manifest), "served_model_identity": model_identity(manifest),
            "server_version": manifest["server_version"]}


def collect_evaluation(output, prompt, prepared):
    evaluation_dir = prepared["output"]
    manifest = read_json(evaluation_dir / "manifest.json")
    metrics = read_json(evaluation_dir / "metrics.json")
    if manifest["status"] not in {"complete", "complete_with_request_errors"}:
        raise ValueError("Evaluation has no completed score")
    instruction = (output / prompt["instruction_file"]).read_bytes()
    if (instruction != (evaluation_dir / "instruction.txt").read_bytes()
            or classifier.digest(instruction) != prompt["instruction_sha256"]
            or manifest["instruction_sha256"] != prompt["instruction_sha256"]):
        raise ValueError("File prompt and evaluated instruction contents differ")
    requests = prepared["candidate"]["requests"]
    counts = Counter(item["gold"] for item in requests)
    expected_counts = {label: counts[label] for label in classifier.LABEL_MAP.values()}
    if (metrics["n"] != len(requests) or manifest["sample_count"] != len(requests)
            or manifest["class_counts"] != expected_counts
            or {label: metrics["per_class"][label]["support"] for label in expected_counts} != expected_counts):
        raise ValueError("Evaluation size or class counts differ from the prepared data")
    row = {
        "prompt_id": prompt["id"], "instruction_file": prompt["instruction_file"],
        "instruction_sha256": prompt["instruction_sha256"],
        "evaluation_directory": prompt["evaluation_directory"],
        **{key: metrics[key] for key in ("n", "macro_f1", "invalid_output_rate", "request_error_count",
                                        "nonempty_reasoning_count", "length_limited_output_count")},
        "truncated_count": manifest["truncated_count"], "sample_sha256": manifest["sample_sha256"],
        "semantic_review": prompt["semantic_review"],
    }
    return row, evaluation_signature(manifest)


def run_pipeline(output, *, prompts_file=DEFAULT_PROMPTS, data_file=None, prepare_only=False):
    output = Path(output).resolve()
    prompts_file = Path(prompts_file).resolve()
    data_file = Path(data_file).resolve() if data_file is not None else (classifier.ROOT / classifier.CLEAN_DEV).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "running", "stage": "load_prompts", "purpose": PURPOSE,
        "prompts_source": str(prompts_file), "prompts_source_sha256": None,
        "prompts_snapshot": "prompts_snapshot.json", "requested_prompts": None,
        "prepared_prompts": 0, "scored_prompts": 0, "samples_per_prompt": None,
        "data_source": str(data_file), "data_source_sha256": None,
        "class_counts": None, "evaluation_scope": None, "planned_classification_requests": None,
        "generation_requests": 0, "logged_classification_requests": 0,
        "prompts": [], "results": [], "semantic_review": "not_assessed",
        "prompt_provenance": "provided_file_authorship_unspecified", "error": None,
        "instruction_whitespace_policy": "strip outer whitespace only; preserve internal text",
        "pipeline_script_sha256": classifier.digest(Path(__file__).read_bytes()),
    }
    save_summary(output, summary)
    print(f"Pipeline output directory: {output}", flush=True)
    try:
        data = prompts_file.read_bytes()
        (output / "prompts_snapshot.json").write_bytes(data)
        summary["prompts_source_sha256"] = classifier.digest(data)
        prompts = load_prompts(data)
        summary["requested_prompts"] = len(prompts)
        summary["prompts"] = [{**item, "status": "pending"} for item in prompts]
        save_summary(output, summary)
        print(f"Loaded {len(prompts)} fixed prompts from {prompts_file}; no prompt generation", flush=True)
        (output / "instructions").mkdir()
        for prompt in prompts:
            (output / prompt["instruction_file"]).write_text(prompt["instruction"], encoding="utf-8")

        # Check every actual prompt and freeze all requests before any model call.
        tokenizer = classifier.load_tokenizer()
        prepared_runs = []
        common_inputs = None
        for prompt, record in zip(prompts, summary["prompts"]):
            summary["stage"] = f"prepare:{prompt['id']}"
            save_summary(output, summary)
            prepared = classifier.prepare_run(output / prompt["instruction_file"],
                                               output / prompt["evaluation_directory"], tokenizer=tokenizer,
                                               data_file=data_file)
            signature = input_signature(prepared["manifest"])
            if common_inputs is not None and signature != common_inputs:
                raise ValueError("Evaluation sample or classifier configuration changed while preparing prompts")
            common_inputs = signature
            prepared_runs.append(prepared)
            record["status"] = "prepared"
            summary["prepared_prompts"] = len(prepared_runs)
            summary["sample_sha256"] = signature["sample_sha256"]
            summary.update(
                samples_per_prompt=signature["sample_count"],
                data_source_sha256=signature["source_sha256"],
                class_counts=signature["class_counts"],
                evaluation_scope=signature["evaluation_scope"],
                planned_classification_requests=len(prompts) * signature["sample_count"],
            )
            save_summary(output, summary)

        print(f"Prepared {len(prompts)} prompts × {summary['samples_per_prompt']} records = "
              f"{summary['planned_classification_requests']} classification requests", flush=True)
        if prepare_only:
            summary.update(status="prepared", stage="prepared")
        else:
            common_evaluation = None
            for prompt, record, prepared in zip(prompts, summary["prompts"], prepared_runs):
                summary["stage"] = f"evaluation:{prompt['id']}"
                record["status"] = "running"
                save_summary(output, summary)
                classifier.evaluate_run(prepared)
                row, signature = collect_evaluation(output, prompt, prepared)
                if input_signature(prepared["manifest"]) != common_inputs:
                    raise ValueError("Prepared evaluation inputs changed")
                if common_evaluation is not None and signature != common_evaluation:
                    raise ValueError("Evaluation sample or classifier configuration changed between prompts")
                common_evaluation = signature
                summary["results"].append(row)
                summary["scored_prompts"] = len(summary["results"])
                record["status"] = "complete_with_request_errors" if row["request_error_count"] else "complete"
                save_summary(output, summary)
                if row["request_error_count"]:
                    raise RuntimeError(f"{prompt['id']} completed with {row['request_error_count']} request errors; score retained")
            summary.update(status="complete", stage="complete")
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                       error=f"{type(exc).__name__}: {exc}")
        if ":" in summary["stage"]:
            failed_id = summary["stage"].split(":", 1)[1]
            for record in summary["prompts"]:
                if record["id"] == failed_id and record["status"] not in {"complete", "complete_with_request_errors"}:
                    record["status"] = "failed"
    finally:
        summary["logged_classification_requests"] = sum(
            len(path.read_text(encoding="utf-8").splitlines())
            for path in (output / "evaluations").glob("*/predictions.jsonl")
        )
        save_summary(output, summary)
    print(f"Pipeline status: {summary['status']}", flush=True)
    for row in summary["results"]:
        print(f"{row['prompt_id']}: macro-F1={row['macro_f1']:.6f}, invalid={row['invalid_output_rate']:.2%}, request_errors={row['request_error_count']}", flush=True)
    if summary["error"]:
        print(f"Pipeline error at {summary['stage']}: {summary['error']}", flush=True)
    print(f"Summary: {output / 'summary.csv'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS,
                        help="JSON file with a prompts list of {id, instruction} entries; default: prompts/humanPrompt.json")
    parser.add_argument("--data-file", type=Path,
                        help="Evaluate ALL rows of this dev TSV; default: data/processed/depsign/dev_v1/dev_clean.tsv")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Check and save all file prompts and classifier requests; no model calls")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ROOT / "data/processed/depsign/pipeline_smoke_runs" / stamp
    summary = run_pipeline(output, prompts_file=args.prompts_file, data_file=args.data_file,
                           prepare_only=args.prepare_only)
    if summary["status"] not in {"complete", "prepared"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
