"""Evaluate one instruction file on a DepSign-format TSV and report errors."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

if __package__:
    from . import depsign_evaluator as evaluator
else:
    import depsign_evaluator as evaluator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "prompts/BestPrompts/best_prompt.txt"
DEFAULT_DATA = ROOT / "data/processed/depsign/dev_100.tsv"
RESULTS = ROOT / "Results/depsign"
CSV_FIELDS = ["PID", "gold", "prediction", "error_type", "text", "input_text_used",
              "truncated", "raw_output", "finish_reason", "request_error", "reasoning"]


def analyze_predictions(records, label_mode="three_class"):
    """Extend the shared macro-F1 calculation without changing its invalid policy."""
    labels = evaluator.task_spec(label_mode)["labels"]
    metrics = evaluator.calculate_metrics(records, label_mode)
    matrix = metrics["confusion_matrix"]["values"]
    n = metrics["n"]
    metrics["accuracy"] = sum(matrix[i][i] for i in range(len(labels))) / n
    metrics["accuracy_definition"] = "Correct canonical labels / all evaluated records; INVALID is incorrect."
    metrics["class_accuracy_definition"] = (
        "One-vs-rest accuracy = (TP + TN) / n. All non-target outputs, including INVALID, "
        "are negative in this binary calculation. This is different from recall."
    )
    metrics["zero_division"] = 0
    pairs, by_class = [], {}
    for i, label in enumerate(labels):
        row = matrix[i]
        support = sum(row)
        tp = row[i]
        fp = sum(other[i] for other in matrix) - tp
        fn = support - tp
        tn = n - tp - fp - fn
        metrics["per_class"][label].update(accuracy_ovr=(tp + tn) / n, tp=tp, fp=fp, fn=fn, tn=tn)
        wrong_labels = []
        for j, prediction in enumerate(labels):
            if i != j and row[j]:
                pair = {"gold": label, "prediction": prediction, "count": row[j],
                        "rate_within_true_class": row[j] / support}
                pairs.append(pair)
                wrong_labels.append(pair)
        largest = max((pair["count"] for pair in wrong_labels), default=0)
        by_class[label] = {
            "support": support, "correct_count": tp, "error_count": fn,
            "error_rate": fn / support if support else None,
            "misclassified_count": sum(pair["count"] for pair in wrong_labels),
            "invalid_count": row[-1], "invalid_rate": row[-1] / support if support else None,
            "most_common_wrong_labels": [pair["prediction"] for pair in wrong_labels if pair["count"] == largest],
        }
    pairs.sort(key=lambda pair: (-pair["count"], labels.index(pair["gold"]),
                                 labels.index(pair["prediction"])))
    max_count = max((pair["count"] for pair in pairs), default=0)
    max_rate = max((pair["rate_within_true_class"] for pair in pairs), default=0)
    worst_rate = max((item["error_rate"] for item in by_class.values() if item["support"]), default=0)
    analysis = {
        "error_count": sum(record["gold"] != record["prediction"] for record in records),
        "misclassification_count": sum(pair["count"] for pair in pairs),
        "confusion_pairs": pairs,
        "most_frequent_confusions": [pair for pair in pairs if pair["count"] == max_count],
        "highest_rate_confusions": [pair for pair in pairs if pair["rate_within_true_class"] == max_rate],
        "highest_error_rate_classes": [label for label, item in by_class.items()
                                       if worst_rate > 0 and item["error_rate"] == worst_rate],
        "by_true_class": by_class,
        "rate_definition": "Directional confusion rate = count(gold -> predicted) / support(gold).",
        "invalid_policy": "INVALID outputs and request errors are reported separately from wrong-label confusions.",
    }
    return metrics, analysis


def render_report(report):
    lines = ["# Single-prompt DepSign evaluation", "", f"Status: **{report['status']}**",
             f"Evaluated: {report['evaluated_n']} / {report['requested_n']} records.",
             f"Label mode: {report.get('label_mode', 'three_class')}", ""]
    if report["error"]:
        lines += ["Run error:", "", "```text", report["error"], "```", ""]
    metrics = report["metrics"]
    if metrics is None:
        return "\n".join(lines + ["No scored predictions available.", ""])
    if report["evaluated_n"] != report["requested_n"]:
        lines += ["**Partial evaluation: metrics below describe only the evaluated rows.**", ""]
    lines += [f"Macro-F1: **{metrics['macro_f1']:.6f}**",
              f"Overall accuracy: **{metrics['accuracy']:.6f}**",
              f"Invalid-output rate: {metrics['invalid_output_rate']:.2%}; request errors: {metrics['request_error_count']}.", "",
              "| Class | Support | Precision (P) | Recall (R) | F1 | Accuracy (A, one-vs-rest) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, values in metrics["per_class"].items():
        lines.append(f"| {label} | {values['support']} | {values['precision']:.6f} | "
                     f"{values['recall']:.6f} | {values['f1']:.6f} | {values['accuracy_ovr']:.6f} |")
    lines += ["", "A is (TP + TN) / n with this class positive and all other outputs negative; it is not recall.",
              f"Undefined precision/recall/F1 is set to zero. Macro-F1 averages the {len(metrics['per_class'])} task labels.", "",
              "## Confusion matrix", "", "Rows are true labels; columns are predictions. Cells are counts.", "",
              "| True / Predicted | " + " | ".join(metrics["confusion_matrix"]["columns"]) + " |",
              "| --- | " + " | ".join(["---:"] * len(metrics["confusion_matrix"]["columns"])) + " |"]
    for label, values in zip(metrics["confusion_matrix"]["rows"], metrics["confusion_matrix"]["values"]):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in values) + " |")
    analysis = report["error_analysis"]
    lines += ["", "## Error analysis", "",
              "Wrong-label confusions exclude INVALID outputs and request errors. Rates use the true class's support.", ""]
    for title, key in (("Most frequent confusion(s)", "most_frequent_confusions"),
                       ("Highest within-class confusion rate(s)", "highest_rate_confusions")):
        pairs = analysis[key]
        text = "; ".join(f"{p['gold']} → {p['prediction']}: {p['count']} ({p['rate_within_true_class']:.2%} of {p['gold']})"
                         for p in pairs) if pairs else "None"
        lines += [f"**{title}:** {text}", ""]
    lines += ["| True class | Errors (including INVALID) | Error rate | INVALID | Most common wrong label(s) |",
              "| --- | ---: | ---: | ---: | --- |"]
    for label, values in analysis["by_true_class"].items():
        rate = f"{values['error_rate']:.2%}" if values["error_rate"] is not None else "N/A (no support)"
        lines.append(f"| {label} | {values['error_count']} | {rate} | {values['invalid_count']} | "
                     f"{', '.join(values['most_common_wrong_labels']) or 'None'} |")
    lines += ["", "| True → Predicted | Count | Rate within true class |", "| --- | ---: | ---: |"]
    for pair in analysis["confusion_pairs"]:
        lines.append(f"| {pair['gold']} → {pair['prediction']} | {pair['count']} | {pair['rate_within_true_class']:.2%} |")
    lines += ["", "See errors.csv for every incorrect/invalid prediction and predictions.csv for all evaluated rows.",
              "Both files include the original post, the text actually sent to the model, and raw model output.",
              "These tables identify error patterns, not their causes. Evaluation on search data is not held-out evaluation.", ""]
    return "\n".join(lines)


def write_predictions(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run_evaluation(output, *, prompt_file=DEFAULT_PROMPT, data_file=DEFAULT_DATA, prepare_only=False, label_mode="three_class"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    report = {"status": "preparing", "error": None, "purpose": "single_prompt_classification_evaluation",
              "requested_n": 0, "evaluated_n": 0, "metrics": None, "error_analysis": None,
              "model": evaluator.MODEL, "base_url": evaluator.BASE_URL,
              "classification_parameters": dict(evaluator.PARAMETERS), **evaluator.task_metadata(label_mode)}
    print(f"Results: {output}", flush=True)
    try:
        prompt_file = Path(prompt_file).resolve()
        prompt_bytes = prompt_file.read_bytes()
        dataset = evaluator.load_data(data_file)
        samples = dataset["selected"]
        report["requested_n"] = len(samples)
        report["dataset"] = {"path": str(dataset["source"]), "sha256": evaluator.digest(dataset["data"]),
                             "class_counts": dict(Counter(report["label_mapping"][row["Label"]] for row in samples))}
        tokenizer = evaluator.load_tokenizer()
        candidate = evaluator.prepare_candidate(prompt_bytes.decode("utf-8"), samples, tokenizer, label_mode=label_mode)
        report["prompt"] = {"path": str(prompt_file), "source_sha256": evaluator.digest(prompt_bytes),
                            "instruction": candidate["instruction"], "tokens": candidate["instruction_tokens"],
                            "instruction_sha256": evaluator.digest(candidate["instruction"].encode("utf-8"))}
        report["task_template"] = {"path": report["task_template_path"], "text": candidate["task_template"],
                                   "sha256": evaluator.digest(candidate["task_template"].encode("utf-8"))}
        report["truncated_count"] = sum(item["truncated"] for item in candidate["requests"])
        report["text_token_limit"] = evaluator.TEXT_TOKENS
        if prepare_only:
            report["status"] = "prepared"
        else:
            models = json.loads(evaluator.http_call("/v1/models"))
            matches = [model for model in models["data"] if model["id"] == evaluator.MODEL]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one served model named {evaluator.MODEL}")
            if matches[0].get("max_model_len") is not None and matches[0]["max_model_len"] < evaluator.CONTEXT_TOKENS:
                raise ValueError("Server context is smaller than the configured context")
            report["service"] = {"model": matches[0], "version": json.loads(evaluator.http_call("/version"))}
            report["status"] = "running"

            def record_result(index, total, result):
                request = candidate["requests"][index - 1]
                error_type = ("request_error" if result["request_error"] else
                              "invalid_output" if result["prediction"] == "INVALID" else
                              "misclassification" if result["prediction"] != result["gold"] else "correct")
                records.append({**result, "error_type": error_type, "text": samples[index - 1]["Text data"],
                                "input_text_used": request["input_text_used"], "truncated": request["truncated"]})
                evaluator.print_prediction(index, total, result)

            evaluation = evaluator.evaluate_candidate(candidate, on_result=record_result)
            report["status"] = "complete_with_request_errors" if evaluation["metrics"]["request_error_count"] else "complete"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        report["evaluated_n"] = len(records)
        if records:
            report["metrics"], report["error_analysis"] = analyze_predictions(records, label_mode)
        evaluator.save_json(output / "performance.json", report)
        if report["status"] != "prepared":
            (output / "report.md").write_text(render_report(report), encoding="utf-8")
            write_predictions(output / "predictions.csv", records)
            write_predictions(output / "errors.csv", [r for r in records if r["error_type"] != "correct"])
    print(f"Evaluation status: {report['status']}; evaluated {report['evaluated_n']}/{report['requested_n']}", flush=True)
    if report["metrics"]:
        metrics = report["metrics"]
        print(f"Macro-F1: {metrics['macro_f1']:.6f}; accuracy: {metrics['accuracy']:.6f}", flush=True)
        for label, values in metrics["per_class"].items():
            print(f"{label}: P={values['precision']:.6f}, R={values['recall']:.6f}, "
                  f"F1={values['f1']:.6f}, A_OVR={values['accuracy_ovr']:.6f}", flush=True)
    if report["error"]:
        print(report["error"], flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-mode", choices=evaluator.LABEL_MODES, default="three_class")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT,
                        help="One instruction block in a UTF-8 text file")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA,
                        help="TSV with PID, Text data, Label columns; defaults to dev_100.tsv")
    parser.add_argument("--output-dir", type=Path, help="New run directory; existing paths are never overwritten")
    parser.add_argument("--prepare-only", action="store_true", help="Validate all inputs without model requests")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.label_mode == "binary":
        stamp = "binary_" + stamp
    output = args.output_dir or RESULTS / stamp
    report = run_evaluation(output, prompt_file=args.prompt_file, data_file=args.data_file, prepare_only=args.prepare_only, label_mode=args.label_mode)
    if report["status"] not in {"complete", "prepared"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
