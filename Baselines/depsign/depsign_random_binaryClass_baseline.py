"""Uniform random binary-class baseline on dev100; no model or GPU required.

Gold labels moderate and severe are merged into depression. Each post independently
receives not_depression or depression with probability 1/2.
Example: python -B Baselines/depsign/depsign_random_binaryClass_baseline.py --seed 11
"""

import argparse
from collections import Counter
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import depsign_evaluator as evaluator

DEFAULT_DATA = ROOT / "data/processed/depsign/dev_100.tsv"
DEFAULT_OUTPUT = Path(__file__).with_name("depsign_random_binaryClass_baseline_results.json")


def run_baseline(data_file=DEFAULT_DATA, seed=11):
    dataset = evaluator.load_data(data_file)
    spec = evaluator.task_spec("binary")
    labels = spec["labels"]
    rng = random.Random(seed)
    records = [{"PID": row["PID"], "gold": spec["label_mapping"][row["Label"]],
                "prediction": rng.choice(labels)} for row in dataset["selected"]]
    metrics = evaluator.calculate_metrics(records, label_mode="binary")
    return {
        "baseline": "uniform_random_binary_class",
        "label_mapping": spec["label_mapping"],
        "purpose": "development_baseline_not_held_out_test",
        "seed": seed,
        "sampling": "Independent uniform choice per row, in source file order; no use of gold labels or post text for prediction.",
        "label_probabilities": {label: 1 / len(labels) for label in labels},
        "data_file": str(dataset["source"]),
        "data_sha256": evaluator.digest(dataset["data"]),
        "n": len(records),
        "gold_counts": {label: Counter(r["gold"] for r in records)[label] for label in labels},
        "prediction_counts": {label: Counter(r["prediction"] for r in records)[label] for label in labels},
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "predictions": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11, help="Random seed (default: 11)")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA, help="Default: fixed dev_100.tsv")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT,
                        help="JSON result file; replaced on each run")
    args = parser.parse_args()
    report = run_baseline(args.data_file, args.seed)
    output = args.output_file.resolve()
    if output in {args.data_file.resolve(), Path(__file__).resolve()}:
        parser.error("Output must not overwrite the dataset or script")
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_json(output, report)
    print(f"Uniform random baseline: n={report['n']}, seed={args.seed}")
    print(f"Macro-F1: {report['macro_f1']:.6f}")
    for label, values in report["per_class"].items():
        print(f"{label}: P={values['precision']:.6f}, R={values['recall']:.6f}, F1={values['f1']:.6f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
