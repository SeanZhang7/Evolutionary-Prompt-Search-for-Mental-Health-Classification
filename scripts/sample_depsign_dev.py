"""Create a reproducible proportional subset of the complete cleaned DepSign dev."""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import platform
import random

if __package__:
    from . import depsign_evaluator as classifier
else:
    import depsign_evaluator as classifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/depsign/dev_100.tsv"


def sample_dev(output=DEFAULT_OUTPUT, *, size=100, seed=1729):
    output = Path(output).resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output == manifest_path or output.exists() or manifest_path.exists():
        raise FileExistsError("Use a new output path; existing sample or manifest will not be overwritten")
    dataset = classifier.load_data(ROOT / classifier.CLEAN_DEV)
    rows = dataset["rows"]
    if not 1 <= size <= len(rows):
        raise ValueError(f"Sample size must be between 1 and {len(rows)}")
    labels = list(classifier.LABEL_MAP)
    source_counts = Counter(row["Label"] for row in rows)
    # Integer largest-remainder allocation avoids floating-point rounding drift.
    quotas = {label: size * source_counts[label] // len(rows) for label in labels}
    remainder_order = sorted(labels, key=lambda label: (
        -(size * source_counts[label] % len(rows)), labels.index(label)
    ))
    for label in remainder_order[:size - sum(quotas.values())]:
        quotas[label] += 1
    rng = random.Random(seed)
    indices = []
    for label in labels:
        indices.extend(rng.sample(
            [index for index, row in enumerate(rows) if row["Label"] == label], quotas[label]
        ))
    rng.shuffle(indices)
    selected = [rows[index] for index in indices]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["PID", "Text data", "Label"],
                            delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
    sample_bytes = buffer.getvalue().encode("utf-8")
    manifest = {
        "policy_version": "depsign-proportional-dev-v1",
        "purpose": "exploratory_development_subset_not_held_out_test",
        "source": str(dataset["source"]),
        "source_sha256": classifier.digest(dataset["data"]),
        "source_record_count": len(rows),
        "cleaning_manifest_verified": dataset["is_cleaned_dev"],
        "sample_file": output.name,
        "sample_sha256": classifier.digest(sample_bytes),
        "sample_size": size,
        "sampling_seed": seed,
        "label_mapping": classifier.LABEL_MAP,
        "source_class_counts": {label: source_counts[label] for label in labels},
        "sample_class_counts": quotas,
        "sampling": {
            "quota": "largest remainder of size * class_count / total; ties use label_order",
            "label_order": labels,
            "within_class": "random.Random(seed).sample without replacement in source order",
            "output_order": "same RNG shuffles combined selected indices once",
            "record_index": "1-based data record excluding header; not physical line number",
            "text_and_labels_modified": False,
            "predictions_or_scores_used": False,
            "train_or_test_accessed": False,
        },
        "selected_records": [
            {"source_record": index + 1, "PID": rows[index]["PID"], "raw_label": rows[index]["Label"]}
            for index in indices
        ],
        "python_version": platform.python_version(),
        "script_sha256": classifier.digest(Path(__file__).read_bytes()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(sample_bytes)
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Sample: {output}")
    print(f"Manifest: {manifest_path}")
    print(f"Records: {len(selected)}; seed: {seed}; classes: {quotas}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New output TSV; default: data/processed/depsign/dev_100.tsv")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    sample_dev(args.output, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
