"""Clean exact train-text duplicates and sample a fixed Dreaddit search subset."""

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import random

if __package__:
    from .depsign_evaluator import digest
else:
    from depsign_evaluator import digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/dreaddit/dreaddit-train.csv"
OUTPUT = ROOT / "data/processed/dreaddit/dre_dev_100.tsv"


def sample_dev(output=OUTPUT, *, source=SOURCE, size=100, seed=1729):
    output, source = Path(output).resolve(), Path(source).resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Existing sample or manifest will not be overwritten")
    source_bytes = source.read_bytes()
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"), newline=""))
    if not {"id", "post_id", "text", "label"} <= set(reader.fieldnames or []):
        raise ValueError("Expected Dreaddit CSV columns: id, post_id, text, label")
    rows = list(reader)
    groups = defaultdict(list)
    ids = set()
    for index, row in enumerate(rows, 1):
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("id", "post_id", "text", "label")) or row["label"] not in ("0", "1"):
            raise ValueError(f"Invalid Dreaddit record {index}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate source ID: {row['id']}")
        ids.add(row["id"])
        groups[row["text"]].append(index - 1)
    conflicts, duplicates, retained = [], [], []
    for indices in groups.values():
        if len({rows[i]["label"] for i in indices}) > 1:
            conflicts.append(indices)
        else:
            retained.append(indices[0])
            duplicates.extend(indices[1:])
    retained.sort()
    if type(size) is not int or not 1 <= size <= len(retained):
        raise ValueError(f"Sample size must be between 1 and {len(retained)}")
    labels = ["0", "1"]
    counts = Counter(rows[i]["label"] for i in retained)
    quotas = {label: size * counts[label] // len(retained) for label in labels}
    order = sorted(labels, key=lambda label: (-(size * counts[label] % len(retained)), labels.index(label)))
    for label in order[:size - sum(quotas.values())]:
        quotas[label] += 1
    rng = random.Random(seed)
    selected = []
    for label in labels:
        selected.extend(rng.sample([i for i in retained if rows[i]["label"] == label], quotas[label]))
    rng.shuffle(selected)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["PID", "Text data", "Label"], delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({"PID": rows[i]["id"], "Text data": rows[i]["text"], "Label": rows[i]["label"]}
                     for i in selected)
    sample_bytes = buffer.getvalue().encode("utf-8")

    def record(i):
        return {"source_record": i + 1, "PID": rows[i]["id"], "post_id": rows[i]["post_id"],
                "raw_label": rows[i]["label"]}

    manifest = {
        "policy_version": "dreaddit-proportional-search-v1", "dataset": "dreaddit",
        "purpose": "train_derived_prompt_search_subset_not_official_dev_or_held_out_test",
        "source": str(source), "source_sha256": digest(source_bytes), "source_record_count": len(rows),
        "source_class_counts": dict(Counter(r["label"] for r in rows)),
        "cleaning": {
            "text_comparison": "exact parsed text; no normalization or stripping",
            "same_label_policy": "keep first record in source order",
            "conflict_policy": "exclude every record in a conflicting text group",
            "duplicate_text_groups": sum(len(g) > 1 for g in groups.values()),
            "conflicting_text_groups": len(conflicts),
            "conflict_records_removed": sum(map(len, conflicts)),
            "same_label_duplicates_removed": len(duplicates),
            "excluded_conflict_groups": [[record(i) for i in group] for group in conflicts],
            "excluded_same_label_duplicates": [record(i) for i in duplicates],
        },
        "cleaned_record_count": len(retained), "cleaned_class_counts": dict(counts),
        "sample_file": output.name, "sample_sha256": digest(sample_bytes),
        "sample_size": size, "sampling_seed": seed, "sample_class_counts": quotas,
        "label_mapping": {"0": "0", "1": "1"}, "label_meanings": {"0": "not stress", "1": "stress"},
        "sampling": {"quota": "largest remainder; ties use label order 0, 1",
                     "within_class": "random.Random(seed).sample without replacement in source order",
                     "output_order": "same RNG shuffles selected indices once",
                     "unit": "text segment/item; multiple segments may share a post_id",
                     "text_and_labels_modified": False, "predictions_or_scores_used": False},
        "selected_records": [record(i) for i in selected],
        "test_accessed": False, "script_sha256": digest(Path(__file__).read_bytes()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(sample_bytes)
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Sample: {output}\nManifest: {manifest_path}\nCleaned: {len(retained)}; sample: {quotas}; seed: {seed}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    sample_dev(args.output, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
