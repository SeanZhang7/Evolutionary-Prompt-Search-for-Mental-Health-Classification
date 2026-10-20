"""Clean exact duplicates/conflicts within DepSign dev; never read train/test."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path


COLUMNS = ["PID", "Text data", "Label"]
LABELS = ["not depression", "moderate", "severe"]
DECISION_COLUMNS = [
    "source_record", "PID", "Label", "text_sha256", "decision", "retained_pid"
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def classify_rows(rows):
    """Conflict detection precedes deduplication; preserve input order and values."""
    groups = defaultdict(list)
    seen_pids = set()
    for index, row in enumerate(rows, start=1):
        if set(row) != set(COLUMNS) or any(
            not isinstance(row[k], str) or not row[k].strip() for k in COLUMNS
        ):
            raise ValueError(f"Missing, blank or malformed fields at record {index}")
        if row["Label"] not in LABELS:
            raise ValueError(f"Unknown label at record {index}")
        if row["PID"] in seen_pids:
            raise ValueError(f"Repeated PID at record {index}")
        seen_pids.add(row["PID"])
        groups[row["Text data"]].append(row)
    conflicting = {
        text for text, group in groups.items()
        if len({row["Label"] for row in group}) > 1
    }
    kept, excluded, decisions = [], [], []
    for index, row in enumerate(rows, start=1):
        text = row["Text data"]
        first_pid = groups[text][0]["PID"]
        if text in conflicting:
            decision, retained_pid = "label_conflict", ""
        elif row["PID"] != first_pid:
            decision, retained_pid = "duplicate_same_label", first_pid
        else:
            decision, retained_pid = "keep", first_pid
        metadata = {
            "source_record": index,
            "PID": row["PID"],
            "Label": row["Label"],
            "text_sha256": sha256(text.encode("utf-8")),
            "decision": decision,
            "retained_pid": retained_pid,
        }
        decisions.append(metadata)
        if decision == "keep":
            kept.append(row)
        else:
            excluded.append({**row, **metadata})
    return kept, excluded, decisions, len(conflicting)


def write_tsv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clean(source, output_dir):
    if source.resolve().parent == output_dir.resolve() or output_dir.resolve().is_relative_to(source.resolve().parent):
        raise ValueError("Write derived files outside the source data directory")
    source_bytes = source.read_bytes()
    reader = csv.DictReader(
        io.StringIO(source_bytes.decode("utf-8-sig"), newline=""),
        delimiter="\t", strict=True,
    )
    if reader.fieldnames != COLUMNS:
        raise ValueError(f"Unexpected dev schema: {reader.fieldnames}")
    rows = list(reader)
    if not rows:
        raise ValueError("Empty development set")
    kept, excluded, decisions, conflict_groups = classify_rows(rows)
    before = Counter(row["Label"] for row in rows)
    after = Counter(row["Label"] for row in kept)
    duplicates = Counter(row["Label"] for row in excluded if row["decision"] == "duplicate_same_label")
    conflicts = Counter(row["Label"] for row in excluded if row["decision"] == "label_conflict")
    summary = {
        "policy_version": "depsign-dev-exact-v1",
        "scope": "dev_only",
        "source_file": source.name,
        "source_sha256": sha256(source_bytes),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "rules": {
            "grouping": "exact parsed text; no normalization",
            "conflicts": "exclude every row of a text group with multiple labels",
            "duplicates": "keep first source record of a non-conflicting text group",
            "record_index": "1-based data record excluding header; not physical line number",
            "train_test_used": False,
            "text_and_labels_modified": False,
        },
        "conflicting_text_groups": conflict_groups,
        "counts": {
            label: {
                "before": before[label], "after": after[label],
                "duplicate_same_label": duplicates[label], "label_conflict": conflicts[label],
            } for label in LABELS
        },
        "totals": {
            "before": len(rows), "after": len(kept),
            "duplicate_same_label": sum(duplicates.values()), "label_conflict": sum(conflicts.values()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "dev_clean.tsv", COLUMNS, kept)
    write_tsv(output_dir / "dev_excluded.tsv", COLUMNS + [c for c in DECISION_COLUMNS if c not in COLUMNS], excluded)
    write_tsv(output_dir / "dev_decisions.tsv", DECISION_COLUMNS, decisions)
    summary["output_sha256"] = {
        name: sha256((output_dir / name).read_bytes())
        for name in ("dev_clean.tsv", "dev_excluded.tsv", "dev_decisions.tsv")
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "data/external/detecting-signs-of-depression-from-social-media-postings/dev.tsv"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/depsign/dev_v1"))
    args = parser.parse_args()
    print(json.dumps(clean(args.input, args.output_dir), ensure_ascii=False, indent=2))
