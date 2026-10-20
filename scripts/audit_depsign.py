"""Audit train/dev only, without changing source data or exporting post text."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata


LABELS = {"not depression": "not_depression", "moderate": "moderate", "severe": "severe"}


def normalize(text):
    """Whitespace/NFC audit only: preserve case, punctuation and meaning."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def group_rows(rows, text_column, normalized=False):
    groups = defaultdict(list)
    for row in rows:
        text = row[text_column]
        groups[normalize(text) if normalized else text].append(row["Label"])
    return groups


def group_stats(groups):
    conflicts = [labels for labels in groups.values() if len(set(labels)) > 1]
    return {
        "unique_texts": len(groups),
        "duplicate_groups": sum(len(labels) > 1 for labels in groups.values()),
        "duplicate_extra_rows": sum(len(labels) - 1 for labels in groups.values()),
        "conflicting_label_groups": len(conflicts),
        "rows_in_conflicting_groups": sum(map(len, conflicts)),
        "maximum_multiplicity": max(map(len, groups.values()), default=0),
    }


def git_value(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def audit(directory):
    result = {
        "audit_version": 1,
        "source_commit": git_value(directory, "rev-parse", "HEAD"),
        "source_tracked_changes": git_value(directory, "diff", "HEAD", "--name-only"),
        "scope": ["train", "dev"],
        "test": {"present": (directory / "test.tsv").is_file(), "content_audited": False},
        "normalization": "Unicode NFC + whitespace collapse, case preserved",
        "proposed_label_mapping": LABELS,
        "user_id_available": False,
        "splits": {},
        "overlap": {},
    }
    all_rows, all_groups = {}, {}
    for split, text_column in (("train", "Text_data"), ("dev", "Text data")):
        path = directory / f"{split}.tsv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t", strict=True)
            columns = reader.fieldnames
            if columns != ["PID", text_column, "Label"]:
                raise ValueError(f"Unexpected {split} schema: {columns}")
            rows = list(reader)
        if not rows or any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Empty dataset or malformed rows in {split}")
        all_rows[split] = rows
        groups = {
            "exact": group_rows(rows, text_column),
            "normalized": group_rows(rows, text_column, normalized=True),
        }
        all_groups[split] = groups
        counts = Counter(row["Label"] for row in rows)
        if set(counts) != set(LABELS):
            raise ValueError(f"Unexpected labels in {split}: {sorted(counts)}")
        lengths = sorted(len(row[text_column]) for row in rows)
        result["splits"][split] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "columns": columns,
            "text_column": text_column,
            "row_count": len(rows),
            "labels": dict(sorted(counts.items())),
            "missing_or_blank": {
                name: sum(not row[name].strip() for row in rows) for name in columns
            },
            "duplicate_pid_extra_rows": len(rows) - len({row["PID"] for row in rows}),
            "text_characters": {"min": lengths[0], "max": lengths[-1]},
            **{mode: group_stats(g) for mode, g in groups.items()},
        }
    for mode in ("exact", "normalized"):
        train = all_groups["train"][mode]
        dev = all_groups["dev"][mode]
        common = train.keys() & dev.keys()
        result["overlap"][mode] = {
            "shared_unique_texts": len(common),
            "train_rows_with_shared_text": sum(len(train[t]) for t in common),
            "dev_rows_with_shared_text": sum(len(dev[t]) for t in common),
            "shared_groups_with_conflicting_labels": sum(
                len(set(train[t] + dev[t])) > 1 for t in common
            ),
        }
    result["overlap"]["pid_count"] = len(
        {r["PID"] for r in all_rows["train"]} & {r["PID"] for r in all_rows["dev"]}
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/external/detecting-signs-of-depression-from-social-media-postings"),
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir), ensure_ascii=False, indent=2, sort_keys=True))
