#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path


FAILED_PATTERNS = [
    re.compile(r"Failed on\s+\[['\"]([^'\"]+)['\"]\]", re.IGNORECASE),
]

TEST_SKIP_PATTERNS = [
    re.compile(r"The\s+test\s+dataset\s+did\s+not\s+contain\s+([A-Za-z0-9_.-]+)", re.IGNORECASE),
]

CONF_SKIP_PATTERNS = [
    re.compile(r"The\s+confidence\s+dataset\s+did\s+not\s+contain\s+\[['\"]([^'\"]+)['\"]\]", re.IGNORECASE),
]


def parse_hits_from_line(line, patterns):
    hits = []
    for pat in patterns:
        for m in pat.finditer(line):
            hits.append(m.group(1).strip())
    return hits


def collect_cases_from_log(log_path: Path):
    cases = {
        "failure": set(),
        "skip_test": set(),
        "skip_confidence": set(),
    }

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for x in parse_hits_from_line(line, FAILED_PATTERNS):
                cases["failure"].add(x)
            for x in parse_hits_from_line(line, TEST_SKIP_PATTERNS):
                cases["skip_test"].add(x)
            for x in parse_hits_from_line(line, CONF_SKIP_PATTERNS):
                cases["skip_confidence"].add(x)

    cases["all_failure_or_skip"] = (
        cases["failure"]
        | cases["skip_test"]
        | cases["skip_confidence"]
    )
    return cases


def read_csv_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_metadata_index(target_path: Path):
    metadata_index = {}
    target = target_path.name

    csv_specs = [
        (target_path / f"{target}_actives.csv", "actives"),
        (target_path / f"{target}_decoys.csv", "decoys"),
    ]

    for csv_path, split in csv_specs:
        if not csv_path.exists():
            continue

        rows = read_csv_rows(csv_path)
        for row in rows:
            complex_name = str(row.get("complex_name", "")).strip()
            if not complex_name:
                continue

            metadata_index[complex_name] = {
                "complex_name": complex_name,
                "protein_path": str(row.get("protein_path", "")).strip(),
                "protein_sequence": str(row.get("protein_sequence", "")).strip(),
                "ligand_description": str(row.get("ligand_description", "")).strip(),
                "split": split,
            }

    return metadata_index


def append_records(records, complex_names, metadata_index):
    for complex_name in sorted(complex_names):
        meta = metadata_index.get(complex_name)

        if meta is None:
            records.append({
                "complex_name": complex_name,
                "protein_path": "",
                "protein_sequence": "",
                "ligand_description": "",
                "split": "",
            })
        else:
            records.append({
                "complex_name": meta["complex_name"],
                "protein_path": meta["protein_path"],
                "protein_sequence": meta["protein_sequence"],
                "ligand_description": meta["ligand_description"],
                "split": meta["split"],
            })


def deduplicate_records(records):
    seen = set()
    unique = []
    for r in records:
        key = (
            r["complex_name"],
            r["protein_path"],
            r["protein_sequence"],
            r["ligand_description"],
            r["split"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def write_csv(out_path: Path, records):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "complex_name",
                "protein_path",
                "protein_sequence",
                "ligand_description",
                "split",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def find_target_dirs(dude_root: Path):
    return sorted([p for p in dude_root.iterdir() if p.is_dir()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dude_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--log_glob", default="*.err")
    args = parser.parse_args()

    dude_root = Path(args.dude_root)
    out_dir = Path(args.out_dir)

    all_fail_or_skip_records = []
    failure_only_records = []
    skip_test_only_records = []
    skip_conf_only_records = []

    target_dirs = find_target_dirs(dude_root)

    for target_path in target_dirs:
        logs_dir = target_path / "logs"
        if not logs_dir.exists():
            continue

        metadata_index = build_metadata_index(target_path)

        log_files = sorted(logs_dir.glob(args.log_glob))
        if not log_files:
            continue

        for log_path in log_files:
            cases = collect_cases_from_log(log_path)

            append_records(all_fail_or_skip_records, cases["all_failure_or_skip"], metadata_index)
            append_records(failure_only_records, cases["failure"], metadata_index)
            append_records(skip_test_only_records, cases["skip_test"], metadata_index)
            append_records(skip_conf_only_records, cases["skip_confidence"], metadata_index)

    all_fail_or_skip_records = deduplicate_records(all_fail_or_skip_records)
    failure_only_records = deduplicate_records(failure_only_records)
    skip_test_only_records = deduplicate_records(skip_test_only_records)
    skip_conf_only_records = deduplicate_records(skip_conf_only_records)

    write_csv(out_dir / "failure_and_skipped_cases.csv", all_fail_or_skip_records)
    write_csv(out_dir / "failure_cases.csv", failure_only_records)
    write_csv(out_dir / "skipped_test_dataset_cases.csv", skip_test_only_records)
    write_csv(out_dir / "skipped_confidence_dataset_cases.csv", skip_conf_only_records)


if __name__ == "__main__":
    main()
    
'''
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/aggregate/build_failure_skip_exports.py \
  --dude_root /home/deepfold/users/hosung/dataset/DUD-E/dude_raw \
  --out_dir /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports
'''
