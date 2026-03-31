#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_success.py

목적:
- failure_and_skipped_cases.csv 와 DUDE_ROOT 를 입력으로 받는다.
- DUDE_ROOT 아래 각 target 디렉토리의 actives.csv, decoys.csv 를 읽는다.
- failure_and_skipped_cases.csv 에 없는 complex_name 만 모아서 success csv 를 생성한다.

입력:
- --failure_csv : failure_and_skipped_cases.csv 경로
- --dude_root   : DUD-E root 경로 (바로 아래에 target 디렉토리들이 존재해야 함)
- --out_csv     : 생성할 success csv 경로

출력 컬럼:
- complex_name
- protein_path
- protein_sequence
- ligand_description
- split   # actives 또는 decoys

가정:
- 각 target 디렉토리 아래에 actives.csv, decoys.csv 가 존재한다.
- actives.csv / decoys.csv 에 최소한 아래 컬럼이 존재한다.
    complex_name, protein_path, protein_sequence, ligand_description
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd


REQUIRED_FAILURE_COLUMNS = [
    "complex_name",
]

REQUIRED_TARGET_COLUMNS = [
    "complex_name",
    "protein_path",
    "protein_sequence",
    "ligand_description",
]

OUTPUT_COLUMNS = [
    "complex_name",
    "protein_path",
    "protein_sequence",
    "ligand_description",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build success_cases.csv by excluding failure/skipped complex_name from all actives/decoys rows."
    )
    parser.add_argument(
        "--failure_csv",
        type=str,
        required=True,
        help="Path to failure_and_skipped_cases.csv",
    )
    parser.add_argument(
        "--dude_root",
        type=str,
        required=True,
        help="Path to DUD-E root directory containing target subdirectories",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output path for success csv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-target processing logs",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required: List[str], csv_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def read_csv_checked(csv_path: Path, required_cols: List[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    validate_columns(df, required_cols, csv_path)
    return df


def list_target_dirs(dude_root: Path) -> List[Path]:
    if not dude_root.exists():
        raise FileNotFoundError(f"DUDE root not found: {dude_root}")
    if not dude_root.is_dir():
        raise NotADirectoryError(f"DUDE root is not a directory: {dude_root}")

    target_dirs = sorted([p for p in dude_root.iterdir() if p.is_dir()])
    if not target_dirs:
        raise ValueError(f"No target directories found under: {dude_root}")
    return target_dirs


def load_failure_complex_names(failure_csv: Path) -> Set[str]:
    df_fail = read_csv_checked(failure_csv, REQUIRED_FAILURE_COLUMNS)
    failure_names = set(
        df_fail["complex_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    return failure_names


def load_split_csv(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = read_csv_checked(csv_path, REQUIRED_TARGET_COLUMNS).copy()
    df["split"] = split_name
    return df


def build_success_rows(
    dude_root: Path,
    failure_names: Set[str],
    verbose: bool = False,
) -> pd.DataFrame:
    all_success_parts: List[pd.DataFrame] = []
    target_dirs = list_target_dirs(dude_root)

    total_all_rows = 0
    total_success_rows = 0

    for target_dir in target_dirs:
        target_name = target_dir.name

        actives_csv = target_dir / f"{target_name}_actives.csv"
        decoys_csv = target_dir / f"{target_name}_decoys.csv"

        if not actives_csv.exists() or not decoys_csv.exists():
            if verbose:
                print(f"[skip] {target_name}: {target_name}_actives.csv or {target_name}_decoys.csv missing")
            continue

        df_act = load_split_csv(actives_csv, "actives")
        df_dec = load_split_csv(decoys_csv, "decoys")

        df_target = pd.concat([df_act, df_dec], ignore_index=True)

        df_target["complex_name"] = df_target["complex_name"].astype(str).str.strip()

        total_rows_before = len(df_target)
        total_all_rows += total_rows_before

        df_success = df_target.loc[~df_target["complex_name"].isin(failure_names), OUTPUT_COLUMNS].copy()

        total_rows_after = len(df_success)
        total_success_rows += total_rows_after

        if verbose:
            print(
                f"[ok] {target_name}: total={total_rows_before}, "
                f"success={total_rows_after}, removed={total_rows_before - total_rows_after}"
            )

        all_success_parts.append(df_success)

    if not all_success_parts:
        raise ValueError("No valid target actives/decoys data found to build success cases.")

    df_success_all = pd.concat(all_success_parts, ignore_index=True)

    # 중복 제거
    # 동일 complex_name 이 여러 번 들어가는 경우를 방지한다.
    df_success_all = df_success_all.drop_duplicates(subset=["complex_name"], keep="first").copy()

    if verbose:
        print(f"[summary] total input rows from all targets = {total_all_rows}")
        print(f"[summary] total success rows before dedup = {total_success_rows}")
        print(f"[summary] total success rows after dedup = {len(df_success_all)}")

    return df_success_all


def main() -> None:
    args = parse_args()

    failure_csv = Path(args.failure_csv)
    dude_root = Path(args.dude_root)
    out_csv = Path(args.out_csv)

    try:
        failure_names = load_failure_complex_names(failure_csv)

        if args.verbose:
            print(f"[info] loaded failure/skipped complex_name count = {len(failure_names)}")

        df_success = build_success_rows(
            dude_root=dude_root,
            failure_names=failure_names,
            verbose=args.verbose,
        )

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_success.to_csv(out_csv, index=False)

        print(f"[done] saved success csv to: {out_csv}")
        print(f"[done] success row count: {len(df_success)}")

    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    
'''
python ./dataset/DUD-E/scripts_2/aggregate/build_success.py \
  --failure_csv ./dataset/DUD-E/aggregate_exports/failure_and_skipped_cases.csv \
  --dude_root ./dataset/DUD-E/dude_raw \
  --out_csv ./dataset/DUD-E/aggregate_exports/success_cases.csv \
  --verbose
'''
