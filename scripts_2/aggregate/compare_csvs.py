#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_csvs.py

목적
- 동일 schema를 가진 2개 이상의 csv를 입력받는다.
- 각 csv의 ligand_description(SMILES)에서 RDKit descriptor를 대량 추출한다.
- group(label)별 descriptor 요약 통계를 저장한다.
- 2개 group이면 Mann-Whitney U / Cliff's delta / Cohen's d 를 계산한다.
- 3개 이상 group이면 Kruskal-Wallis 와 pairwise Mann-Whitney U / Cliff's delta / Cohen's d 를 계산한다.

입력 csv 요구 컬럼
- complex_name
- protein_path
- protein_sequence
- ligand_description
- split

사용 예시
1) success vs all-failure
python compare_csvs.py \
  --csv /path/success_cases.csv success \
  --csv /path/failure_and_skipped_cases.csv all_failure \
  --out_dir /path/compare_success_vs_failure

2) failure vs skip-test vs skip-conf
python compare_csvs.py \
  --csv /path/failure_cases.csv failure \
  --csv /path/skipped_test_dataset_cases.csv skip_test \
  --csv /path/skipped_confidence_dataset_cases.csv skip_conf \
  --out_dir /path/compare_failure_modes
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import kruskal, mannwhitneyu

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors


REQUIRED_COLUMNS = [
    "complex_name",
    "protein_path",
    "protein_sequence",
    "ligand_description",
    "split",
]


DESCRIPTOR_COLUMNS = [
    "mol_wt",
    "exact_mol_wt",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "heavy_atom_count",
    "formal_charge",
    "fraction_csp3",
    "num_atoms",
    "num_hetero_atoms",
    "bertz_ct",
    "chiral_center_count",
    "num_spiro_atoms",
    "num_bridgehead_atoms",
    "smiles_length",
    "smiles_bracket_count",
    "smiles_branch_open_count",
    "smiles_branch_close_count",
    "smiles_chiral_symbol_count",
    "smiles_ring_digit_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 2 or more CSV groups by RDKit descriptors."
    )
    parser.add_argument(
        "--csv",
        nargs=2,
        action="append",
        metavar=("CSV_PATH", "LABEL"),
        required=True,
        help="Add one input csv and its label. Use at least twice.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--drop_invalid_smiles",
        action="store_true",
        help="Drop rows with invalid SMILES from descriptor analysis.",
    )
    parser.add_argument(
        "--keep_only_common_columns",
        action="store_true",
        help="If schema differs slightly, keep common columns instead of strict equality check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs.",
    )
    return parser.parse_args()


def validate_input_count(csv_args: List[List[str]]) -> None:
    if csv_args is None or len(csv_args) < 2:
        raise ValueError("You must provide at least 2 --csv arguments.")


def read_and_validate_csv(
    csv_path: Path,
    keep_only_common_columns: bool = False,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def validate_schema_strict(dfs: List[pd.DataFrame], paths: List[Path]) -> List[str]:
    base_cols = list(dfs[0].columns)
    for i, df in enumerate(dfs[1:], start=1):
        cols = list(df.columns)
        if cols != base_cols:
            raise ValueError(
                f"Column mismatch between:\n"
                f"  {paths[0]} -> {base_cols}\n"
                f"  {paths[i]} -> {cols}"
            )
    return base_cols


def validate_schema_common(dfs: List[pd.DataFrame], paths: List[Path]) -> List[str]:
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    common_cols = [c for c in dfs[0].columns if c in common_cols]
    if not common_cols:
        raise ValueError("No common columns across input CSVs.")

    return common_cols


def ensure_required_columns(columns: List[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"Required columns missing after schema alignment: {missing}\n"
            f"Required columns are: {REQUIRED_COLUMNS}"
        )


def count_chiral_centers(mol: Chem.Mol) -> int:
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return len(centers)
    except Exception:
        return np.nan


def safe_formal_charge(mol: Chem.Mol) -> float:
    try:
        return float(Chem.GetFormalCharge(mol))
    except Exception:
        return np.nan


def smiles_proxy_features(smiles: str) -> Dict[str, float]:
    s = smiles if isinstance(smiles, str) else ""
    return {
        "smiles_length": float(len(s)),
        "smiles_bracket_count": float(s.count("[") + s.count("]")),
        "smiles_branch_open_count": float(s.count("(")),
        "smiles_branch_close_count": float(s.count(")")),
        "smiles_chiral_symbol_count": float(s.count("@")),
        "smiles_ring_digit_count": float(sum(ch.isdigit() for ch in s)),
    }


def compute_descriptors_from_smiles(smiles: str) -> Dict[str, float]:
    result = {k: np.nan for k in DESCRIPTOR_COLUMNS}
    if not isinstance(smiles, str) or not smiles.strip():
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    try:
        result.update({
            "mol_wt": float(Descriptors.MolWt(mol)),
            "exact_mol_wt": float(Descriptors.ExactMolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "aliphatic_ring_count": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
            "formal_charge": safe_formal_charge(mol),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            "num_atoms": float(mol.GetNumAtoms()),
            "num_hetero_atoms": float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "bertz_ct": float(Descriptors.BertzCT(mol)),
            "chiral_center_count": float(count_chiral_centers(mol)),
            "num_spiro_atoms": float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
            "num_bridgehead_atoms": float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        })
        result.update(smiles_proxy_features(smiles))
    except Exception:
        pass

    return result


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return np.nan
    return (mx - my) / math.sqrt(pooled)


def summarize_group(df: pd.DataFrame, group_col: str, desc_cols: List[str]) -> pd.DataFrame:
    rows = []
    for g, dfg in df.groupby(group_col):
        for col in desc_cols:
            vals = pd.to_numeric(dfg[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            rows.append({
                "group": g,
                "descriptor": col,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
                "q1": float(np.quantile(vals, 0.25)) if len(vals) else np.nan,
                "q3": float(np.quantile(vals, 0.75)) if len(vals) else np.nan,
                "min": float(np.min(vals)) if len(vals) else np.nan,
                "max": float(np.max(vals)) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def run_pairwise_tests(
    df: pd.DataFrame,
    group_col: str,
    desc_cols: List[str],
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    rows = []
    dfa = df[df[group_col] == group_a]
    dfb = df[df[group_col] == group_b]

    for col in desc_cols:
        xa = pd.to_numeric(dfa[col], errors="coerce").to_numpy(dtype=float)
        xb = pd.to_numeric(dfb[col], errors="coerce").to_numpy(dtype=float)

        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]

        if len(xa) == 0 or len(xb) == 0:
            pval = np.nan
            stat = np.nan
        else:
            try:
                stat, pval = mannwhitneyu(xa, xb, alternative="two-sided")
            except Exception:
                stat, pval = np.nan, np.nan

        rows.append({
            "test_type": "pairwise",
            "group_a": group_a,
            "group_b": group_b,
            "descriptor": col,
            "n_a": int(len(xa)),
            "n_b": int(len(xb)),
            "mean_a": float(np.mean(xa)) if len(xa) else np.nan,
            "mean_b": float(np.mean(xb)) if len(xb) else np.nan,
            "median_a": float(np.median(xa)) if len(xa) else np.nan,
            "median_b": float(np.median(xb)) if len(xb) else np.nan,
            "mw_u_stat": float(stat) if pd.notna(stat) else np.nan,
            "p_value": float(pval) if pd.notna(pval) else np.nan,
            "cliffs_delta": float(cliffs_delta(xa, xb)),
            "cohens_d": float(cohens_d(xa, xb)),
            "mean_diff_a_minus_b": (
                float(np.mean(xa) - np.mean(xb)) if len(xa) and len(xb) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def run_global_tests(
    df: pd.DataFrame,
    group_col: str,
    desc_cols: List[str],
) -> pd.DataFrame:
    rows = []
    groups = list(df[group_col].dropna().unique())

    for col in desc_cols:
        arrays = []
        ns = []
        for g in groups:
            vals = pd.to_numeric(df.loc[df[group_col] == g, col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            arrays.append(vals)
            ns.append(len(vals))

        if sum(n > 0 for n in ns) < 2:
            stat = np.nan
            pval = np.nan
        else:
            try:
                valid_arrays = [a for a in arrays if len(a) > 0]
                stat, pval = kruskal(*valid_arrays)
            except Exception:
                stat, pval = np.nan, np.nan

        rows.append({
            "test_type": "global",
            "group_a": "",
            "group_b": "",
            "descriptor": col,
            "n_groups": int(len(groups)),
            "group_names": "|".join(groups),
            "kruskal_stat": float(stat) if pd.notna(stat) else np.nan,
            "p_value": float(pval) if pd.notna(pval) else np.nan,
        })

    return pd.DataFrame(rows)


def group_counts_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for g, dfg in df.groupby(group_col):
        valid_smiles = dfg["ligand_description"].astype(str).str.strip().ne("").sum()
        rows.append({
            "group": g,
            "n_rows": int(len(dfg)),
            "n_unique_complex_name": int(dfg["complex_name"].nunique()),
            "n_valid_smiles_string": int(valid_smiles),
            "n_invalid_or_failed_descriptor": int(dfg["descriptor_valid"].eq(False).sum()),
            "n_actives": int(dfg["split"].astype(str).eq("actives").sum()),
            "n_decoys": int(dfg["split"].astype(str).eq("decoys").sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    validate_input_count(args.csv)

    csv_paths = [Path(x[0]) for x in args.csv]
    labels = [x[1] for x in args.csv]

    if len(set(labels)) != len(labels):
        raise ValueError(f"Labels must be unique. Got: {labels}")

    dfs = [read_and_validate_csv(p, keep_only_common_columns=args.keep_only_common_columns) for p in csv_paths]

    if args.keep_only_common_columns:
        common_cols = validate_schema_common(dfs, csv_paths)
    else:
        common_cols = validate_schema_strict(dfs, csv_paths)

    ensure_required_columns(common_cols)

    aligned_dfs = []
    for df, label in zip(dfs, labels):
        dfa = df[common_cols].copy()
        dfa["group"] = label
        aligned_dfs.append(dfa)

    merged = pd.concat(aligned_dfs, ignore_index=True)

    if args.verbose:
        print(f"[info] merged rows = {len(merged)}")
        print(f"[info] groups = {labels}")

    descriptor_records = []
    valid_flags = []

    for smiles in merged["ligand_description"].tolist():
        desc = compute_descriptors_from_smiles(smiles)
        descriptor_records.append(desc)
        valid_flags.append(bool(pd.notna(desc["mol_wt"])))

    desc_df = pd.DataFrame(descriptor_records)
    merged = pd.concat([merged.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    merged["descriptor_valid"] = valid_flags

    if args.drop_invalid_smiles:
        merged = merged.loc[merged["descriptor_valid"]].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_dir / "annotated_descriptors.csv", index=False)

    summary_df = summarize_group(merged, "group", DESCRIPTOR_COLUMNS)
    summary_df.to_csv(out_dir / "descriptor_summary_by_group.csv", index=False)

    counts_df = group_counts_table(merged, "group")
    counts_df.to_csv(out_dir / "group_counts.csv", index=False)

    n_groups = merged["group"].nunique()

    test_frames = []
    if n_groups == 2:
        g1, g2 = list(merged["group"].dropna().unique())
        pair_df = run_pairwise_tests(merged, "group", DESCRIPTOR_COLUMNS, g1, g2)
        test_frames.append(pair_df)
    else:
        global_df = run_global_tests(merged, "group", DESCRIPTOR_COLUMNS)
        test_frames.append(global_df)

        unique_groups = list(merged["group"].dropna().unique())
        for ga, gb in itertools.combinations(unique_groups, 2):
            pair_df = run_pairwise_tests(merged, "group", DESCRIPTOR_COLUMNS, ga, gb)
            test_frames.append(pair_df)

    tests_df = pd.concat(test_frames, ignore_index=True)
    tests_df.to_csv(out_dir / "descriptor_tests.csv", index=False)

    print(f"[done] saved annotated descriptors to: {out_dir / 'annotated_descriptors.csv'}")
    print(f"[done] saved group summary to: {out_dir / 'descriptor_summary_by_group.csv'}")
    print(f"[done] saved group counts to: {out_dir / 'group_counts.csv'}")
    print(f"[done] saved statistical tests to: {out_dir / 'descriptor_tests.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
        
'''
success vs all-failure
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/aggregate/compare_csvs.py \
  --csv /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports/success_cases.csv success \
  --csv /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports/failure_and_skipped_cases.csv all_failure \
  --out_dir /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports/success_all_fail_compare \
  --verbose

failure vs skip-test vs skip-conf
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/aggregate/compare_csvs.py \
  --csv /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports/failure_cases.csv failure \
  --csv /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports/skipped_test_dataset_cases.csv skip_test \
  --csv /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports/skipped_confidence_dataset_cases.csv skip_conf \
  --out_dir /home/deepfold/users/hosung/dataset/DUD-E/aggregate_exports/compare_failure_cases \
  --verbose
'''
