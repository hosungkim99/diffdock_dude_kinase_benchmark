#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_pseudo_efb.py

Purpose
-------
Compute pseudo-EFB metrics from active/decoy inference scores and update
metrics_summary_all_<eval_subdir>.csv.

Pseudo-EFB definition
---------------------
This is NOT the true Bayes enrichment factor from the paper.
Instead, it uses decoys as a proxy for random compounds:

    pEFB_chi = P(S >= S_chi | active) / P(S >= S_chi | decoy)

where S_chi is determined from the decoy score distribution.

Expected input
--------------
For each target, a score CSV containing at least:
    - ligand_id
    - score column (e.g. confidence)
    - label column (e.g. 1 for active, 0 for decoy)

Typical use
-----------
python compute_pseudo_efb.py \
  --dude_root ./dataset/DUD-E/dude_raw \
  --eval_subdir diffdock_2 \
  --score_csv_name diffdock_scores_rank1.csv \
  --score_col confidence \
  --label_col label \
  --active_label 1 \
  --decoy_label 0 \
  --higher_is_better \
  --out_summary_csv ./dataset/DUD-E/dude_raw/metrics_summary_all_diffdock_2.csv \
  --save_curves

Notes
-----
1. This script updates the summary CSV by target.
2. If the summary CSV does not exist, it creates a new one.
3. If a target score CSV is missing or invalid, that target is skipped.
4. Use --higher_is_better for confidence-like scores.
   Omit it for energy-like scores where lower is better.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Core pseudo-EFB routines
# =========================

@dataclass
class PseudoEFBResult:
    curve: pd.DataFrame
    pefb_max: float
    pefb_max_chi: float
    pefb_max_threshold: float


def _validate_binary_groups(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    active_label,
    decoy_label,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract active and decoy score vectors.
    """
    use_df = df[[score_col, label_col]].dropna().copy()

    active_scores = use_df.loc[use_df[label_col] == active_label, score_col].to_numpy()
    decoy_scores = use_df.loc[use_df[label_col] == decoy_label, score_col].to_numpy()

    if len(active_scores) == 0:
        raise ValueError("No active scores found.")
    if len(decoy_scores) == 0:
        raise ValueError("No decoy scores found.")

    return active_scores.astype(float), decoy_scores.astype(float)


def compute_pseudo_efb_curve(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    active_label=1,
    decoy_label=0,
    higher_is_better: bool = True,
) -> PseudoEFBResult:
    """
    Compute the full pseudo-EFB curve by sweeping thresholds over decoy scores.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain score_col and label_col.
    score_col : str
        Name of screening score column.
    label_col : str
        Name of label column.
    active_label :
        Label value for actives.
    decoy_label :
        Label value for decoys.
    higher_is_better : bool
        True for confidence-like scores, False for energy-like scores.

    Returns
    -------
    PseudoEFBResult
    """
    active_scores, decoy_scores = _validate_binary_groups(
        df=df,
        score_col=score_col,
        label_col=label_col,
        active_label=active_label,
        decoy_label=decoy_label,
    )

    n_a = len(active_scores)
    n_d = len(decoy_scores)

    if higher_is_better:
        decoy_sorted = np.sort(decoy_scores)[::-1]
    else:
        decoy_sorted = np.sort(decoy_scores)

    rows: List[Dict] = []

    for k in range(1, n_d + 1):
        thr = decoy_sorted[k - 1]

        if higher_is_better:
            n_active_hits = int(np.sum(active_scores >= thr))
            n_decoy_hits = int(np.sum(decoy_scores >= thr))
        else:
            n_active_hits = int(np.sum(active_scores <= thr))
            n_decoy_hits = int(np.sum(decoy_scores <= thr))

        if n_decoy_hits == 0:
            continue

        p_active = n_active_hits / n_a
        p_decoy = n_decoy_hits / n_d
        pefb = p_active / p_decoy if p_decoy > 0 else np.nan
        chi = k / n_d

        rows.append(
            {
                "k_decoy": k,
                "chi": chi,
                "threshold": float(thr),
                "n_active_hits": n_active_hits,
                "n_decoy_hits": n_decoy_hits,
                "p_active": p_active,
                "p_decoy": p_decoy,
                "pEFB": pefb,
            }
        )

    curve = pd.DataFrame(rows)
    if curve.empty:
        raise ValueError("Pseudo-EFB curve is empty.")

    idx = curve["pEFB"].idxmax()
    return PseudoEFBResult(
        curve=curve,
        pefb_max=float(curve.loc[idx, "pEFB"]),
        pefb_max_chi=float(curve.loc[idx, "chi"]),
        pefb_max_threshold=float(curve.loc[idx, "threshold"]),
    )


def get_pefb_at_target_chi(curve: pd.DataFrame, target_chi: float) -> float:
    """
    Return pEFB at the nearest available chi point.
    """
    idx = (curve["chi"] - target_chi).abs().idxmin()
    return float(curve.loc[idx, "pEFB"])


# =========================
# File/path helpers
# =========================

def list_targets(dude_root: Path) -> List[str]:
    """
    Return immediate child directories under dude_root as targets.
    """
    targets = []
    for p in sorted(dude_root.iterdir()):
        if p.is_dir():
            # Skip global helper dirs if needed
            if p.name.startswith("."):
                continue
            targets.append(p.name)
    return targets


def resolve_score_csv(
    dude_root: Path,
    target: str,
    eval_subdir: str,
    score_csv_name: str,
) -> Path:
    """
    Default expected location:
        <dude_root>/<target>/eval/<eval_subdir>/<score_csv_name>
    """
    return dude_root / target / "eval" / eval_subdir / score_csv_name


def resolve_curve_out_csv(
    dude_root: Path,
    target: str,
    eval_subdir: str,
    curve_csv_name: str,
) -> Path:
    """
    Default curve output location:
        <dude_root>/<target>/eval/<eval_subdir>/<curve_csv_name>
    """
    return dude_root / target / "eval" / eval_subdir / curve_csv_name


# =========================
# Summary merge/update
# =========================

PSEUDO_EFB_COLUMNS = [
    "pEFB_1pct",
    "pEFB_5pct",
    "pEFB_10pct",
    "pEFB_max",
    "pEFB_max_chi",
    "pEFB_max_threshold",
]


def update_summary_csv(
    out_summary_csv: Path,
    new_rows_df: pd.DataFrame,
) -> None:
    """
    Update or create summary CSV.
    Merge by target. Existing pseudo-EFB columns are replaced.
    Other columns are preserved.
    """
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_summary_csv.exists():
        base_df = pd.read_csv(out_summary_csv)
    else:
        base_df = pd.DataFrame(columns=["target"])

    if "target" not in base_df.columns:
        raise ValueError(f"'target' column not found in summary CSV: {out_summary_csv}")

    # Remove old pseudo-EFB columns if present to avoid duplicate columns after merge
    cols_to_drop = [c for c in PSEUDO_EFB_COLUMNS if c in base_df.columns]
    if cols_to_drop:
        base_df = base_df.drop(columns=cols_to_drop)

    merged = base_df.merge(new_rows_df, on="target", how="outer")

    # Prefer keeping a sensible ordering: target first, then existing cols, then new pseudo-EFB cols
    ordered_cols = ["target"]
    ordered_cols += [c for c in merged.columns if c != "target" and c not in PSEUDO_EFB_COLUMNS]
    ordered_cols += [c for c in PSEUDO_EFB_COLUMNS if c in merged.columns]

    merged = merged[ordered_cols]
    merged.to_csv(out_summary_csv, index=False)


# =========================
# Main target loop
# =========================

def process_target(
    target: str,
    score_csv_path: Path,
    score_col: str,
    label_col: str,
    active_label,
    decoy_label,
    higher_is_better: bool,
    save_curves: bool,
    curve_out_csv: Optional[Path],
) -> Optional[Dict]:
    """
    Process one target score CSV and return summary row.
    """
    if not score_csv_path.exists():
        print(f"[WARN] Missing score CSV for {target}: {score_csv_path}")
        return None

    try:
        df = pd.read_csv(score_csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read CSV for {target}: {score_csv_path} ({e})")
        return None

    missing_cols = [c for c in [score_col, label_col] if c not in df.columns]
    if missing_cols:
        print(f"[WARN] {target}: missing required columns {missing_cols} in {score_csv_path}")
        return None

    try:
        result = compute_pseudo_efb_curve(
            df=df,
            score_col=score_col,
            label_col=label_col,
            active_label=active_label,
            decoy_label=decoy_label,
            higher_is_better=higher_is_better,
        )
    except Exception as e:
        print(f"[WARN] Failed pseudo-EFB computation for {target}: {e}")
        return None

    curve = result.curve

    row = {
        "target": target,
        "pEFB_1pct": get_pefb_at_target_chi(curve, 0.01),
        "pEFB_5pct": get_pefb_at_target_chi(curve, 0.05),
        "pEFB_10pct": get_pefb_at_target_chi(curve, 0.10),
        "pEFB_max": result.pefb_max,
        "pEFB_max_chi": result.pefb_max_chi,
        "pEFB_max_threshold": result.pefb_max_threshold,
    }

    if save_curves and curve_out_csv is not None:
        curve_out_csv.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(curve_out_csv, index=False)

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Compute pseudo-EFB metrics from active/decoy score CSVs and update metrics summary CSV."
    )

    parser.add_argument(
        "--dude_root",
        type=str,
        required=True,
        help="Root directory containing target subdirectories.",
    )
    parser.add_argument(
        "--eval_subdir",
        type=str,
        required=True,
        help="Evaluation subdir name, e.g. diffdock_2",
    )
    parser.add_argument(
        "--score_csv_name",
        type=str,
        default="diffdock_scores_rank1.csv",
        help="Per-target score CSV filename under <target>/eval/<eval_subdir>/",
    )
    parser.add_argument(
        "--curve_csv_name",
        type=str,
        default="pseudo_efb_curve.csv",
        help="Per-target pseudo-EFB curve CSV filename.",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="confidence",
        help="Score column name in per-target score CSV.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Label column name in per-target score CSV.",
    )
    parser.add_argument(
        "--active_label",
        type=str,
        default="1",
        help="Label value for actives. Comparison is string-safe after normalization.",
    )
    parser.add_argument(
        "--decoy_label",
        type=str,
        default="0",
        help="Label value for decoys. Comparison is string-safe after normalization.",
    )
    parser.add_argument(
        "--higher_is_better",
        action="store_true",
        help="Use this flag if larger score means better rank, e.g. confidence.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit target list. If omitted, targets are inferred from dude_root subdirs.",
    )
    parser.add_argument(
        "--out_summary_csv",
        type=str,
        required=True,
        help="Path to metrics_summary_all_<eval_subdir>.csv to create/update.",
    )
    parser.add_argument(
        "--save_curves",
        action="store_true",
        help="If set, save per-target pseudo-EFB curve CSVs.",
    )

    args = parser.parse_args()

    dude_root = Path(args.dude_root)
    out_summary_csv = Path(args.out_summary_csv)

    if not dude_root.exists():
        raise FileNotFoundError(f"dude_root does not exist: {dude_root}")

    if args.targets is None or len(args.targets) == 0:
        targets = list_targets(dude_root)
    else:
        targets = list(args.targets)

    # Normalize labels to robust string comparison first, then cast back if possible
    active_label_raw = args.active_label
    decoy_label_raw = args.decoy_label

    # We will normalize the input df label column to string before comparison,
    # so these can safely remain strings.
    active_label = str(active_label_raw)
    decoy_label = str(decoy_label_raw)

    summary_rows = []

    for target in targets:
        score_csv_path = resolve_score_csv(
            dude_root=dude_root,
            target=target,
            eval_subdir=args.eval_subdir,
            score_csv_name=args.score_csv_name,
        )

        curve_out_csv = None
        if args.save_curves:
            curve_out_csv = resolve_curve_out_csv(
                dude_root=dude_root,
                target=target,
                eval_subdir=args.eval_subdir,
                curve_csv_name=args.curve_csv_name,
            )

        # Read once to normalize label column as string before compute
        if not score_csv_path.exists():
            print(f"[WARN] Missing score CSV for {target}: {score_csv_path}")
            continue

        try:
            raw_df = pd.read_csv(score_csv_path)
        except Exception as e:
            print(f"[WARN] Failed to read CSV for {target}: {score_csv_path} ({e})")
            continue

        if args.label_col not in raw_df.columns:
            print(f"[WARN] {target}: missing label_col={args.label_col}")
            continue

        # Normalize label values to string for safe matching
        raw_df[args.label_col] = raw_df[args.label_col].astype(str)

        row = process_target(
            target=target,
            score_csv_path=score_csv_path,
            score_col=args.score_col,
            label_col=args.label_col,
            active_label=active_label,
            decoy_label=decoy_label,
            higher_is_better=args.higher_is_better,
            save_curves=args.save_curves,
            curve_out_csv=curve_out_csv,
        )

        # process_target reads from file again; to keep code simpler, patch by recomputing
        # directly from normalized raw_df so label matching is robust.
        if row is None:
            try:
                result = compute_pseudo_efb_curve(
                    df=raw_df,
                    score_col=args.score_col,
                    label_col=args.label_col,
                    active_label=active_label,
                    decoy_label=decoy_label,
                    higher_is_better=args.higher_is_better,
                )
                curve = result.curve

                row = {
                    "target": target,
                    "pEFB_1pct": get_pefb_at_target_chi(curve, 0.01),
                    "pEFB_5pct": get_pefb_at_target_chi(curve, 0.05),
                    "pEFB_10pct": get_pefb_at_target_chi(curve, 0.10),
                    "pEFB_max": result.pefb_max,
                    "pEFB_max_chi": result.pefb_max_chi,
                    "pEFB_max_threshold": result.pefb_max_threshold,
                }

                if args.save_curves and curve_out_csv is not None:
                    curve_out_csv.parent.mkdir(parents=True, exist_ok=True)
                    curve.to_csv(curve_out_csv, index=False)
            except Exception as e:
                print(f"[WARN] Failed pseudo-EFB computation for {target}: {e}")
                continue

        print(
            f"[INFO] {target}: "
            f"pEFB_1pct={row['pEFB_1pct']:.4f}, "
            f"pEFB_max={row['pEFB_max']:.4f}, "
            f"pEFB_max_chi={row['pEFB_max_chi']:.6f}"
        )
        summary_rows.append(row)

    if not summary_rows:
        print("[WARN] No valid targets processed. Summary CSV not updated.")
        return

    new_rows_df = pd.DataFrame(summary_rows)
    update_summary_csv(out_summary_csv=out_summary_csv, new_rows_df=new_rows_df)

    print(f"[INFO] Updated summary CSV: {out_summary_csv}")


if __name__ == "__main__":
    main()
    
'''
python ./dataset/DUD-E/scripts_2/eval/compute_pseudo_efb.py \
  --dude_root ./dataset/DUD-E/dude_raw \
  --eval_subdir diffdock_2 \
  --score_csv_name diffdock_scores_rank1.csv \
  --score_col score \
  --label_col label \
  --active_label 1 \
  --decoy_label 0 \
  --higher_is_better \
  --out_summary_csv ./dataset/DUD-E/dude_raw/metrics_summary_all_diffdock_2.csv \
  --save_curves
'''
