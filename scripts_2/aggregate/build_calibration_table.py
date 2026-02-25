#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# repo root import
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.aggregate.calibration import CalibrationConfig, build_calibration_tables_from_master


CALIB_COLUMN_ORDER = [
    "target",
    "calib_type",          # "label" or "pose"
    "bin_id",
    "bin_low",
    "bin_high",
    "n",
    "mean_conf",
    "empirical_rate",
    "abs_gap",
]

SUMMARY_COLUMN_ORDER = [
    "target",
    "n_total_success",
    "label_n",
    "label_ece",
    "label_mce",
    "label_brier",
    "pose_n",
    "pose_ece",
    "pose_mce",
    "pose_brier",
    "n_bins",
    "binning",
    "pose_cutoff_A",
    "require_success",
]


def _parse_targets_arg(s: str) -> Optional[List[str]]:
    t = s.strip()
    if not t:
        return None
    raw = t.replace(",", " ").split()
    out = [x.strip() for x in raw if x.strip()]
    # stable unique
    out = list(dict.fromkeys(out))
    out.sort()
    return out


def _discover_targets(dude_root: Path, eval_subdir: str) -> List[str]:
    out = []
    for tdir in sorted(dude_root.iterdir()):
        if not tdir.is_dir():
            continue
        master_csv = tdir / "eval" / eval_subdir / "master_table.csv"
        if master_csv.exists():
            out.append(tdir.name)
    return out


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Build calibration tables (label + pose) from master_table.csv")
    ap.add_argument("--dude_root", required=True, type=Path,
                    help="e.g. /home/.../dataset/DUD-E/dude_raw")
    ap.add_argument("--eval_subdir", required=True, type=str,
                    help="e.g. diffdock_2")
    ap.add_argument("--targets", default="", type=str,
                    help="optional: comma/space separated targets. If empty, auto-detect by master_table.csv existence")

    ap.add_argument("--out_csv", required=True, type=Path,
                    help="combined calibration table CSV (bins x targets x types)")
    ap.add_argument("--out_summary_csv", default=None, type=Path,
                    help="optional: per-target summary metrics (ECE/MCE/Brier)")

    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--binning", choices=["uniform", "quantile"], default="uniform")
    ap.add_argument("--pose_cutoff_A", type=float, default=2.0)

    ap.add_argument("--require_success", action="store_true",
                    help="if set, filter success==1 before calibration (recommended)")
    ap.set_defaults(require_success=True)

    ap.add_argument("--skipped_csv", type=Path, default=None)
    ap.add_argument("--errors_csv", type=Path, default=None)

    args = ap.parse_args()

    dude_root: Path = args.dude_root
    eval_subdir: str = args.eval_subdir

    requested = _parse_targets_arg(args.targets)
    if requested is None:
        targets = _discover_targets(dude_root, eval_subdir)
    else:
        targets = requested

    cfg = CalibrationConfig(
        n_bins=int(args.n_bins),
        binning=str(args.binning),
        pose_cutoff_A=float(args.pose_cutoff_A),
        require_success=bool(args.require_success),
    )

    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for target in targets:
        base = dude_root / target / "eval" / eval_subdir
        master_csv = base / "master_table.csv"
        if not master_csv.exists():
            skipped.append({
                "target": target,
                "reason": "missing_master_table",
                "path": str(master_csv),
            })
            continue

        try:
            master = pd.read_csv(master_csv)
            out = build_calibration_tables_from_master(master, cfg=cfg)

            # label table
            lt: pd.DataFrame = out["label_table"]
            lt = lt.copy()
            lt.insert(0, "calib_type", "label")
            lt.insert(0, "target", target)

            # pose table
            pt: pd.DataFrame = out["pose_table"]
            pt = pt.copy()
            pt.insert(0, "calib_type", "pose")
            pt.insert(0, "target", target)

            # append rows
            for df_t in (lt, pt):
                # enforce schema/order
                for c in CALIB_COLUMN_ORDER:
                    if c not in df_t.columns:
                        df_t[c] = np.nan
                df_t = df_t[CALIB_COLUMN_ORDER]
                rows.extend(df_t.to_dict(orient="records"))

            # summary
            summ = out["summary"]
            srow = {
                "target": target,
                "n_total_success": int(summ.get("n_total_success", 0)),
                "label_n": int(summ["label"]["n"]),
                "label_ece": float(summ["label"]["ece"]),
                "label_mce": float(summ["label"]["mce"]),
                "label_brier": float(summ["label"]["brier"]),
                "pose_n": int(summ["pose"]["n"]),
                "pose_ece": float(summ["pose"]["ece"]),
                "pose_mce": float(summ["pose"]["mce"]),
                "pose_brier": float(summ["pose"]["brier"]),
                "n_bins": int(summ["config"]["n_bins"]),
                "binning": str(summ["config"]["binning"]),
                "pose_cutoff_A": float(summ["config"]["pose_cutoff_A"]),
                "require_success": bool(summ["config"]["require_success"]),
            }
            summaries.append(srow)

        except Exception as e:
            errors.append({
                "target": target,
                "reason": type(e).__name__,
                "error": str(e),
                "master_table_csv": str(master_csv),
            })
            continue

    out_df = pd.DataFrame(rows, columns=CALIB_COLUMN_ORDER)
    _ensure_parent(args.out_csv)
    out_df.to_csv(args.out_csv, index=False)

    if args.out_summary_csv is not None:
        sum_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMN_ORDER)
        _ensure_parent(args.out_summary_csv)
        sum_df.to_csv(args.out_summary_csv, index=False)

    if args.skipped_csv is not None:
        _ensure_parent(args.skipped_csv)
        pd.DataFrame(skipped).to_csv(args.skipped_csv, index=False)

    if args.errors_csv is not None:
        _ensure_parent(args.errors_csv)
        pd.DataFrame(errors).to_csv(args.errors_csv, index=False)

    print("=== build_calibration_table done ===")
    print(f"targets_requested : {len(targets)}")
    print(f"targets_succeeded : {len(set([r['target'] for r in rows]))}")
    print(f"targets_skipped   : {len(skipped)}")
    print(f"targets_errored   : {len(errors)}")
    print(f"saved_table       : {args.out_csv}")
    if args.out_summary_csv is not None:
        print(f"saved_summary     : {args.out_summary_csv}")


if __name__ == "__main__":
    main()
    
'''
cd /home/deepfold/users/hosung/dataset/DUD-E
TARGET="wee1"
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/aggregate/build_calibration_table.py \
  --dude_root /home/deepfold/users/hosung/dataset/DUD-E/dude_raw \
  --eval_subdir diffdock_2 \
  --targets $TARGET \
  --n_bins 20 \
  --binning uniform \
  --pose_cutoff_A 2.0 \
  --out_csv /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/calibration_table_diffdock_2.csv \
  --out_summary_csv /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/calibration_summary_diffdock_2.csv
'''