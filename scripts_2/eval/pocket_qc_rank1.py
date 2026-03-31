#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.qc.pocket_rank1 import run_pocket_qc_rank1, PocketQCThresholds


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--target_dir", required=True, help=".../dude_raw/abl1")
    ap.add_argument("--outdir", required=True)

    ap.add_argument(
        "--mode",
        choices=["mixed", "actives_only"],
        default="mixed",
        help="mixed: TopK from ranking_csv; actives_only: TopK from actives_scores_ok_csv (confidence)",
    )

    # Mixed inputs
    ap.add_argument("--ranking_csv", default="", help="metrics_rank1/ranking.csv (preferred for mixed)")
    ap.add_argument("--scores_csv", default="", help="diffdock_scores_rank1.csv (fallback for mixed)")

    # Active-only input
    ap.add_argument(
        "--actives_scores_ok_csv",
        default="",
        help="*_actives_scores_ok.csv with columns: complex_name,confidence,label,... (required for actives_only)",
    )

    # results root (NOT split dir)
    ap.add_argument(
        "--results_root",
        default="",
        help="results root dir (default: target_dir/results). Must contain actives/ and decoys/ for mixed.",
    )

    ap.add_argument("--topk", type=int, default=100)

    ap.add_argument("--pocket_ligand", default="", help="crystal_ligand.mol2 (default: target_dir/crystal_ligand.mol2)")
    ap.add_argument("--receptor_pdb", default="", help="receptor.pdb (default: target_dir/receptor.pdb)")

    ap.add_argument("--pocket_radius", type=float, default=6.0)
    ap.add_argument("--contact_cutoff", type=float, default=4.0)
    ap.add_argument("--clash_cutoff", type=float, default=2.0)

    ap.add_argument("--in_dcenter", type=float, default=8.0)
    ap.add_argument("--in_dmin", type=float, default=4.0)
    ap.add_argument("--in_contacts", type=int, default=10)
    ap.add_argument("--out_dcenter", type=float, default=15.0)
    ap.add_argument("--out_dmin", type=float, default=8.0)

    args = ap.parse_args()

    thr = PocketQCThresholds(
        contact_cutoff=args.contact_cutoff,
        clash_cutoff=args.clash_cutoff,
        in_dcenter=args.in_dcenter,
        in_dmin=args.in_dmin,
        in_contacts=args.in_contacts,
        out_dcenter=args.out_dcenter,
        out_dmin=args.out_dmin,
    )

    summary = run_pocket_qc_rank1(
        target_dir=Path(args.target_dir),
        outdir=Path(args.outdir),
        topk=int(args.topk),
        mode=str(args.mode),
        results_root=Path(args.results_root) if args.results_root else None,
        ranking_csv=Path(args.ranking_csv) if args.ranking_csv else None,
        scores_csv=Path(args.scores_csv) if args.scores_csv else None,
        actives_scores_ok_csv=Path(args.actives_scores_ok_csv) if args.actives_scores_ok_csv else None,
        pocket_ligand=Path(args.pocket_ligand) if args.pocket_ligand else None,
        receptor_pdb=Path(args.receptor_pdb) if args.receptor_pdb else None,
        pocket_radius=float(args.pocket_radius),
        thr=thr,
    )

    print("[OK] wrote:")
    for k, v in summary.get("_outputs", {}).items():
        print(" ", k, "->", v)


if __name__ == "__main__":
    main()


"""
실행 예시

# (A) active-only QC: *_actives_scores_ok.csv 기반 Top100 (confidence)
TARGET="mp2k1"
python ./dataset/DUD-E/scripts_2/eval/pocket_qc_rank1.py \
  --target_dir ./dataset/DUD-E/dude_raw/$TARGET \
  --mode actives_only \
  --actives_scores_ok_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/actives/${TARGET}_actives_scores_ok.csv \
  --results_root ./dataset/DUD-E/dude_raw/$TARGET/results \
  --outdir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/QC_pocket_rank1_actives_only \
  --topk 100

# (B) mixed QC: metrics_rank1/ranking.csv 기반 Top100 (actives+decoys)
python ./dataset/DUD-E/scripts_2/eval/pocket_qc_rank1.py \
  --target_dir ./dataset/DUD-E/dude_raw/$TARGET \
  --mode mixed \
  --ranking_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/metrics_rank1/ranking.csv \
  --results_root ./dataset/DUD-E/dude_raw/$TARGET/results \
  --outdir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/QC_pocket_rank1_mixed \
  --topk 100
"""
