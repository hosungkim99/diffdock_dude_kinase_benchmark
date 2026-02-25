#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

# repo root에서 실행한다고 가정하면 src import가 되지만,
# 환경에 따라 안 되면 PYTHONPATH를 DUD-E 루트로 잡아야 한다.
# 예: export PYTHONPATH=/home/deepfold/users/hosung/dataset/DUD-E:$PYTHONPATH
from src.aggregate.master import build_master_table, MasterTableConfig


def main():
    ap = argparse.ArgumentParser(description="Build per-target master_table.csv from score table + QC + err status + COMdist")

    ap.add_argument("--dude_root", required=True, help="e.g. /home/.../dataset/DUD-E/dude_raw")
    ap.add_argument("--target", required=True, help="e.g. wee1")
    ap.add_argument("--scores_csv", required=True, help="e.g. .../eval/diffdock_2/diffdock_scores_rank1.csv")
    ap.add_argument("--out_csv", required=True, help="e.g. .../eval/diffdock_2/master_table.csv")

    ap.add_argument("--comdist_csv", default="", help="optional: .../eval/diffdock_2/COM/comdist_all.csv")
    ap.add_argument("--err_status_csv", default="", help="optional: .../eval/diffdock_2/inference_status_err.csv")

    ap.add_argument("--receptor_pdb", default="", help="optional override (default: <target>/receptor.pdb)")
    ap.add_argument("--crystal_ligand_mol2", default="", help="optional override (default: <target>/crystal_ligand.mol2)")
    ap.add_argument("--cache_qc_csv", default="", help="optional cache for QC features")

    # thresholds override (필요한 것만 노출)
    ap.add_argument("--pocket_radius_A", type=float, default=6.0)
    ap.add_argument("--clash_cutoff_A", type=float, default=2.0)

    # pocket_in rule override (원하면 추가로 더 열어도 됨)
    ap.add_argument("--in_dcenter_A", type=float, default=8.0)
    ap.add_argument("--in_dmin_A", type=float, default=4.0)
    ap.add_argument("--in_contacts_ge", type=int, default=10)

    args = ap.parse_args()

    cfg = MasterTableConfig(
        pocket_radius_A=float(args.pocket_radius_A),
        clash_cutoff_A=float(args.clash_cutoff_A),
        in_dcenter_A=float(args.in_dcenter_A),
        in_dmin_A=float(args.in_dmin_A),
        in_contacts_ge=int(args.in_contacts_ge),
    )

    build_master_table(
        dude_root=Path(args.dude_root),
        target=args.target,
        scores_csv=Path(args.scores_csv),
        out_csv=Path(args.out_csv),
        cfg=cfg,
        comdist_csv=Path(args.comdist_csv) if args.comdist_csv.strip() else None,
        err_status_csv=Path(args.err_status_csv) if args.err_status_csv.strip() else None,
        receptor_pdb=Path(args.receptor_pdb) if args.receptor_pdb.strip() else None,
        crystal_ligand_mol2=Path(args.crystal_ligand_mol2) if args.crystal_ligand_mol2.strip() else None,
        cache_qc_csv=Path(args.cache_qc_csv) if args.cache_qc_csv.strip() else None,
    )


if __name__ == "__main__":
    main()
    
    
'''
cd /home/deepfold/users/hosung/dataset/DUD-E
export PYTHONPATH=/home/deepfold/users/hosung/dataset/DUD-E:$PYTHONPATH

DUDE_ROOT=/home/deepfold/users/hosung/dataset/DUD-E/dude_raw
TARGET="wee1"
EVAL=$DUDE_ROOT/$TARGET/eval/diffdock_2

python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/aggregate/build_master_table.py \
  --dude_root $DUDE_ROOT \
  --target $TARGET \
  --scores_csv $EVAL/diffdock_scores_rank1.csv \
  --err_status_csv $EVAL/inference_status_err.csv \
  --comdist_csv $EVAL/COM/comdist_all.csv \
  --out_csv $EVAL/master_table.csv \
  --cache_qc_csv $EVAL/_cache_master_qc.csv
'''