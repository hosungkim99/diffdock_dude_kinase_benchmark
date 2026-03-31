#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.qc.sanity import run_sanity_check


def main():
    p = argparse.ArgumentParser(description="Split-agnostic structure sanity check for DiffDock rank1 poses.")
    p.add_argument("--target", required=True)
    p.add_argument("--split", required=True, choices=["actives", "decoys"])
    p.add_argument("--receptor_pdb", required=True, type=Path)
    p.add_argument("--results_dir", required=True, type=Path, help=".../results/{split}")
    p.add_argument("--scores_ok_csv", required=True, type=Path, help=".../{target}_{split}_scores_ok.csv")
    p.add_argument("--out_dir", required=True, type=Path)

    p.add_argument("--pocket_out_min_dist", type=float, default=6.0)
    p.add_argument("--clash_threshold", type=float, default=1.0)
    p.add_argument("--clash_pairs_flag", type=int, default=50)
    args = p.parse_args()

    summary = run_sanity_check(
        target=args.target,
        split=args.split,
        receptor_pdb=args.receptor_pdb,
        results_dir=args.results_dir,
        scores_ok_csv=args.scores_ok_csv,
        out_dir=args.out_dir,
        pocket_out_min_dist=args.pocket_out_min_dist,
        clash_threshold=args.clash_threshold,
        clash_pairs_flag=args.clash_pairs_flag,
    )

    print("=== Sanity Check Done ===")
    print(f"target={args.target} split={args.split}")
    print(f"summary_csv={summary}")


if __name__ == "__main__":
    main()

'''
실행 예시
- actives 
TARGET="abl1"
cd ./dataset/DUD-E

python ./dataset/DUD-E/scripts_2/qc/sanity_check.py \
  --target $TARGET --split actives \
  --receptor_pdb ./dataset/DUD-E/dude_raw/$TARGET/receptor.pdb \
  --results_dir  ./dataset/DUD-E/dude_raw/$TARGET/results/actives \
  --scores_ok_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/actives/$TARGET"_actives_scores_ok.csv" \
  --out_dir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/sanity/actives

- decoys
TARGET="abl1"
cd ./dataset/DUD-E

python ./dataset/DUD-E/scripts_2/qc/sanity_check.py \
  --target $TARGET --split decoys \
  --receptor_pdb ./dataset/DUD-E/dude_raw/$TARGET/receptor.pdb \
  --results_dir  ./dataset/DUD-E/dude_raw/$TARGET/results/decoys \
  --scores_ok_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/decoys/$TARGET"_decoys_scores_ok.csv" \
  --out_dir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/sanity/decoys

'''
