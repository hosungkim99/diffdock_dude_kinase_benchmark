#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.qc.diffdock_scores import build_global_score_table


def main():
    ap = argparse.ArgumentParser(
        description="Build global DiffDock score table (actives + decoys)."
    )
    ap.add_argument("--actives_root", required=True, type=Path)
    ap.add_argument("--decoys_root", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--score_mode", choices=["rank1", "max"], default="rank1")
    ap.add_argument("--max_rank", type=int, default=10)

    args = ap.parse_args()

    rows = build_global_score_table(
        actives_root=args.actives_root,
        decoys_root=args.decoys_root,
        score_mode=args.score_mode,
        max_rank=args.max_rank,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ligand_id",
        "label",
        "score",
        "has_rank1",
        "n_ranks",
        "rank_used",
    ]

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_total = len(rows)
    n_missing = sum(1 for r in rows if r["score"] is None)

    print("=== Global Score Table Built ===")
    print(f"Total ligands : {n_total}")
    print(f"Missing score : {n_missing}")
    print(f"Saved to      : {args.out_csv}")


if __name__ == "__main__":
    main()


'''
실행 예시(rank1)
TARGET="abl1"
cd ./dataset/DUD-E
python ./dataset/DUD-E/scripts_2/postprocess/make_diffdock_score_table.py \
  --actives_root ./dataset/DUD-E/dude_raw/$TARGET/results/actives \
  --decoys_root  ./dataset/DUD-E/dude_raw/$TARGET/results/decoys \
  --out_csv      ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/diffdock_scores_rank1.csv \
  --score_mode rank1

실행 예시(max)
TARGET="abl1"
cd ./dataset/DUD-E
python ./dataset/DUD-E/scripts_2/postprocess/make_diffdock_score_table.py \
  --actives_root ./dataset/DUD-E/dude_raw/$TARGET/results/actives \
  --decoys_root  ./dataset/DUD-E/dude_raw/$TARGET/results/decoys \
  --out_csv      ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock/diffdock_scores_max.csv \
  --score_mode max
'''
