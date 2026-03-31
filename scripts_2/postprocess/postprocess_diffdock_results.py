#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# repo 루트 import (repo/scripts/postprocess 기준 2단계 위)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.qc.postprocess import (
    read_csv_rows,
    scan_best_rank_conf,
    write_list_txt,
    write_retry_csv,
    write_scores_ok_csv,
)


def main():
    p = argparse.ArgumentParser(
        description="Split-level postprocess: classify ok/missing/retry and write ok-only score table."
    )
    p.add_argument("--target", required=True, help="target name, e.g., abl1")
    p.add_argument("--split", required=True, choices=["actives", "decoys"])
    p.add_argument("--csv", required=True, type=Path, help="input {target}_{split}.csv path")
    p.add_argument("--results_dir", required=True, type=Path, help="results/{split} directory")
    p.add_argument("--out_dir", default=None, type=Path, help="output directory (default: csv parent)")
    p.add_argument("--prefix", default=None, help="output filename prefix (default: {target}_{split})")
    p.add_argument("--max_rank", type=int, default=10)
    args = p.parse_args()

    csv_path: Path = args.csv
    results_dir: Path = args.results_dir
    out_dir: Path = args.out_dir if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix else f"{args.target}_{args.split}"
    label = 1 if args.split == "actives" else 0

    complex_names, all_rows, idx_cname = read_csv_rows(csv_path)
    all_unique = sorted(set(complex_names))

    best_all = scan_best_rank_conf(results_dir, max_rank=args.max_rank)
    ok_best = {c: best_all[c] for c in all_unique if c in best_all}
    ok_list = sorted(ok_best.keys())
    missing_list = sorted([c for c in all_unique if c not in ok_best])
    missing_set = set(missing_list)

    # outputs
    all_txt = out_dir / f"{prefix}_all.txt"
    ok_txt = out_dir / f"{prefix}_ok.txt"
    missing_txt = out_dir / f"{prefix}_missing.txt"
    retry_csv = out_dir / f"{prefix}_retry.csv"
    scores_ok_csv = out_dir / f"{prefix}_scores_ok.csv"

    write_list_txt(all_txt, all_unique)
    write_list_txt(ok_txt, ok_list)
    write_list_txt(missing_txt, missing_list)

    n_retry = write_retry_csv(retry_csv, all_rows, idx_cname, missing_set)
    n_scores = write_scores_ok_csv(scores_ok_csv, ok_best, label)

    print("=== DiffDock Postprocess Summary ===")
    print(f"target={args.target} split={args.split}")
    print(f"unique complexes in CSV : {len(all_unique)}")
    print(f"ok (rankK found)        : {len(ok_list)}")
    print(f"missing (no rankK)      : {len(missing_list)}")
    print(f"retry.csv rows          : {n_retry}")
    print(f"scores_ok.csv rows      : {n_scores}")
    print(f"Wrote: {scores_ok_csv}")


if __name__ == "__main__":
    main()

'''
실행 예시
TARGET="abl1"
SPLIT="actives"
cd ./dataset/DUD-E
python ./dataset/DUD-E/scripts_2/postprocess/postprocess_diffdock_results.py \
  --target $TARGET \
  --split $SPLIT \
  --csv ./dataset/DUD-E/dude_raw/$TARGET/$TARGET"_"$SPLIT".csv" \
  --results_dir ./dataset/DUD-E/dude_raw/$TARGET/results/$SPLIT \
  --out_dir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/$SPLIT \
  --max_rank 10

'''
