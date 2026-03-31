#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.qc.report import build_qc_report


def main():
    p = argparse.ArgumentParser(description="Split-level QC report from ok-only score table.")
    p.add_argument("--target", required=True)
    p.add_argument("--split", required=True, choices=["actives", "decoys"])
    p.add_argument("--scores_ok_csv", required=True, type=Path, help=".../{target}_{split}_scores_ok.csv")
    p.add_argument("--out_dir", required=True, type=Path)

    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--bottom_k", type=int, default=50)
    p.add_argument("--high_thr", type=float, default=0.0)
    p.add_argument("--low_thr", type=float, default=-1.5)

    args = p.parse_args()

    summary = build_qc_report(
        target=args.target,
        split=args.split,
        scores_ok_csv=args.scores_ok_csv,
        out_dir=args.out_dir,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        high_thr=args.high_thr,
        low_thr=args.low_thr,
    )

    print("=== QC Report Done ===")
    print(f"target={summary['target']} split={summary['split']}")
    print(f"n_ok={summary['n_ok']}")
    print(f"summary_txt={summary['outputs']['summary_txt']}")


if __name__ == "__main__":
    main()

'''
실행 예시
actives QC report
TARGET="abl1"
cd ./dataset/DUD-E

python ./dataset/DUD-E/scripts_2/qc/qc_report.py \
  --target $TARGET --split actives \
  --scores_ok_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/actives/$TARGET"_actives_scores_ok.csv" \
  --out_dir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/qc_report/actives \
  --top_k 50 --bottom_k 50 \
  --high_thr 0.0 --low_thr -1.5

decoys QC report
TARGET="abl1"
cd ./dataset/DUD-E

python ./dataset/DUD-E/scripts_2/qc/qc_report.py \
  --target $TARGET --split decoys \
  --scores_ok_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/decoys/$TARGET"_decoys_scores_ok.csv" \
  --out_dir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/qc_report/decoys \
  --top_k 50 --bottom_k 50 \
  --high_thr 0.0 --low_thr -1.5
'''
