#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

# repo 루트 기준 import
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.geometry.comdist import compute_comdist_table, summarize_comdist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dude_root", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument(
        "--split",
        choices=["actives", "decoys", "both"],
        required=True,
        help="actives / decoys / both"
    )
    ap.add_argument("--cutoff_A", type=float, default=2.0)
    ap.add_argument("--results_dir", default="")
    ap.add_argument("--crystal_ligand", default="")

    args = ap.parse_args()

    dude_root = Path(args.dude_root)
    target = args.target
    split = args.split

    eval_com_dir = dude_root / target / "eval" / "diffdock_2" / "COM"
    eval_com_dir.mkdir(parents=True, exist_ok=True)

    crystal_ligand = Path(args.crystal_ligand) if args.crystal_ligand.strip() else None

    def run_one(one_split: str):
        results_dir = Path(args.results_dir) if args.results_dir.strip() else None

        df = compute_comdist_table(
            dude_root=dude_root,
            target=target,
            split=one_split,
            cutoff_A=float(args.cutoff_A),
            results_dir=results_dir,
            crystal_ligand_path=crystal_ligand,
        )

        out_csv = eval_com_dir / f"{target}_comdist_{one_split}.csv"
        df.to_csv(out_csv, index=False)

        summ = summarize_comdist(df, cutoff_A=float(args.cutoff_A))
        print(
            f"[{target} {one_split}] "
            f"N={summ['N']} "
            f"success={summ['success']} "
            f"pass@{summ['cutoff_A']}A={summ['pass_rate']:.3f} "
            f"median={summ['median_dist_A']:.3f}A"
        )
        print(f"[SAVE] {out_csv}")

        return df, out_csv

    # ---- single split ----
    if split in ("actives", "decoys"):
        run_one(split)
        return

    # ---- BOTH (actives + decoys 자동 합치기) ----
    df_a, path_a = run_one("actives")
    df_d, path_d = run_one("decoys")

    df_all = pd.concat([df_a, df_d], ignore_index=True)

    out_all = eval_com_dir / f"{target}_comdist_all.csv"
    df_all.to_csv(out_all, index=False)

    summ_all = summarize_comdist(df_all, cutoff_A=float(args.cutoff_A))
    print(
        f"[{target} all] "
        f"N={summ_all['N']} "
        f"success={summ_all['success']} "
        f"pass@{summ_all['cutoff_A']}A={summ_all['pass_rate']:.3f} "
        f"median={summ_all['median_dist_A']:.3f}A"
    )
    print(f"[SAVE] {out_all}")


if __name__ == "__main__":
    main()
    
'''
TARGET="wee1"
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/eval/compute_comdist.py \
  --dude_root /home/deepfold/users/hosung/dataset/DUD-E/dude_raw \
  --target $TARGET \
  --split both \
  --cutoff_A 2.0

'''