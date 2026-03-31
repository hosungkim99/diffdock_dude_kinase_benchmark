#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# repo 루트 import (repo/scripts/eval 기준 2단계 위)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.metrics.dude_metrics import evaluate_from_scores_csv, save_outputs


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--outdir", default="", help="If set, write metrics.json/roc.csv/ranking.csv/config.json here.")

    ap.add_argument("--missing_policy", choices=["drop", "bottom"], default="drop",
                    help="How to handle missing scores: drop (default) or bottom (-inf).")

    ap.add_argument("--alpha_logauc", type=float, default=0.1)
    ap.add_argument("--alpha_bedroc", type=float, default=20.0)

    ap.add_argument("--dude_fpr_min", type=float, default=0.001)
    ap.add_argument("--dude_random_logauc_pct", type=float, default=14.462)
    ap.add_argument("--dude_ef_fpr", type=float, default=0.01)

    args = ap.parse_args()

    metrics, ex = evaluate_from_scores_csv(
        args.scores_csv,
        missing_policy=args.missing_policy,
        alpha_logauc=args.alpha_logauc,
        alpha_bedroc=args.alpha_bedroc,
        dude_fpr_min=args.dude_fpr_min,
        dude_random_logauc_pct=args.dude_random_logauc_pct,
        dude_ef_fpr=args.dude_ef_fpr,
    )

    # 원본 출력 포맷 그대로 유지
    print("=== Input ===")
    print(f"Scores CSV      : {args.scores_csv}")
    print(f"Used ligands    : {len(ex['y'])} (missing score total: {ex['missing']}, policy: {args.missing_policy})")
    print(f"Actives         : {int((ex['y']==1).sum())}, Decoys: {int((ex['y']==0).sum())}")
    print(f"Base rate       : {ex['base_rate']:.6f}")
    print()

    print("=== Metrics (higher is better) ===")
    print(f"ROC-AUC                     : {ex['auc']:.4f}")
    print()

    print("=== DUD-E style (paper) ===")
    print(f"Adjusted LogAUC@FPR>={args.dude_fpr_min:g} : {ex['dude_logauc']:.3f}  (random baseline subtracted: {args.dude_random_logauc_pct:.3f}%)")
    print(f"ROC-EF@FPR={args.dude_ef_fpr*100:.1f}%            : {ex['dude_ef1_pct']:.2f}% actives found when {args.dude_ef_fpr*100:.1f}% decoys found")
    print()

    print("=== Extra (top-fraction / legacy outputs) ===")
    print(f"EF@1% (top 1% screened)      : {ex['ef1']:.3f}  (hits {ex['hits1']}/{ex['k1']})")
    print(f"EF@5% (top 5% screened)      : {ex['ef5']:.3f}  (hits {ex['hits5']}/{ex['k5']})")
    print(f"EF@10% (top 10% screened)    : {ex['ef10']:.3f} (hits {ex['hits10']}/{ex['k10']})")
    print(f"nEF@1%                       : {ex['nef1']:.4f}")
    print(f"nEF@5%                       : {ex['nef5']:.4f}")
    print(f"nEF@10%                      : {ex['nef10']:.4f}")
    print(f"LogAUC@{args.alpha_logauc*100:.1f}% (rank/N)        : {ex['la_old']:.4f}")
    print(f"BEDROC(alpha={args.alpha_bedroc})         : {ex['bd']:.4f}")
    print()

    if args.outdir.strip() != "":
        outdir = Path(args.outdir)
        save_outputs(outdir, args, ex["ligand_id"], ex["y"], ex["s"], ex["fpr"], ex["tpr"], metrics)
        print(f"[Saved] metrics.json, roc.csv, ranking.csv, config.json -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
    

'''
TARGET="wee1"
cd ./dataset/DUD-E
python ./dataset/DUD-E/scripts_2/eval/eval_dude_metrics.py \
  --scores_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/diffdock_scores_rank1.csv \
  --outdir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/metrics_rank1 \
  --dude_fpr_min 0.001 \
  --dude_ef_fpr 0.01 \
  --alpha_logauc 0.1 \
  --alpha_bedroc 20.0 \
  --missing_policy bottom
'''
