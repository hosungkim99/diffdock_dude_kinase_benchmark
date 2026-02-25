#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.qc.rank1_bundle import run_rank1_qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--metrics_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    scores_csv = Path(args.scores_csv)
    metrics_dir = Path(args.metrics_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qc = run_rank1_qc(scores_csv, metrics_dir, topk=args.topk)

    # JSON 저장
    (outdir / "qc_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    # Human-readable 요약
    report_lines = []
    report_lines.append(f"Base rate: {qc['base_rate_active']:.4f}")
    report_lines.append(f"Top{qc['topk']} active rate: {qc['topk_active_rate']:.4f}")
    report_lines.append(f"Enrichment x{qc['topk_enrichment_over_base']:.2f}")
    report_lines.append(f"Duplicates: {qc['n_duplicates_scores_csv']}")
    report_lines.append(f"-inf count: {qc['n_neg_inf_in_ranking']}")
    report_lines.append(f"ROC monotonic: {qc['roc_monotonic_fpr']}/{qc['roc_monotonic_tpr']}")
    report_lines.append(f"ROC-AUC: {qc['metrics_snapshot']['roc_auc']:.4f}")
    report_lines.append(f"Adj LogAUC0.001: {qc['metrics_snapshot']['adjusted_logauc0.001']:.3f}")

    (outdir / "qc_summary.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("[QC] wrote:")
    print(" ", outdir / "qc_summary.json")
    print(" ", outdir / "qc_summary.txt")


if __name__ == "__main__":
    main()

'''
TARGET="vgfr2"
cd /home/deepfold/users/hosung/dataset/DUD-E
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/eval/qc_rank1_bundle.py \
  --scores_csv /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/diffdock_scores_rank1.csv \
  --metrics_dir /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/metrics_rank1 \
  --outdir /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/qc_rank1_bundle \
  --topk 100

'''