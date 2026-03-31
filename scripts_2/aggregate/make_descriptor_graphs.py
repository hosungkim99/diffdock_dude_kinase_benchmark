#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_descriptor_graphs.py

입력:
- descriptor_tests.csv

출력:
- global_top_descriptors.png
- pairwise_grouped_bar_signed_delta.png

설명:
1) global 그래프
   - test_type == "global" 인 row만 사용
   - descriptor별 Kruskal-Wallis p-value를
       -log10(p_value)
     로 변환해서 상위 descriptor를 barplot으로 그림

2) pairwise 그래프
   - test_type == "pairwise" 인 row만 사용
   - x축 descriptor는 사용자가 지정한 목록으로 고정
   - y축은 signed Cliff's delta
     즉 절댓값이 아니라 양수/음수 그대로 그림
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===== 사용자가 고정하고 싶은 descriptor 순서 =====
PAIRWISE_DESCRIPTOR_ORDER = [
    "rotatable_bonds",
    "chiral_center_count",
    "smiles_chiral_symbol_count",
    "ring_count",
    "aromatic_ring_count",
    "heavy_atom_count",
    "fraction_csp3",
    "tpsa",
    "hba",
    "hbd",
    "logp",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to descriptor_tests.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save graphs",
    )
    parser.add_argument(
        "--top_n_global",
        type=int,
        default=12,
        help="Number of top descriptors for global plot",
    )
    return parser.parse_args()


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def make_global_plot(df: pd.DataFrame, out_dir: Path, top_n: int = 12):
    """
    global plot:
    descriptor vs -log10(p_value)
    """
    global_df = df[df["test_type"] == "global"].copy()

    global_df["p_value"] = pd.to_numeric(global_df["p_value"], errors="coerce")
    global_df["kruskal_stat"] = pd.to_numeric(global_df["kruskal_stat"], errors="coerce")
    global_df = global_df[np.isfinite(global_df["p_value"])].copy()

    if len(global_df) == 0:
        print("[warn] No global rows found. Skip global plot.")
        return

    global_df["neg_log10_p"] = -np.log10(global_df["p_value"].clip(lower=1e-300))

    plot_df = (
        global_df.sort_values(
            by=["neg_log10_p", "kruskal_stat"],
            ascending=[False, False]
        )
        .head(top_n)
        .copy()
    )

    plot_df = plot_df.sort_values("neg_log10_p", ascending=True)

    plt.figure(figsize=(11, 7))
    plt.barh(plot_df["descriptor"], plot_df["neg_log10_p"])
    plt.xlabel("-log10(p_value)")
    plt.ylabel("descriptor")
    plt.title("Global comparison: top descriptors by Kruskal-Wallis significance")
    plt.tight_layout()

    out_path = out_dir / "global_top_descriptors.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[done] saved global plot to: {out_path}")


def make_pairwise_grouped_bar_signed(df: pd.DataFrame, out_dir: Path):
    """
    pairwise grouped bar:
    x축 = 사용자가 지정한 descriptor 순서
    막대 = pairwise comparison
    값 = signed Cliff's delta
    """
    pair_df = df[df["test_type"] == "pairwise"].copy()

    pair_df["cliffs_delta"] = pd.to_numeric(pair_df["cliffs_delta"], errors="coerce")
    pair_df["p_value"] = pd.to_numeric(pair_df["p_value"], errors="coerce")

    pair_df = pair_df[np.isfinite(pair_df["cliffs_delta"])].copy()

    if len(pair_df) == 0:
        print("[warn] No pairwise rows found. Skip pairwise grouped bar plot.")
        return

    pair_df["pair_label"] = pair_df["group_a"].astype(str) + " vs " + pair_df["group_b"].astype(str)

    # 지정 descriptor만 사용
    plot_df = pair_df[pair_df["descriptor"].isin(PAIRWISE_DESCRIPTOR_ORDER)].copy()

    if len(plot_df) == 0:
        print("[warn] None of the requested descriptors were found in pairwise rows.")
        return

    pivot = plot_df.pivot_table(
        index="descriptor",
        columns="pair_label",
        values="cliffs_delta",
        aggfunc="first"
    )

    # 지정 순서 유지
    descriptor_order = [d for d in PAIRWISE_DESCRIPTOR_ORDER if d in pivot.index]
    pivot = pivot.reindex(descriptor_order)

    pair_labels = list(pivot.columns)
    x = np.arange(len(pivot.index))

    n_pairs = len(pair_labels)
    width = 0.8 / max(n_pairs, 1)

    plt.figure(figsize=(15, 8))

    for i, pair_label in enumerate(pair_labels):
        xpos = x + i * width - (n_pairs - 1) * width / 2
        plt.bar(xpos, pivot[pair_label].values, width=width, label=pair_label)

    # y=0 기준선 추가: 양수/음수 해석 편하게
    plt.axhline(0.0, linewidth=1.0)

    plt.xticks(x, pivot.index, rotation=30, ha="right")
    plt.ylabel("Cliff's delta")
    plt.xlabel("descriptor")
    plt.title("Pairwise comparison: signed Cliff's delta for selected descriptors")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "pairwise_grouped_bar_signed_delta.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[done] saved pairwise grouped bar plot to: {out_path}")


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)

    required_cols = ["test_type", "descriptor", "p_value"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    make_global_plot(df, out_dir, top_n=args.top_n_global)
    make_pairwise_grouped_bar_signed(df, out_dir)


if __name__ == "__main__":
    main()
    
'''
python ./dataset/DUD-E/scripts_2/aggregate/make_descriptor_graphs.py \
  --csv ./dataset/DUD-E/aggregate_exports/compare_failure_cases/descriptor_tests.csv \
  --out_dir ./dataset/DUD-E/aggregate_exports/compare_failure_cases/graphs
'''
