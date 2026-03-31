#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoDock(Song et al.) vs DiffDock EF comparison plotter

입력:
    - metric_for_compare_AutoDock.xlsx

필수 column:
    - target
    - EF_1pct
    - EF_5pct
    - EF_10pct
    - song_crystal_ef1
    - song_crystal_ef5
    - song_crystal_ef10

출력:
    - ef1_scatter.png
    - ef5_scatter.png
    - ef10_scatter.png
    - correlation_summary.csv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def rankdata(values):
    """
    scipy 없이 Spearman 계산하기 위한 average rank 구현
    동점은 average rank를 부여한다.
    """
    s = pd.Series(values)
    ranks = s.rank(method="average")
    return ranks.to_numpy()


def pearson_corr(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)

    if len(x) < 2:
        return float("nan")

    x_mean = x.mean()
    y_mean = y.mean()

    num = ((x - x_mean) * (y - y_mean)).sum()
    den = math.sqrt(((x - x_mean) ** 2).sum()) * math.sqrt(((y - y_mean) ** 2).sum())

    if den == 0:
        return float("nan")
    return num / den


def spearman_corr(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def mae(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    return (x - y).abs().mean()


def ensure_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 column이 없습니다: {missing}")


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
):
    plot_df = df[["target", x_col, y_col]].copy()
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        raise ValueError(f"유효한 데이터가 없습니다: {x_col}, {y_col}")

    x = plot_df[x_col].astype(float)
    y = plot_df[y_col].astype(float)

    r = pearson_corr(x, y)
    rho = spearman_corr(x, y)
    plot_mae = mae(x, y)
    n = len(plot_df)

    max_val = max(x.max(), y.max())
    axis_max = max_val * 1.08 if max_val > 0 else 1.0

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=55, alpha=0.9, edgecolors="black", linewidths=0.6)

    # target label
    for _, row in plot_df.iterrows():
        plt.text(
            row[x_col] + axis_max * 0.007,
            row[y_col] + axis_max * 0.007,
            str(row["target"]),
            fontsize=9,
        )

    # y = x
    plt.plot(
        [0, axis_max],
        [0, axis_max],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

    stats_text = (
        f"n = {n}\n"
        f"Pearson r = {r:.3f}\n"
        f"Spearman ρ = {rho:.3f}\n"
        f"MAE = {plot_mae:.3f}"
    )

    plt.text(
        0.06 * axis_max,
        0.82 * axis_max,
        stats_text,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    plt.xlim(0, axis_max)
    plt.ylim(0, axis_max)

    plt.xlabel(x_label, fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=15, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "metric_x": x_col,
        "metric_y": y_col,
        "n": n,
        "pearson_r": r,
        "spearman_rho": rho,
        "mae": plot_mae,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="metric_for_compare_AutoDock.xlsx 경로",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="출력 디렉토리",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        "target",
        "EF_1pct",
        "EF_5pct",
        "EF_10pct",
        "song_crystal_ef1",
        "song_crystal_ef5",
        "song_crystal_ef10",
    ]
    ensure_required_columns(df, required_cols)

    # target 문자열 정리
    df["target"] = df["target"].astype(str).str.strip()

    summaries = []

    summaries.append(
        plot_scatter(
            df=df,
            x_col="EF_1pct",
            y_col="song_crystal_ef1",
            title="EF1%: DiffDock vs Song et al. (Crystal Structure)",
            x_label="EF1% (This Study; DiffDock)",
            y_label="EF1% (Song et al.; Crystal Structure)",
            out_path=out_dir / "ef1_scatter.png",
        )
    )

    summaries.append(
        plot_scatter(
            df=df,
            x_col="EF_5pct",
            y_col="song_crystal_ef5",
            title="EF5%: DiffDock vs Song et al. (Crystal Structure)",
            x_label="EF5% (This Study; DiffDock)",
            y_label="EF5% (Song et al.; Crystal Structure)",
            out_path=out_dir / "ef5_scatter.png",
        )
    )

    summaries.append(
        plot_scatter(
            df=df,
            x_col="EF_10pct",
            y_col="song_crystal_ef10",
            title="EF10%: DiffDock vs Song et al. (Crystal Structure)",
            x_label="EF10% (This Study; DiffDock)",
            y_label="EF10% (Song et al.; Crystal Structure)",
            out_path=out_dir / "ef10_scatter.png",
        )
    )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "correlation_summary.csv", index=False)

    print(f"[완료] 결과 저장 디렉토리: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
    

'''
python ./dataset/DUD-E/scripts_2/aggregate/compare_autodock_diffdock.py \
  --input ./dataset/DUD-E/aggregate_exports/compare_autodock/metric_for_compare_AutoDock.csv \
  --out_dir ./dataset/DUD-E/aggregate_exports/compare_autodock
'''
