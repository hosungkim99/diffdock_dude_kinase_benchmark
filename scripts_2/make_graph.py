#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_graph.py

- metrics_summary_all_diffdock_2.csv에서 metric column을 골라 그래프 생성
- x/y 범위는 min/max를 소수 둘째 자리에서 내림/올림(floor/ceil) 적용
- scatter, line, hist, box, violin, bar, corr plot 지원
- target -> kinase family 매핑을 내부 config로 정의
- family 스타일(color, marker, alpha, edgecolor, linewidth, size)을 plot에 반영
- scatter에서는 target label을 그래프 x축 중앙 기준으로 좌/우 자동 배치하고 선으로 연결
- 필요 시 x축에 자동 broken axis 적용

사용 예:
  1) scatter
    python make_graph.py --x nEF_1pct --y ROC_AUC --plot scatter

  2) scatter + auto broken axis
    python make_graph.py --x mean_COMdist --y nEF_1pct --plot scatter --auto_broken_axis

  3) line
    python make_graph.py --x nEF_1pct --y ROC_AUC --plot line

  4) hist
    python make_graph.py --x retry_rate --plot hist

  5) box
    python make_graph.py --y nEF_1pct --plot box

  6) corr
    python make_graph.py --plot corr --corr-metrics nEF_1pct nEF_5pct nEF_10pct ROC_AUC BEDROC

출력:
- --out_dir 아래에 figure 저장
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CSV = "/mnt/data/metrics_summary_all_diffdock_2.csv"


# ================================
# Kinase family settings
# ================================
# ADDED
FAMILY_TO_TARGETS = {
    "TK_receptor": ["egfr", "fgfr1", "igf1r", "csf1r", "kit", "met", "vgfr2", "tgfr1"],
    "TK_cytoplasmic": ["abl1", "src", "fak1", "jak2"],
    "MAPK": ["mk01", "mk10", "mk14", "mapk2"],
    "AGC": ["akt1", "akt2", "kpcb", "rock1"],
    "CDK": ["cdk2"],
    "other": ["braf", "plk1", "wee1"],
}

# ADDED
TARGET_TO_FAMILY = {
    t: f for f, ts in FAMILY_TO_TARGETS.items() for t in ts
}

# ADDED
FAMILY_ORDER = [
    "TK_receptor",
    "TK_cytoplasmic",
    "MAPK",
    "AGC",
    "CDK",
    "other",
    "unknown",
]

# ================================
# Visualization settings
# ================================
# ADDED
FAMILY_TO_COLOR = {
    "TK_receptor": "#1f77b4",
    "TK_cytoplasmic": "#ff7f0e",
    "MAPK": "#2ca02c",
    "AGC": "#d62728",
    "CDK": "#9467bd",
    "other": "#8c564b",
    "unknown": "#7f7f7f",
}

# ADDED
FAMILY_TO_MARKER = {
    "TK_receptor": "o",
    "TK_cytoplasmic": "s",
    "MAPK": "^",
    "AGC": "D",
    "CDK": "P",
    "other": "X",
    "unknown": "o",
}

# ADDED
FAMILY_TO_ALPHA = {
    "TK_receptor": 0.85,
    "TK_cytoplasmic": 0.85,
    "MAPK": 0.85,
    "AGC": 0.85,
    "CDK": 0.85,
    "other": 0.85,
    "unknown": 0.85,
}

# ADDED
FAMILY_TO_EDGECOLOR = {
    "TK_receptor": "black",
    "TK_cytoplasmic": "black",
    "MAPK": "black",
    "AGC": "black",
    "CDK": "black",
    "other": "black",
    "unknown": "black",
}

# ADDED
POINT_SIZE = 90

# ADDED
EDGE_WIDTH = 0.6

# ADDED
LABEL_FONT_SIZE = 8

# ADDED
LABEL_LINE_WIDTH = 0.7

# ADDED
LABEL_LINE_ALPHA = 0.8


# -------------------------
# Utils
# -------------------------

def floor_2(x: float) -> float:
    return math.floor(x * 100.0) / 100.0


def ceil_2(x: float) -> float:
    return math.ceil(x * 100.0) / 100.0


def axis_limits(values: Sequence[float]) -> Tuple[float, float]:
    """min/max에 대해 소수 둘째 자리 내림/올림 범위 산출."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    lo = floor_2(float(arr.min()))
    hi = ceil_2(float(arr.max()))
    if lo == hi:
        pad = 0.01 if lo == 0 else abs(lo) * 0.01
        lo -= pad
        hi += pad
        lo = floor_2(lo)
        hi = ceil_2(hi)
    return lo, hi


def ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 없는 column이 있음: {missing}\n사용 가능 column: {list(df.columns)}")


def sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)


def load_group_map(path: str) -> pd.DataFrame:
    """
    group-map CSV 포맷:
      target,group
    또는
      target,family
    처럼 target과 group 컬럼(두 번째 컬럼)이 있으면 OK.
    """
    m = pd.read_csv(path)
    if "target" not in m.columns:
        raise ValueError("--group-map 파일에 'target' column이 필요함.")
    group_col = None
    for cand in ["group", "family", "class", "kinase_family"]:
        if cand in m.columns:
            group_col = cand
            break
    if group_col is None:
        other_cols = [c for c in m.columns if c != "target"]
        if len(other_cols) == 0:
            raise ValueError("--group-map 파일에 target 외에 그룹 컬럼이 필요함.")
        group_col = other_cols[0]
    return m[["target", group_col]].rename(columns={group_col: "group"})


# ADDED
def add_family_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    target -> family 매핑을 사용해 df['family'] 생성.
    정의되지 않은 target은 unknown 처리.
    """
    df = df.copy()
    if "target" not in df.columns:
        raise ValueError("family 생성에는 'target' column이 필요함.")
    df["family"] = df["target"].map(TARGET_TO_FAMILY).fillna("unknown")
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    return df


# ADDED
def get_family_style(family_name: str) -> dict:
    return {
        "color": FAMILY_TO_COLOR.get(family_name, FAMILY_TO_COLOR["unknown"]),
        "marker": FAMILY_TO_MARKER.get(family_name, FAMILY_TO_MARKER["unknown"]),
        "alpha": FAMILY_TO_ALPHA.get(family_name, FAMILY_TO_ALPHA["unknown"]),
        "edgecolors": FAMILY_TO_EDGECOLOR.get(family_name, FAMILY_TO_EDGECOLOR["unknown"]),
        "linewidths": EDGE_WIDTH,
        "s": POINT_SIZE,
    }


# ADDED
def iter_family_order_present(df: pd.DataFrame) -> List[str]:
    present = set(df["family"].astype(str).tolist())
    return [f for f in FAMILY_ORDER if f in present]


# ================================
# Broken axis detection
# ================================
# ADDED
def detect_broken_axis(values: Sequence[float], threshold: float = 2.5) -> Optional[str]:
    """
    extreme outlier가 있으면 x축 broken axis 방향을 반환한다.
    반환값:
      - "upper": 최댓값 쪽 extreme outlier
      - "lower": 최솟값 쪽 extreme outlier
      - None   : broken axis 불필요
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    vals = np.sort(vals)

    if vals.size < 4:
        return None

    eps = 1e-9
    x1 = vals[0]
    x2 = vals[1]
    xn1 = vals[-2]
    xn = vals[-1]

    upper_ratio = (xn - xn1) / (xn1 - x1 + eps)
    lower_ratio = (x2 - x1) / (xn - x2 + eps)

    if upper_ratio >= threshold:
        return "upper"
    if lower_ratio >= threshold:
        return "lower"
    return None


# ADDED
def get_broken_axis_bounds(values: Sequence[float], direction: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    broken axis용 두 구간을 반환한다.
    direction == "upper":
      main  = [min, 2nd max]
      out   = [max, max]
    direction == "lower":
      out   = [min, min]
      main  = [2nd min, max]
    단, 실제 표시를 위해 약간의 패딩을 준다.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    vals = np.sort(vals)

    if vals.size < 4:
        lo, hi = axis_limits(vals)
        return (lo, hi), (lo, hi)

    if direction == "upper":
        main_vals = vals[:-1]
        out_vals = vals[-1:]
    elif direction == "lower":
        out_vals = vals[:1]
        main_vals = vals[1:]
    else:
        lo, hi = axis_limits(vals)
        return (lo, hi), (lo, hi)

    main_lo, main_hi = axis_limits(main_vals)
    out_lo, out_hi = axis_limits(out_vals)

    # 단일값 구간이면 시각화용 padding 강화
    if len(out_vals) == 1:
        v = float(out_vals[0])
        pad = max(abs(v) * 0.03, 0.5)
        out_lo = floor_2(v - pad)
        out_hi = ceil_2(v + pad)

    return (main_lo, main_hi), (out_lo, out_hi)


# ADDED
def create_broken_x_figure(direction: str):
    """
    x축 broken axis용 figure/axes 생성.
    direction == "upper":
      왼쪽: main zone, 오른쪽: upper outlier zone
    direction == "lower":
      왼쪽: lower outlier zone, 오른쪽: main zone
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(10.5, 6.2),
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05},
    )

    if direction == "upper":
        return fig, ax_left, ax_right
    elif direction == "lower":
        # lower outlier는 왼쪽 작은 축, main은 오른쪽 큰 축이 더 자연스럽다.
        plt.close(fig)
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(10.5, 6.2),
            gridspec_kw={"width_ratios": [1, 4], "wspace": 0.05},
        )
        return fig, ax_left, ax_right
    else:
        raise ValueError(f"unknown broken direction: {direction}")


# ADDED
def draw_broken_x_marks(ax_left, ax_right):
    """
    x축 broken axis 경계의 지그재그 표시.
    """
    d = 0.015
    kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs = dict(transform=ax_right.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)


# ADDED
def split_dataframe_for_broken_x(df: pd.DataFrame, x_col: str, direction: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    broken x-axis 방향에 따라 df를 main/outlier subset으로 나눈다.
    direction == "upper":
      main    = x <= 2nd largest
      outlier = x == largest 근처
    direction == "lower":
      outlier = x == smallest 근처
      main    = x >= 2nd smallest
    """
    vals = np.asarray(df[x_col].astype(float), dtype=float)
    uniq_sorted = np.sort(np.unique(vals))

    if uniq_sorted.size < 2:
        return df.copy(), df.iloc[0:0].copy()

    if direction == "upper":
        split_value = uniq_sorted[-2]
        df_main = df[df[x_col].astype(float) <= split_value].copy()
        df_out = df[df[x_col].astype(float) > split_value].copy()
        return df_main, df_out
    elif direction == "lower":
        split_value = uniq_sorted[1]
        df_out = df[df[x_col].astype(float) < split_value].copy()
        df_main = df[df[x_col].astype(float) >= split_value].copy()
        return df_main, df_out
    else:
        return df.copy(), df.iloc[0:0].copy()


# MODIFIED
def add_scatter_labels_with_connectors(
    ax,
    df_sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_global_lo: float,
    x_global_hi: float,
    expand_xlim: bool = True,   # ADDED
):
    """
    scatter point 옆에 target name을 자동 배치한다.
    그래프 전체 x축 중앙을 기준으로:
      - 포인트가 왼쪽에 있으면 label도 왼쪽
      - 포인트가 오른쪽에 있으면 label도 오른쪽
    그리고 point와 text를 선으로 연결한다.
    """
    if df_sub.empty:
        return

    ensure_columns(df_sub, [x_col, y_col, "target", "family"])

    y_vals = df_sub[y_col].astype(float).to_numpy()
    ylo_sub, yhi_sub = axis_limits(y_vals)
    y_span_sub = max(yhi_sub - ylo_sub, 1e-6)

    x_span_global = max(x_global_hi - x_global_lo, 1e-6)
    x_mid = 0.5 * (x_global_lo + x_global_hi)

    left_df = df_sub[df_sub[x_col].astype(float) < x_mid].copy()
    right_df = df_sub[df_sub[x_col].astype(float) >= x_mid].copy()

    def _draw_side(sub_df: pd.DataFrame, side: str):
        if sub_df.empty:
            return

        sub_df = sub_df.sort_values(by=y_col).reset_index(drop=True)
        n = len(sub_df)

        if side == "left":
            x_text = x_global_lo - 0.08 * x_span_global
            ha = "right"
        else:
            x_text = x_global_hi + 0.08 * x_span_global
            ha = "left"

        if n == 1:
            y_text_positions = [float(sub_df.loc[0, y_col])]
        else:
            y_min = float(sub_df[y_col].min()) - 0.02 * y_span_sub
            y_max = float(sub_df[y_col].max()) + 0.02 * y_span_sub
            y_text_positions = np.linspace(y_min, y_max, n)

        for i, (_, row) in enumerate(sub_df.iterrows()):
            x0 = float(row[x_col])
            y0 = float(row[y_col])
            xt = x_text
            yt = float(y_text_positions[i])

            family_name = str(row["family"])
            target_name = str(row["target"])
            line_color = FAMILY_TO_COLOR.get(family_name, FAMILY_TO_COLOR["unknown"])

            ax.plot(
                [x0, xt],
                [y0, yt],
                color=line_color,
                linewidth=LABEL_LINE_WIDTH,
                alpha=LABEL_LINE_ALPHA,
            )

            ax.text(
                xt,
                yt,
                target_name,
                fontsize=LABEL_FONT_SIZE,
                ha=ha,
                va="center",
                color=line_color,
            )

    _draw_side(left_df, "left")
    _draw_side(right_df, "right")

    if expand_xlim:
        ax.set_xlim(
            x_global_lo - 0.22 * x_span_global,
            x_global_hi + 0.22 * x_span_global,
        )


# ADDED
def plot_scatter_single_axis(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    with_labels: bool = True,
):
    """
    단일 axis 위에 scatter를 그린다.
    """
    ensure_columns(df, [x, y, "family", "target"])
    xlo, xhi = axis_limits(df[x].astype(float))

    for family_name in iter_family_order_present(df):
        sub = df[df["family"].astype(str) == family_name]
        style = get_family_style(family_name)

        ax.scatter(
            sub[x],
            sub[y],
            label=family_name,
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            s=style["s"],
        )

        if with_labels:
            add_scatter_labels_with_connectors(
                ax=ax,
                df_sub=sub,
                x_col=x,
                y_col=y,
                x_global_lo=xlo,
                x_global_hi=xhi,
                expand_xlim=True,
            )


# MODIFIED
def plot_scatter(
    df: pd.DataFrame,
    x: str,
    ys: List[str],
    group: Optional[str],
    show_group: bool,
    title: str,
    auto_broken_axis: bool = False,            # ADDED
    broken_axis_threshold: float = 2.5,        # ADDED
):
    ensure_columns(df, [x] + ys + ["family", "target"])

    y0 = ys[0]

    # ADDED
    broken_direction = None
    if auto_broken_axis:
        broken_direction = detect_broken_axis(df[x].astype(float).to_numpy(), threshold=broken_axis_threshold)

    if broken_direction is None:
        fig, ax = plt.subplots(figsize=(8.4, 6.2))

        plot_scatter_single_axis(ax=ax, df=df, x=x, y=y0, with_labels=True)

        if len(ys) > 1:
            for y in ys[1:]:
                ax.scatter(
                    df[x],
                    df[y],
                    alpha=0.30,
                    marker="x",
                    s=POINT_SIZE * 0.7,
                    linewidths=0.8,
                    label=f"{y} (all)",
                )

        ax.legend(title="family", fontsize=9)
        ax.set_xlabel(x)
        ax.set_ylabel(", ".join(ys))
        ax.set_title(title)

        yvals = np.concatenate([df[y].astype(float).to_numpy() for y in ys])
        ylo, yhi = axis_limits(yvals)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    # ADDED
    # broken axis 사용
    fig, ax_left, ax_right = create_broken_x_figure(broken_direction)
    df_main, df_out = split_dataframe_for_broken_x(df, x_col=x, direction=broken_direction)
    (main_lo, main_hi), (out_lo, out_hi) = get_broken_axis_bounds(df[x].astype(float).to_numpy(), broken_direction)

    if broken_direction == "upper":
        ax_main = ax_left
        ax_out = ax_right
    else:
        ax_out = ax_left
        ax_main = ax_right

    # main zone
    for family_name in iter_family_order_present(df_main):
        sub = df_main[df_main["family"].astype(str) == family_name]
        style = get_family_style(family_name)
        ax_main.scatter(
            sub[x],
            sub[y0],
            label=family_name,
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            s=style["s"],
        )

    # outlier zone
    for family_name in iter_family_order_present(df_out):
        sub = df_out[df_out["family"].astype(str) == family_name]
        style = get_family_style(family_name)
        ax_out.scatter(
            sub[x],
            sub[y0],
            label=family_name,
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            s=style["s"],
        )

    # label은 복잡도 때문에 main zone에만 우선 적용
    if not df_main.empty:
        add_scatter_labels_with_connectors(
            ax=ax_main,
            df_sub=df_main,
            x_col=x,
            y_col=y0,
            x_global_lo=main_lo,
            x_global_hi=main_hi,
            expand_xlim=False,
        )

    if not df_out.empty:
        add_scatter_labels_with_connectors(
            ax=ax_out,
            df_sub=df_out,
            x_col=x,
            y_col=y0,
            x_global_lo=out_lo,
            x_global_hi=out_hi,
            expand_xlim=False,
        )

    # 추가 y는 label 없이 보조 표시
    if len(ys) > 1:
        for y in ys[1:]:
            if not df_main.empty:
                ax_main.scatter(
                    df_main[x],
                    df_main[y],
                    alpha=0.30,
                    marker="x",
                    s=POINT_SIZE * 0.7,
                    linewidths=0.8,
                    label=f"{y} (all)",
                )
            if not df_out.empty:
                ax_out.scatter(
                    df_out[x],
                    df_out[y],
                    alpha=0.30,
                    marker="x",
                    s=POINT_SIZE * 0.7,
                    linewidths=0.8,
                    label=f"{y} (all)",
                )

    # 축 범위
    ax_main.set_xlim(main_lo, main_hi)
    ax_out.set_xlim(out_lo, out_hi)

    yvals = np.concatenate([df[y].astype(float).to_numpy() for y in ys])
    ylo, yhi = axis_limits(yvals)
    ax_main.set_ylim(ylo, yhi)
    ax_out.set_ylim(ylo, yhi)

    # 외형 정리
    ax_main.grid(True, alpha=0.25)
    ax_out.grid(True, alpha=0.25)
    ax_main.set_ylabel(", ".join(ys))
    ax_main.set_xlabel(x)
    ax_out.set_xlabel(x)
    fig.suptitle(title)

    # broken mark
    draw_broken_x_marks(ax_left, ax_right)

    # 내부 spine 정리
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(labelleft=False, left=False)

    handles, labels = ax_main.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax_right.legend(list(uniq.values()), list(uniq.keys()), title="family", fontsize=9, loc="best")

    fig.tight_layout()
    return fig


# ADDED
def plot_line_single_axis(ax, df: pd.DataFrame, x: str, ys: List[str]):
    ensure_columns(df, [x] + ys + ["family"])
    df2 = df.sort_values(by=x)

    if len(ys) == 1:
        y0 = ys[0]
        for family_name in iter_family_order_present(df2):
            sub = df2[df2["family"].astype(str) == family_name].sort_values(by=x)
            style = get_family_style(family_name)
            ax.plot(
                sub[x],
                sub[y0],
                marker=style["marker"],
                linewidth=1.2,
                alpha=style["alpha"],
                color=style["color"],
                label=family_name,
            )
        ax.legend(title="family", fontsize=9)
    else:
        for y in ys:
            ax.plot(df2[x], df2[y], marker="o", linewidth=1.2, label=y, alpha=0.9)
        ax.legend(fontsize=9)


# MODIFIED
def plot_line(
    df: pd.DataFrame,
    x: str,
    ys: List[str],
    group: Optional[str],
    show_group: bool,
    title: str,
    auto_broken_axis: bool = False,            # ADDED
    broken_axis_threshold: float = 2.5,        # ADDED
):
    ensure_columns(df, [x] + ys + ["family"])

    broken_direction = None
    if auto_broken_axis:
        broken_direction = detect_broken_axis(df[x].astype(float).to_numpy(), threshold=broken_axis_threshold)

    if broken_direction is None:
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        plot_line_single_axis(ax=ax, df=df, x=x, ys=ys)

        ax.set_xlabel(x)
        ax.set_ylabel(", ".join(ys))
        ax.set_title(title)

        xlo, xhi = axis_limits(df[x].astype(float))
        ax.set_xlim(xlo, xhi)

        yvals = np.concatenate([df[y].astype(float).to_numpy() for y in ys])
        ylo, yhi = axis_limits(yvals)
        ax.set_ylim(ylo, yhi)

        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    # broken axis
    fig, ax_left, ax_right = create_broken_x_figure(broken_direction)
    df_main, df_out = split_dataframe_for_broken_x(df, x_col=x, direction=broken_direction)
    (main_lo, main_hi), (out_lo, out_hi) = get_broken_axis_bounds(df[x].astype(float).to_numpy(), broken_direction)

    if broken_direction == "upper":
        ax_main = ax_left
        ax_out = ax_right
    else:
        ax_out = ax_left
        ax_main = ax_right

    if not df_main.empty:
        plot_line_single_axis(ax=ax_main, df=df_main, x=x, ys=ys)
    if not df_out.empty:
        plot_line_single_axis(ax=ax_out, df=df_out, x=x, ys=ys)

    ax_main.set_xlim(main_lo, main_hi)
    ax_out.set_xlim(out_lo, out_hi)

    yvals = np.concatenate([df[y].astype(float).to_numpy() for y in ys])
    ylo, yhi = axis_limits(yvals)
    ax_main.set_ylim(ylo, yhi)
    ax_out.set_ylim(ylo, yhi)

    ax_main.grid(True, alpha=0.25)
    ax_out.grid(True, alpha=0.25)
    ax_main.set_ylabel(", ".join(ys))
    ax_main.set_xlabel(x)
    ax_out.set_xlabel(x)
    fig.suptitle(title)

    draw_broken_x_marks(ax_left, ax_right)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(labelleft=False, left=False)

    handles, labels = ax_main.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        ax_right.legend(list(uniq.values()), list(uniq.keys()), title="family", fontsize=9, loc="best")

    fig.tight_layout()
    return fig


def plot_hist(df: pd.DataFrame, x: str, bins: int, title: str):
    ensure_columns(df, [x, "family"])

    fig, ax = plt.subplots(figsize=(7.8, 5.6))

    for family_name in iter_family_order_present(df):
        sub = df[df["family"].astype(str) == family_name]
        style = get_family_style(family_name)
        data = sub[x].astype(float).to_numpy()
        data = data[~np.isnan(data)]
        if data.size == 0:
            continue
        ax.hist(
            data,
            bins=bins,
            alpha=0.35,
            label=family_name,
            color=style["color"],
            edgecolor=style["edgecolors"],
            linewidth=0.5,
        )

    ax.set_xlabel(x)
    ax.set_ylabel("count")
    ax.set_title(title)

    xlo, xhi = axis_limits(df[x].astype(float))
    ax.set_xlim(xlo, xhi)

    ax.legend(title="family", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_box_or_violin(df: pd.DataFrame, y: str, group: Optional[str], kind: str, title: str):
    ensure_columns(df, [y, "family"])

    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    labels = []
    groups = []
    for family_name in iter_family_order_present(df):
        sub = df[df["family"].astype(str) == family_name]
        vals = sub[y].astype(float).to_numpy()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        labels.append(family_name)
        groups.append(vals)

    if len(groups) == 0:
        raise ValueError(f"{y}에 대해 plotting 가능한 데이터가 없음.")

    if kind == "box":
        box = ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
        for patch, family_name in zip(box["boxes"], labels):
            patch.set_facecolor(FAMILY_TO_COLOR.get(family_name, FAMILY_TO_COLOR["unknown"]))
            patch.set_alpha(0.45)
            patch.set_edgecolor(FAMILY_TO_EDGECOLOR.get(family_name, FAMILY_TO_EDGECOLOR["unknown"]))
    else:
        vp = ax.violinplot(groups, showmeans=False, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        for body, family_name in zip(vp["bodies"], labels):
            body.set_facecolor(FAMILY_TO_COLOR.get(family_name, FAMILY_TO_COLOR["unknown"]))
            body.set_edgecolor(FAMILY_TO_EDGECOLOR.get(family_name, FAMILY_TO_EDGECOLOR["unknown"]))
            body.set_alpha(0.45)

    ax.set_xlabel("family")
    ax.set_ylabel(y)
    ax.set_title(title)

    ylo, yhi = axis_limits(df[y].astype(float))
    ax.set_ylim(ylo, yhi)

    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_bar(df: pd.DataFrame, x: str, y: str, group: Optional[str], show_group: bool, title: str):
    ensure_columns(df, [x, y, "family"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    df2 = df.sort_values(by=["family", y], ascending=[True, False]).copy()

    colors = [
        FAMILY_TO_COLOR.get(str(fam), FAMILY_TO_COLOR["unknown"])
        for fam in df2["family"].astype(str).tolist()
    ]

    edgecolors = [
        FAMILY_TO_EDGECOLOR.get(str(fam), FAMILY_TO_EDGECOLOR["unknown"])
        for fam in df2["family"].astype(str).tolist()
    ]

    ax.bar(
        df2[x].astype(str),
        df2[y].astype(float),
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.5,
        alpha=0.85,
    )

    fam_list = df2["family"].astype(str).tolist()
    last = None
    for i, fam in enumerate(fam_list):
        if last is None:
            last = fam
        elif fam != last:
            ax.axvline(i - 0.5, linestyle="--", linewidth=0.8, alpha=0.5)
            last = fam

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)

    ylo, yhi = axis_limits(df2[y].astype(float))
    ax.set_ylim(ylo, yhi)

    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.25)

    handles = []
    for family_name in iter_family_order_present(df2):
        handle = plt.Line2D(
            [0], [0],
            marker="s",
            color="none",
            markerfacecolor=FAMILY_TO_COLOR.get(family_name, FAMILY_TO_COLOR["unknown"]),
            markeredgecolor=FAMILY_TO_EDGECOLOR.get(family_name, FAMILY_TO_EDGECOLOR["unknown"]),
            markersize=8,
            label=family_name,
        )
        handles.append(handle)
    ax.legend(handles=handles, title="family", fontsize=9)

    fig.tight_layout()
    return fig


def plot_corr(df: pd.DataFrame, metrics: List[str], title: str):
    ensure_columns(df, metrics)

    sub = df[metrics].astype(float)
    corr = sub.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    im = ax.imshow(corr.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(metrics)
    ax.set_title(title)

    for i in range(len(metrics)):
        for j in range(len(metrics)):
            v = corr.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Generate plots from metrics_summary_all_diffdock_2.csv")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="path to metrics_summary_all_diffdock_2.csv")
    p.add_argument("--x", type=str, default=None, help="x-axis metric (required for most plots)")
    p.add_argument("--y", type=str, action="append", default=[], help="y-axis metric (repeatable). ex: --y ROC_AUC --y BEDROC")
    p.add_argument("--plot", type=str, required=True,
                   choices=["scatter", "line", "hist", "box", "violin", "bar", "corr"],
                   help="plot type")
    p.add_argument("--group", type=str, default=None, help="grouping column name (legacy option; family styling is applied automatically)")
    p.add_argument("--group-map", type=str, default=None, help="CSV with target->group mapping (columns: target, group/family/...)")
    p.add_argument("--show-group", action="store_true", help="legacy option; family styling is now applied automatically")
    p.add_argument("--bins", type=int, default=20, help="bins for hist")
    p.add_argument("--corr-metrics", nargs="*", default=None, help="metrics list for corr plot")
    p.add_argument("--outdir", type=str, default=None, help="legacy output directory option")
    p.add_argument("--out_dir", type=str, default="plots", help="output directory")
    p.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "svg"], help="output format")
    p.add_argument("--dpi", type=int, default=200, help="dpi for raster output (png)")
    p.add_argument("--title", type=str, default=None, help="custom title")

    # ADDED
    p.add_argument(
        "--auto_broken_axis",
        action="store_true",
        help="Automatically apply broken x-axis when extreme outlier is detected on x metric",
    )
    # ADDED
    p.add_argument(
        "--broken_axis_threshold",
        type=float,
        default=2.5,
        help="Threshold for broken-axis outlier detection",
    )

    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = add_family_column(df)

    if args.group_map is not None:
        gm = load_group_map(args.group_map)
        df = df.merge(gm, on="target", how="left")
        if args.group is None:
            args.group = "group"

    final_out_dir = args.out_dir if args.out_dir is not None else (args.outdir if args.outdir is not None else "plots")
    outdir = Path(final_out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    title = args.title
    if title is None:
        title = f"{args.plot}: x={args.x}, y={','.join(args.y) if args.y else 'NA'}"

    fig = None

    if args.plot in ["scatter", "line", "bar"]:
        if args.x is None:
            raise ValueError(f"--plot {args.plot}에는 --x가 필요함.")
        if args.plot != "bar" and len(args.y) == 0:
            raise ValueError(f"--plot {args.plot}에는 최소 1개의 --y가 필요함.")
        if args.plot == "scatter":
            fig = plot_scatter(
                df=df,
                x=args.x,
                ys=args.y,
                group=args.group,
                show_group=args.show_group,
                title=title,
                auto_broken_axis=args.auto_broken_axis,                 # ADDED
                broken_axis_threshold=args.broken_axis_threshold,       # ADDED
            )
        elif args.plot == "line":
            fig = plot_line(
                df=df,
                x=args.x,
                ys=args.y,
                group=args.group,
                show_group=args.show_group,
                title=title,
                auto_broken_axis=args.auto_broken_axis,                 # ADDED
                broken_axis_threshold=args.broken_axis_threshold,       # ADDED
            )
        elif args.plot == "bar":
            if len(args.y) != 1:
                raise ValueError("--plot bar에는 --y를 정확히 1개만 지정해야 함.")
            fig = plot_bar(df, args.x, args.y[0], args.group, args.show_group, title)

    elif args.plot == "hist":
        if args.x is None:
            raise ValueError("--plot hist에는 --x가 필요함.")
        fig = plot_hist(df, args.x, args.bins, title)

    elif args.plot in ["box", "violin"]:
        if len(args.y) != 1:
            raise ValueError(f"--plot {args.plot}에는 --y를 정확히 1개만 지정해야 함.")
        kind = "box" if args.plot == "box" else "violin"
        fig = plot_box_or_violin(df, args.y[0], args.group, kind, title)

    elif args.plot == "corr":
        metrics = args.corr_metrics
        if metrics is None or len(metrics) == 0:
            default_metrics = [
                "coverage_rate", "fail_rate", "skip_rate", "retry_rate",
                "ROC_AUC", "LogAUC", "BEDROC",
                "EF_1pct", "EF_5pct", "EF_10pct",
                "nEF_1pct", "nEF_5pct", "nEF_10pct",
                "clash_rate_all", "clash_rate_top1pct",
                "pocket_in_rate_all", "pocket_in_rate_top1pct",
                "mean_COMdist", "active_mean_COMdist", "top1_mean_COMdist",
                "COMdist2rate"
            ]
            metrics = [m for m in default_metrics if m in df.columns]
        fig = plot_corr(df, metrics, title)

    else:
        raise ValueError(f"unknown plot type: {args.plot}")

    # MODIFIED
    if args.plot == "corr":
        if args.corr_metrics is not None and len(args.corr_metrics) > 0:
            metric_tag = "_".join(args.corr_metrics[:3])
            if len(args.corr_metrics) > 3:
                metric_tag += "_etc"
        else:
            metric_tag = "default_metrics"

        fname = sanitize_filename(f"corr__{metric_tag}.{args.fmt}")

    else:
        xs = f"x_{args.x}" if args.x else "x_NA"
        ys = f"y_{'-'.join(args.y)}" if args.y else "y_NA"
        fname = sanitize_filename(f"{args.plot}__{xs}__{ys}.{args.fmt}")

    outpath = outdir / fname

    if args.fmt == "png":
        fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    else:
        fig.savefig(outpath, bbox_inches="tight")

    print(f"[OK] saved: {outpath}")


if __name__ == "__main__":
    main()
    
'''실행 예제(자동 broken axis 켠 scatter/ threshold 직접 조정)
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/make_graph.py \
  --csv /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/metrics_summary_all_diffdock_2.csv \
  --x mean_COMdist \
  --y nEF_1pct \
  --plot scatter \
  --auto_broken_axis \
  --broken_axis_threshold 3.0 \
  --out_dir /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/graphs

python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/make_graph.py \
  --csv /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/metrics_summary_all_diffdock_2.csv \
  --x mean_COMdist \
  --y nEF_1pct \
  --plot line \
  --auto_broken_axis \
  --broken_axis_threshold 3.0 \
  --out_dir /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/graphs
'''