# src/aggregate/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationConfig:
    n_bins: int = 20
    binning: str = "uniform"  # "uniform" or "quantile"
    pose_cutoff_A: float = 2.0
    # scope 정책
    # - label calibration: success==1 전체 (actives+decoys)
    # - pose calibration: success==1 & label==1 (actives only)
    require_success: bool = True


def _to_float_array(x) -> np.ndarray:
    return np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)


def _to_int_array(x) -> np.ndarray:
    return np.asarray(pd.to_numeric(x, errors="coerce").fillna(0).astype(int), dtype=int)


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _bin_edges(conf: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    """
    returns edges of length n_bins+1, monotonically increasing, covering [min, max]
    """
    if n_bins <= 1:
        n_bins = 1

    c = conf[np.isfinite(conf)]
    if c.size == 0:
        # fallback edges
        return np.linspace(0.0, 1.0, n_bins + 1)

    if binning == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(c, qs)
        # ensure monotonic non-decreasing; handle duplicates by small jitter
        edges = np.asarray(edges, dtype=float)
        # If many ties, quantile edges can collapse; keep as-is but binning will skip empty bins naturally.
        edges[0] = np.min(c)
        edges[-1] = np.max(c)
        return edges

    # uniform
    lo = float(np.min(c))
    hi = float(np.max(c))
    if hi == lo:
        hi = lo + 1e-12
    return np.linspace(lo, hi, n_bins + 1)


def _assign_bins(conf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    bins: 0..n_bins-1
    rule: left-closed, right-open except last includes right edge
    """
    n_bins = len(edges) - 1
    # np.digitize: returns 1..n_bins when bins=edges[1:-1]
    # Use right=False: edges are left-closed.
    idx = np.digitize(conf, edges[1:-1], right=False)
    # idx already in [0, n_bins-1]
    idx = np.clip(idx, 0, n_bins - 1)
    return idx.astype(int)


def calibration_table_from_arrays(
    y_true: np.ndarray,
    conf: np.ndarray,
    *,
    n_bins: int,
    binning: str,
) -> pd.DataFrame:
    """
    Produce per-bin calibration table:
      bin_id, bin_low, bin_high, n, mean_conf, empirical_rate, abs_gap
    where:
      empirical_rate = mean(y_true) in bin
      abs_gap = |empirical_rate - mean_conf|
    """
    y = np.asarray(y_true, dtype=float)
    c = np.asarray(conf, dtype=float)

    m = _finite_mask(y, c)
    y = y[m]
    c = c[m]

    edges = _bin_edges(c, n_bins=n_bins, binning=binning)
    bin_id = _assign_bins(c, edges)

    rows = []
    for k in range(len(edges) - 1):
        mk = (bin_id == k)
        nk = int(mk.sum())
        if nk == 0:
            rows.append({
                "bin_id": k,
                "bin_low": float(edges[k]),
                "bin_high": float(edges[k + 1]),
                "n": 0,
                "mean_conf": float("nan"),
                "empirical_rate": float("nan"),
                "abs_gap": float("nan"),
            })
            continue

        mean_conf = float(np.mean(c[mk]))
        emp = float(np.mean(y[mk]))
        rows.append({
            "bin_id": k,
            "bin_low": float(edges[k]),
            "bin_high": float(edges[k + 1]),
            "n": nk,
            "mean_conf": mean_conf,
            "empirical_rate": emp,
            "abs_gap": float(abs(emp - mean_conf)),
        })

    return pd.DataFrame(rows)


def ece_from_table(calib_df: pd.DataFrame) -> float:
    """
    ECE = sum_k (n_k/N) * |acc_k - conf_k|
    """
    if calib_df is None or len(calib_df) == 0:
        return float("nan")
    n = pd.to_numeric(calib_df["n"], errors="coerce").fillna(0).astype(int)
    N = int(n.sum())
    if N <= 0:
        return float("nan")

    gap = pd.to_numeric(calib_df["abs_gap"], errors="coerce")
    # empty bins -> NaN gap; weight 0 anyway
    w = n / N
    return float(np.nansum(w.to_numpy(dtype=float) * gap.to_numpy(dtype=float)))


def mce_from_table(calib_df: pd.DataFrame) -> float:
    """
    MCE = max_k |acc_k - conf_k|
    """
    if calib_df is None or len(calib_df) == 0:
        return float("nan")
    gap = pd.to_numeric(calib_df["abs_gap"], errors="coerce")
    if gap.notna().sum() == 0:
        return float("nan")
    return float(np.nanmax(gap.to_numpy(dtype=float)))


def brier_score(y_true: np.ndarray, conf: np.ndarray) -> float:
    """
    Brier = mean( (conf - y)^2 )
    """
    y = np.asarray(y_true, dtype=float)
    c = np.asarray(conf, dtype=float)
    m = _finite_mask(y, c)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((c[m] - y[m]) ** 2))


def build_calibration_tables_from_master(
    master_df: pd.DataFrame,
    cfg: CalibrationConfig = CalibrationConfig(),
) -> Dict[str, object]:
    """
    Returns:
      {
        "label_table": DataFrame,
        "pose_table": DataFrame,
        "summary": {
           "n_total": int,
           "label": {"n": int, "ece":..., "mce":..., "brier":...},
           "pose":  {"n": int, "ece":..., "mce":..., "brier":...},
        }
      }

    Policies:
      - label calibration: success==1 rows (actives+decoys)
      - pose calibration:  success==1 & label==1 rows (actives only)
                         y_pose = 1[COMdist_A <= pose_cutoff_A]
    """
    df = master_df.copy()

    # 필수 컬럼 체크 (없으면 즉시 오류)
    required = ["label", "confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"master_table missing columns: {missing}")

    # success filter
    if cfg.require_success:
        if "success" not in df.columns:
            raise ValueError("master_table missing column: success (required when require_success=True)")
        df = df[pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int) == 1]

    n_total = int(len(df))

    # --- label calibration (actives+decoys) ---
    y_label = _to_int_array(df["label"])
    c_all = _to_float_array(df["confidence"])

    label_table = calibration_table_from_arrays(
        y_label, c_all, n_bins=cfg.n_bins, binning=cfg.binning
    )
    label_summary = {
        "n": int(_finite_mask(y_label.astype(float), c_all).sum()),
        "ece": ece_from_table(label_table),
        "mce": mce_from_table(label_table),
        "brier": brier_score(y_label, c_all),
    }

    # --- pose calibration (actives only) ---
    # 정의: actives에서 confidence가 "pose hit probability"로 얼마나 calibrate 되었는지
    if "COMdist_A" in df.columns:
        df_act = df[pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int) == 1].copy()
        com = _to_float_array(df_act["COMdist_A"])
        y_pose = (com <= float(cfg.pose_cutoff_A)).astype(int)
        c_pose = _to_float_array(df_act["confidence"])

        pose_table = calibration_table_from_arrays(
            y_pose, c_pose, n_bins=cfg.n_bins, binning=cfg.binning
        )
        pose_summary = {
            "n": int(_finite_mask(y_pose.astype(float), c_pose).sum()),
            "ece": ece_from_table(pose_table),
            "mce": mce_from_table(pose_table),
            "brier": brier_score(y_pose, c_pose),
        }
    else:
        pose_table = calibration_table_from_arrays(
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
            n_bins=cfg.n_bins,
            binning=cfg.binning,
        )
        pose_summary = {
            "n": 0,
            "ece": float("nan"),
            "mce": float("nan"),
            "brier": float("nan"),
        }

    return {
        "label_table": label_table,
        "pose_table": pose_table,
        "summary": {
            "n_total_success": n_total,
            "label": label_summary,
            "pose": pose_summary,
            "config": {
                "n_bins": cfg.n_bins,
                "binning": cfg.binning,
                "pose_cutoff_A": cfg.pose_cutoff_A,
                "require_success": cfg.require_success,
            },
        },
    }