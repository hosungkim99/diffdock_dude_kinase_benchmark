#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def as_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_targets_auto(dude_root: Path, eval_subdir: str) -> List[str]:
    # dude_root/<target>/eval/<eval_subdir>/master_table.csv 존재 여부로 타겟 탐색
    out = []
    for p in dude_root.iterdir():
        if not p.is_dir():
            continue
        cand = p / "eval" / eval_subdir / "master_table.csv"
        if cand.exists():
            out.append(p.name)
    out.sort()
    return out


def top_k(df: pd.DataFrame, k: int, score_col: str) -> pd.DataFrame:
    # confidence 큰 값이 상위라고 가정 (DiffDock confidence)
    if len(df) == 0:
        return df
    k = max(1, int(k))
    return df.sort_values(score_col, ascending=False).head(k)


def compute_rates_from_inference_status(
    status_df: Optional[pd.DataFrame],
    total_n: int,
) -> Tuple[float, float, float, float, int, int]:
    """
    fail_rate = N_fail / total_n
    skip_rate = (N_skip_conf + N_skip_test) / total_n
    retry_rate = fail_rate + skip_rate
    coverage_rate = 1 - retry_rate
    """
    if total_n <= 0:
        return (np.nan, np.nan, np.nan, np.nan, 0, 0)

    fail_n = 0
    skip_n = 0
    if status_df is not None and len(status_df) > 0 and "status" in status_df.columns:
        st = status_df["status"].astype(str)
        fail_n = int((st == "fail").sum())
        skip_n = int(st.str.startswith("skip").sum())

    fail_rate = fail_n / total_n
    skip_rate = skip_n / total_n
    retry_rate = fail_rate + skip_rate
    coverage_rate = 1.0 - retry_rate
    return (coverage_rate, fail_rate, skip_rate, retry_rate, fail_n, skip_n)


def compute_struct_qc_rates(
    df: pd.DataFrame,
    top_frac: float = 0.01,
) -> Tuple[float, float, float, float]:
    """
    clash_rate_all: (clash_count > 0).mean()
    clash_rate_top1pct: among top1% by confidence
    pocket_in_rate_all: pocket_in == True mean
    pocket_in_rate_top1pct: among top1%
    """
    if df is None or len(df) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    # robust columns
    if "clash_count" in df.columns:
        clash = pd.to_numeric(df["clash_count"], errors="coerce").fillna(0.0)
    else:
        clash = pd.Series([np.nan] * len(df))

    if "pocket_in" in df.columns:
        pocket_in = df["pocket_in"].astype("boolean")
    else:
        pocket_in = pd.Series([pd.NA] * len(df), dtype="boolean")

    if "confidence" in df.columns:
        score_col = "confidence"
    else:
        score_col = None

    clash_rate_all = float((clash > 0).mean()) if clash.notna().any() else np.nan
    pocket_in_rate_all = float(pocket_in.fillna(False).mean()) if pocket_in.notna().any() else np.nan

    if score_col is None:
        return (clash_rate_all, np.nan, pocket_in_rate_all, np.nan)

    k = max(1, int(len(df) * top_frac))
    df_top = top_k(df, k, score_col=score_col)

    if "clash_count" in df_top.columns:
        clash_top = pd.to_numeric(df_top["clash_count"], errors="coerce").fillna(0.0)
        clash_rate_top = float((clash_top > 0).mean())
    else:
        clash_rate_top = np.nan

    if "pocket_in" in df_top.columns:
        pocket_top = df_top["pocket_in"].astype("boolean")
        pocket_in_rate_top = float(pocket_top.fillna(False).mean())
    else:
        pocket_in_rate_top = np.nan

    return (clash_rate_all, clash_rate_top, pocket_in_rate_all, pocket_in_rate_top)


def compute_comdist_metrics(
    df: pd.DataFrame,
    top_frac: float = 0.01,
    cutoff_A: float = 2.0,
) -> Tuple[float, float, float, float]:
    """
    mean_COMdist: mean(COMdist_A) on all rows
    active_mean_COMdist: mean on label==1
    top1_mean_COMdist: mean on top1% by confidence
    COMdist2rate: actives 기준 (COMdist_A <= cutoff_A) 비율
    """
    if df is None or len(df) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    if "COMdist_A" not in df.columns:
        return (np.nan, np.nan, np.nan, np.nan)

    com = pd.to_numeric(df["COMdist_A"], errors="coerce")
    mean_all = float(com.mean()) if com.notna().any() else np.nan

    if "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce")
        act = df[lab == 1]
    else:
        act = df.iloc[0:0]

    if len(act) > 0:
        com_act = pd.to_numeric(act["COMdist_A"], errors="coerce")
        active_mean = float(com_act.mean()) if com_act.notna().any() else np.nan
        com2rate = float((com_act <= cutoff_A).mean()) if com_act.notna().any() else np.nan
    else:
        active_mean = np.nan
        com2rate = np.nan

    if "confidence" in df.columns:
        k = max(1, int(len(df) * top_frac))
        df_top = top_k(df, k, score_col="confidence")
        com_top = pd.to_numeric(df_top["COMdist_A"], errors="coerce")
        top1_mean = float(com_top.mean()) if com_top.notna().any() else np.nan
    else:
        top1_mean = np.nan

    return (mean_all, active_mean, top1_mean, com2rate)


def extract_metrics_from_metrics_json(metrics_js: Optional[dict]) -> Dict[str, float]:
    """
    metrics_rank1/metrics.json 구조를 널널하게 파싱한다.
    기대 키:
      - standard.roc_auc
      - dude_style.adjusted_logauc0.001
      - extra_legacy.bedroc   (또는 dude_style/standard에 있을 수도 있어 fallback)
      - extra_legacy.ef_top1pct/5pct/10pct
      - extra_legacy.nef_top1pct/5pct/10pct
    """
    out = {
        "ROC_AUC": np.nan,
        "LogAUC": np.nan,
        "BEDROC": np.nan,
        "EF_1pct": np.nan,
        "EF_5pct": np.nan,
        "EF_10pct": np.nan,
        "nEF_1pct": np.nan,
        "nEF_5pct": np.nan,
        "nEF_10pct": np.nan,
    }
    if not metrics_js:
        return out

    out["ROC_AUC"] = as_float(metrics_js.get("standard", {}).get("roc_auc"), np.nan)

    ds = metrics_js.get("dude_style", {})
    out["LogAUC"] = as_float(ds.get("adjusted_logauc0.001"), ds.get("adjusted_logauc0_001", np.nan))

    extra = metrics_js.get("extra_legacy", {})
    bed = extra.get("bedroc", None)
    if bed is None:
        bed = ds.get("bedroc", None)
    if bed is None:
        bed = metrics_js.get("standard", {}).get("bedroc", None)
    out["BEDROC"] = as_float(bed, np.nan)

    out["EF_1pct"] = as_float(extra.get("ef_top1pct"), np.nan)
    out["EF_5pct"] = as_float(extra.get("ef_top5pct"), np.nan)
    out["EF_10pct"] = as_float(extra.get("ef_top10pct"), np.nan)

    out["nEF_1pct"] = as_float(extra.get("nef_top1pct"), np.nan)
    out["nEF_5pct"] = as_float(extra.get("nef_top5pct"), np.nan)
    out["nEF_10pct"] = as_float(extra.get("nef_top10pct"), np.nan)

    return out


# ----------------------------
# Accumulate-save helpers (build_calibration.py 패턴 이식)
# ----------------------------
def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _upsert_csv(
    path: Path,
    new_df: pd.DataFrame,
    *,
    key_cols: List[str],
    column_order: List[str],
) -> pd.DataFrame:
    """
    기존 CSV가 있으면 읽어서 new_df와 합친 뒤 key_cols로 중복 제거(keep=last)하고 저장한다.
    build_calibration.py의 upsert 로직을 동일하게 사용한다.
    """
    path = Path(path)
    old_df = _safe_read_csv(path)

    if old_df is None:
        merged = new_df.copy()
    else:
        for c in column_order:
            if c not in old_df.columns:
                old_df[c] = np.nan
        old_df = old_df[column_order]

        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")

    for c in column_order:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[column_order]

    ensure_parent(path)
    merged.to_csv(path, index=False)
    return merged


# ----------------------------
# Main
# ----------------------------
COLUMN_ORDER = [
    "target",
    "coverage_rate",
    "fail_rate",
    "skip_rate",
    "retry_rate",
    "ROC_AUC",
    "LogAUC",
    "BEDROC",
    "EF_1pct",
    "EF_5pct",
    "EF_10pct",
    "nEF_1pct",
    "nEF_5pct",
    "nEF_10pct",
    "clash_rate_all",
    "clash_rate_top1pct",
    "pocket_in_rate_all",
    "pocket_in_rate_top1pct",
    "mean_COMdist",
    "active_mean_COMdist",
    "top1_mean_COMdist",
    "COMdist2rate",
]

# diagnostics schema 고정 (빈 리스트여도 헤더 유지)
SKIPPED_COLS = ["target", "reason", "path"]
ERRORS_COLS = ["target", "stage", "path", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dude_root", required=True, help="e.g. /home/.../dataset/DUD-E/dude_raw")
    ap.add_argument("--eval_subdir", required=True, help="e.g. diffdock_2")
    ap.add_argument(
        "--targets",
        default="",
        help="optional: comma-separated targets or space-separated list. "
             "If empty, auto-detect targets with master_table.csv",
    )
    ap.add_argument("--out_csv", required=True)

    # diagnostics (optional)
    ap.add_argument("--skipped_csv", type=Path, default=None)
    ap.add_argument("--errors_csv", type=Path, default=None)

    # comdist-specific options
    ap.add_argument("--top_frac", type=float, default=0.01, help="top fraction for top1% style metrics (default=0.01)")
    ap.add_argument("--comdist_cutoff_A", type=float, default=2.0, help="cutoff for COMdist2rate (default=2.0A)")

    args = ap.parse_args()

    dude_root = Path(args.dude_root)
    eval_subdir = str(args.eval_subdir)

    # parse targets
    targets: List[str]
    if args.targets.strip():
        raw = args.targets.replace(",", " ").split()
        targets = [t.strip() for t in raw if t.strip()]
        targets = sorted(list(dict.fromkeys(targets)))
    else:
        targets = get_targets_auto(dude_root, eval_subdir)

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for target in targets:
        base = dude_root / target / "eval" / eval_subdir
        master_csv = base / "master_table.csv"
        status_csv = base / "inference_status_err.csv"
        metrics_json = base / "metrics_rank1" / "metrics.json"

        if not master_csv.exists():
            skipped.append({
                "target": target,
                "reason": "missing_master_table",
                "path": str(master_csv),
            })
            continue

        try:
            df = pd.read_csv(master_csv)
        except Exception as e:
            errors.append({
                "target": target,
                "stage": "read_master_table",
                "path": str(master_csv),
                "error": repr(e),
            })
            continue

        total_n = int(len(df))

        status_df = safe_read_csv(status_csv)
        (coverage_rate, fail_rate, skip_rate, retry_rate, fail_n, skip_n) = compute_rates_from_inference_status(
            status_df=status_df, total_n=total_n
        )

        mjs = safe_read_json(metrics_json)
        mvals = extract_metrics_from_metrics_json(mjs)

        (clash_all, clash_top, pocket_all, pocket_top) = compute_struct_qc_rates(df, top_frac=float(args.top_frac))

        (mean_cd, active_mean_cd, top1_mean_cd, cd2rate) = compute_comdist_metrics(
            df,
            top_frac=float(args.top_frac),
            cutoff_A=float(args.comdist_cutoff_A),
        )

        row = {
            "target": target,
            "coverage_rate": as_float(coverage_rate),
            "fail_rate": as_float(fail_rate),
            "skip_rate": as_float(skip_rate),
            "retry_rate": as_float(retry_rate),

            "ROC_AUC": as_float(mvals["ROC_AUC"]),
            "LogAUC": as_float(mvals["LogAUC"]),
            "BEDROC": as_float(mvals["BEDROC"]),
            "EF_1pct": as_float(mvals["EF_1pct"]),
            "EF_5pct": as_float(mvals["EF_5pct"]),
            "EF_10pct": as_float(mvals["EF_10pct"]),
            "nEF_1pct": as_float(mvals["nEF_1pct"]),
            "nEF_5pct": as_float(mvals["nEF_5pct"]),
            "nEF_10pct": as_float(mvals["nEF_10pct"]),

            "clash_rate_all": as_float(clash_all),
            "clash_rate_top1pct": as_float(clash_top),
            "pocket_in_rate_all": as_float(pocket_all),
            "pocket_in_rate_top1pct": as_float(pocket_top),

            "mean_COMdist": as_float(mean_cd),
            "active_mean_COMdist": as_float(active_mean_cd),
            "top1_mean_COMdist": as_float(top1_mean_cd),
            "COMdist2rate": as_float(cd2rate),
        }

        row = {k: row.get(k, np.nan) for k in COLUMN_ORDER}
        rows.append(row)

    new_df = pd.DataFrame(rows, columns=COLUMN_ORDER)
    out_csv = Path(args.out_csv)

    # ----------------------------
    # (핵심 수정) 누적 저장: target 키로 upsert
    # ----------------------------
    _ = _upsert_csv(
        out_csv,
        new_df,
        key_cols=["target"],
        column_order=COLUMN_ORDER,
    )

    # diagnostics saves: 헤더 고정 (빈 리스트여도 컬럼 출력)
    if args.skipped_csv is not None:
        ensure_parent(args.skipped_csv)
        pd.DataFrame(skipped, columns=SKIPPED_COLS).to_csv(args.skipped_csv, index=False)

    if args.errors_csv is not None:
        ensure_parent(args.errors_csv)
        pd.DataFrame(errors, columns=ERRORS_COLS).to_csv(args.errors_csv, index=False)

    print(f"[SAVE] {out_csv} new_rows={len(new_df)} targets_requested={len(targets)}")
    if args.skipped_csv is not None:
        print(f"[SAVE] skipped -> {args.skipped_csv} rows={len(skipped)}")
    if args.errors_csv is not None:
        print(f"[SAVE] errors  -> {args.errors_csv} rows={len(errors)}")


if __name__ == "__main__":
    main()
    
'''
DUDE_ROOT="/home/deepfold/users/hosung/dataset/DUD-E/dude_raw"
EVAL_SUBDIR="diffdock_2"

python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/aggregate/build_metrics_summary2.py \
  --dude_root $DUDE_ROOT \
  --eval_subdir $EVAL_SUBDIR \
  --out_csv $DUDE_ROOT/metrics_summary_all_${EVAL_SUBDIR}.csv \
  --skipped_csv $DUDE_ROOT/metrics_summary_all_${EVAL_SUBDIR}.skipped.csv \
  --errors_csv  $DUDE_ROOT/metrics_summary_all_${EVAL_SUBDIR}.errors.csv
'''