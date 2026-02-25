# src/qc/report.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def _bucket(conf: float, high_thr: float, low_thr: float) -> str:
    """
    버킷 정책:
      - high: conf > high_thr
      - moderate: low_thr <= conf <= high_thr
      - low: conf < low_thr
    """
    if conf > high_thr:
        return "high"
    if conf < low_thr:
        return "low"
    return "moderate"


def build_qc_report(
    target: str,
    split: str,
    scores_ok_csv: Path,
    out_dir: Path,
    top_k: int = 50,
    bottom_k: int = 50,
    high_thr: float = 0.0,
    low_thr: float = -1.5,
) -> Dict:
    """
    split별 ok-only scores csv를 입력으로 QC report 산출물을 생성한다.

    입력 CSV 스키마(필수):
      - complex_name
      - confidence

    생성물:
      - {target}_{split}_qc_summary.txt
      - {target}_{split}_top{K}.csv
      - {target}_{split}_bottom{K}.csv
      - {target}_{split}_bucket_high.txt
      - {target}_{split}_bucket_moderate.txt
      - {target}_{split}_bucket_low.txt

    반환:
      summary dict
    """
    scores_ok_csv = Path(scores_ok_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_ok_csv)

    # 최소 스키마 체크
    if "complex_name" not in df.columns or "confidence" not in df.columns:
        raise ValueError(
            f"scores_ok_csv must contain columns: complex_name, confidence. got={list(df.columns)}"
        )

    df["complex_name"] = df["complex_name"].astype(str).str.strip()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["complex_name", "confidence"]).copy()

    if len(df) == 0:
        raise ValueError(f"No valid rows after cleaning: {scores_ok_csv}")

    # 버킷 할당
    df["bucket"] = df["confidence"].apply(lambda x: _bucket(float(x), high_thr=high_thr, low_thr=low_thr))

    # 정렬/탑바텀
    df_sorted = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    top_df = df_sorted.head(top_k).copy()
    bottom_df = df_sorted.tail(bottom_k).sort_values("confidence", ascending=True).copy()

    # 버킷별 리스트
    bucket_high = df[df["bucket"] == "high"].sort_values("confidence", ascending=False)
    bucket_mod = df[df["bucket"] == "moderate"].sort_values("confidence", ascending=False)
    bucket_low = df[df["bucket"] == "low"].sort_values("confidence", ascending=False)

    # 파일 저장
    summary_txt = out_dir / f"{target}_{split}_qc_summary.txt"
    top_csv = out_dir / f"{target}_{split}_top{top_k}.csv"
    bottom_csv = out_dir / f"{target}_{split}_bottom{bottom_k}.csv"
    high_txt = out_dir / f"{target}_{split}_bucket_high.txt"
    mod_txt = out_dir / f"{target}_{split}_bucket_moderate.txt"
    low_txt = out_dir / f"{target}_{split}_bucket_low.txt"

    top_df[["complex_name", "confidence", "bucket"]].to_csv(top_csv, index=False)
    bottom_df[["complex_name", "confidence", "bucket"]].to_csv(bottom_csv, index=False)

    def _write_bucket_txt(path: Path, sdf: pd.DataFrame):
        # 한 줄: complex_name,confidence
        lines = [f"{r.complex_name},{float(r.confidence)}" for r in sdf[["complex_name", "confidence"]].itertuples(index=False)]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    _write_bucket_txt(high_txt, bucket_high)
    _write_bucket_txt(mod_txt, bucket_mod)
    _write_bucket_txt(low_txt, bucket_low)

    # 요약 통계
    n = len(df)
    n_high = int((df["bucket"] == "high").sum())
    n_mod = int((df["bucket"] == "moderate").sum())
    n_low = int((df["bucket"] == "low").sum())

    conf_min = float(df["confidence"].min())
    conf_max = float(df["confidence"].max())
    conf_mean = float(df["confidence"].mean())
    conf_median = float(df["confidence"].median())

    summary = {
        "target": target,
        "split": split,
        "n_ok": n,
        "bucket_thresholds": {"high_thr": high_thr, "low_thr": low_thr},
        "bucket_counts": {"high": n_high, "moderate": n_mod, "low": n_low},
        "confidence_stats": {
            "min": conf_min,
            "max": conf_max,
            "mean": conf_mean,
            "median": conf_median,
        },
        "outputs": {
            "summary_txt": str(summary_txt),
            "top_csv": str(top_csv),
            "bottom_csv": str(bottom_csv),
            "bucket_high_txt": str(high_txt),
            "bucket_moderate_txt": str(mod_txt),
            "bucket_low_txt": str(low_txt),
        },
    }

    # summary 텍스트 작성(README/보고서에 바로 복사 가능하게)
    summary_lines = [
        "=== DiffDock QC Report (split-level, ok-only) ===",
        f"target: {target}",
        f"split : {split}",
        f"n_ok  : {n}",
        "",
        f"bucket thresholds:",
        f"  high     : conf > {high_thr}",
        f"  moderate : {low_thr} <= conf <= {high_thr}",
        f"  low      : conf < {low_thr}",
        "",
        "bucket counts:",
        f"  high     : {n_high}",
        f"  moderate : {n_mod}",
        f"  low      : {n_low}",
        "",
        "confidence stats:",
        f"  min    : {conf_min:.4f}",
        f"  max    : {conf_max:.4f}",
        f"  mean   : {conf_mean:.4f}",
        f"  median : {conf_median:.4f}",
        "",
        "outputs:",
        f"  top_csv          : {top_csv}",
        f"  bottom_csv       : {bottom_csv}",
        f"  bucket_high_txt  : {high_txt}",
        f"  bucket_moderate  : {mod_txt}",
        f"  bucket_low_txt   : {low_txt}",
    ]
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return summary
