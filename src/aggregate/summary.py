# src/aggregate/summary.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Metrics는 "단일 소스 오브 트루스": src/metrics/dude_metrics.py
# - roc_auc(y, s)
# - enrichment_factor(y_sorted, top_frac)
# - nef_from_ef(ef, top_frac, base_rate)
# - bedroc(y_sorted, alpha)
# - roc_curve_sorted(y, s)
# - dude_logauc_adjusted(fpr, tpr, fpr_min, random_logauc_pct)
from src.metrics import dude_metrics


# =========================
# Config
# =========================
@dataclass(frozen=True)
class MetricsSummaryConfig:
    """
    summary.py는 "요약 1-row table"을 만든다.
    성능지표 정의는 src/metrics/dude_metrics.py 와 완전히 동일하게 유지한다.
    """

    # early enrichment fractions
    fprs: Tuple[float, float, float] = (0.01, 0.05, 0.10)

    # DUD-style adjusted LogAUC 설정
    # (주의) dude_metrics.dude_logauc_adjusted는 [fpr_min, 1.0] 구간을 적분한다.
    logauc_fpr_min: float = 0.001
    dude_random_logauc_pct: float = 14.462  # DUD-E baseline (percent)

    # BEDROC alpha
    bedroc_alpha: float = 20.0

    # clash flag threshold: clash_count >= this
    clash_count_ge: int = 5

    # top fraction for top1% stats (QC rates)
    top_frac: float = 0.01


# =========================
# IO helpers
# =========================
def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _read_master_table(master_table_csv: Path) -> pd.DataFrame:
    """
    master_table_csv는 "per-ligand row" 테이블이어야 한다.
    최소 요구 컬럼:
      - target (str)
      - ligand_id (str)
      - label (0/1)
      - success (0/1)
      - confidence (float)
      - pocket_in (bool or 0/1)
      - clash_count (int)
      - COMdist_A (float)  # 없으면 mean_COMdist는 NaN 처리 가능
    """
    master_table_csv = Path(master_table_csv)
    if not master_table_csv.exists():
        raise FileNotFoundError(str(master_table_csv))

    df = pd.read_csv(master_table_csv)

    # 방어적 타입 정리
    for c in ["target", "ligand_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in ["label", "success", "clash_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    if "pocket_in" in df.columns:
        # bool/0/1 혼재 대응
        if df["pocket_in"].dtype != bool:
            df["pocket_in"] = df["pocket_in"].astype(int).astype(bool)

    if "COMdist_A" in df.columns:
        df["COMdist_A"] = pd.to_numeric(df["COMdist_A"], errors="coerce")

    # 필수 컬럼 체크
    required = ["target", "ligand_id", "label", "success", "confidence", "pocket_in", "clash_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"master_table_csv missing required columns: {missing}")

    return df


def _read_err_status(err_status_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    err_status_csv는 parse_inference_err.py 등으로 만든 결과를 가정한다.
    최소 컬럼:
      - ligand_id
      - status  (fail, skip_test_missing, skip_conf_missing, ...)
    """
    if err_status_csv is None:
        return None
    err_status_csv = Path(err_status_csv)
    if not err_status_csv.exists():
        return None

    df = pd.read_csv(err_status_csv)
    if "ligand_id" not in df.columns or "status" not in df.columns:
        return None

    df["ligand_id"] = df["ligand_id"].astype(str)
    df["status"] = df["status"].astype(str)
    return df


def _read_retry_csv(retry_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    retry_csv는 postprocess 단계에서 생성된 retry 리스트를 가정한다.
    ligand_id 컬럼명이 다양할 수 있어 rename 방어를 둔다.
    """
    if retry_csv is None:
        return None
    retry_csv = Path(retry_csv)
    if not retry_csv.exists():
        return None

    df = pd.read_csv(retry_csv)

    # 관용적으로 ligand_id 컬럼명이 다를 수 있어 방어
    for c in ["ligand_id", "complex_name", "name", "id"]:
        if c in df.columns:
            df = df.rename(columns={c: "ligand_id"})
            break
    if "ligand_id" not in df.columns:
        return None

    df["ligand_id"] = df["ligand_id"].astype(str)
    return df


# =========================
# Metrics helpers (delegate to dude_metrics)
# =========================
def _sort_by_score_desc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    return y_true[order]


def _ef_at_frac(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> float:
    y_sorted = _sort_by_score_desc(y_true, y_score)
    ef, _, _, _, _ = dude_metrics.enrichment_factor(y_sorted, float(top_frac))
    return float(ef)


def _nef_at_frac(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> float:
    y_sorted = _sort_by_score_desc(y_true, y_score)
    N = int(len(y_sorted))
    n_act = int(y_sorted.sum())
    base_rate = (n_act / N) if N > 0 else float("nan")

    ef, _, _, _, _ = dude_metrics.enrichment_factor(y_sorted, float(top_frac))
    nef = dude_metrics.nef_from_ef(float(ef), float(top_frac), float(base_rate))
    return float(nef)


def _bedroc_from_scores(y_true: np.ndarray, y_score: np.ndarray, alpha: float) -> float:
    y_sorted = _sort_by_score_desc(y_true, y_score)
    return float(dude_metrics.bedroc(y_sorted, alpha=float(alpha)))


def _dude_logauc_adjusted_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    fpr_min: float,
    random_logauc_pct: float,
) -> float:
    """
    DUD-E 스타일 adjusted LogAUC:
      - ROC curve (fpr,tpr)을 만든 뒤
      - log10(FPR) 축에서 [fpr_min, 1] 적분
      - percent 스케일로 바꾸고
      - random baseline(percent)을 뺀다
    dude_metrics.dude_logauc_adjusted 정의를 그대로 사용한다.
    """
    fpr, tpr = dude_metrics.roc_curve_sorted(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float))
    return float(
        dude_metrics.dude_logauc_adjusted(
            fpr, tpr, fpr_min=float(fpr_min), random_logauc_pct=float(random_logauc_pct)
        )
    )


# =========================
# Main builder
# =========================
def build_metrics_summary(
    *,
    master_table_csv: Path,
    out_csv: Path,
    cfg: MetricsSummaryConfig = MetricsSummaryConfig(),
    err_status_csv: Optional[Path] = None,
    retry_csv: Optional[Path] = None,
    comdist_scope: str = "all_ok",  # NEW: "all_ok" or "actives_ok"
    write_config_json: bool = True,
) -> pd.DataFrame:
    """
    Output: 1-row metrics_summary CSV.

    성능지표는 "성공한 ligands(success==1)"만 사용해 계산한다.
    실패/스킵/리트라이는 err_status_csv 및 retry_csv로 비율만 계산한다.

    mean_COMdist는 comdist_scope로 범위를 선택한다.
      - all_ok: success==1 전체(actives+decoys)
      - actives_ok: success==1이면서 label==1인 actives만
    """
    if comdist_scope not in ("all_ok", "actives_ok"):
        raise ValueError(f"Unknown comdist_scope: {comdist_scope} (use 'all_ok' or 'actives_ok')")

    master = _read_master_table(Path(master_table_csv))

    # base set: 성공한 ligands(=success==1)로 metric 계산
    ok = master[master["success"].fillna(0).astype(int) == 1].copy()
    y_true = ok["label"].to_numpy(dtype=int)
    y_score = ok["confidence"].to_numpy(dtype=float)

    target = str(master["target"].iloc[0]) if len(master) else ""

    # ---------- rates: coverage / fail / skip / retry ----------
    n_success = int(ok.shape[0])

    err_df = _read_err_status(err_status_csv)
    n_fail = 0
    n_skip = 0
    if err_df is not None and len(err_df) > 0:
        # fail: status == "fail"
        n_fail = int((err_df["status"] == "fail").sum())
        # skip: status startswith "skip_"
        n_skip = int((err_df["status"].str.startswith("skip_")).sum())

    total = n_success + n_fail + n_skip
    if total > 0:
        coverage_rate = n_success / total
        fail_rate = n_fail / total
        skip_rate = n_skip / total
    else:
        coverage_rate = float("nan")
        fail_rate = float("nan")
        skip_rate = float("nan")

    retry_df = _read_retry_csv(retry_csv)
    if retry_df is not None and total > 0:
        n_retry = int(retry_df["ligand_id"].nunique())
        retry_rate = n_retry / total
    else:
        retry_rate = float("nan")

    # ---------- metrics (delegate to dude_metrics) ----------
    if len(ok) > 0:
        roc_auc = dude_metrics.roc_auc(y_true, y_score)
        logauc = _dude_logauc_adjusted_from_scores(
            y_true, y_score, fpr_min=cfg.logauc_fpr_min, random_logauc_pct=cfg.dude_random_logauc_pct
        )
        bedroc = _bedroc_from_scores(y_true, y_score, cfg.bedroc_alpha)

        ef_1 = _ef_at_frac(y_true, y_score, cfg.fprs[0])
        ef_5 = _ef_at_frac(y_true, y_score, cfg.fprs[1])
        ef_10 = _ef_at_frac(y_true, y_score, cfg.fprs[2])

        nef_1 = _nef_at_frac(y_true, y_score, cfg.fprs[0])
        nef_5 = _nef_at_frac(y_true, y_score, cfg.fprs[1])
        nef_10 = _nef_at_frac(y_true, y_score, cfg.fprs[2])
    else:
        roc_auc = float("nan")
        logauc = float("nan")
        bedroc = float("nan")
        ef_1 = float("nan")
        ef_5 = float("nan")
        ef_10 = float("nan")
        nef_1 = float("nan")
        nef_5 = float("nan")
        nef_10 = float("nan")

    # ---------- QC rates ----------
    # clash_flag := clash_count >= clash_count_ge
    if len(ok) > 0:
        clash_count = ok["clash_count"].fillna(0).astype(int).to_numpy()
        clash_flag = (clash_count >= int(cfg.clash_count_ge))
        clash_rate_all = float(clash_flag.mean())

        pocket_in_rate_all = float(ok["pocket_in"].to_numpy(dtype=bool).mean())
    else:
        clash_rate_all = float("nan")
        pocket_in_rate_all = float("nan")

    # top1% subset by confidence
    if len(ok) > 0:
        top_k = int(np.ceil(float(cfg.top_frac) * len(ok)))
        top_k = max(1, min(len(ok), top_k))
        top = ok.sort_values("confidence", ascending=False).head(top_k)

        top_clash = top["clash_count"].fillna(0).astype(int).to_numpy()
        clash_rate_top1pct = float((top_clash >= int(cfg.clash_count_ge)).mean())
        pocket_in_rate_top1pct = float(top["pocket_in"].to_numpy(dtype=bool).mean())
    else:
        clash_rate_top1pct = float("nan")
        pocket_in_rate_top1pct = float("nan")

    # ---------- COMdist mean (selectable scope) ----------
    if ("COMdist_A" in ok.columns) and (len(ok) > 0):
        if comdist_scope == "actives_ok":
            ok_com = ok[ok["label"].fillna(0).astype(int) == 1]
        else:
            ok_com = ok

        mean_comdist = float(ok_com["COMdist_A"].mean()) if len(ok_com) > 0 else float("nan")
    else:
        mean_comdist = float("nan")

    out = pd.DataFrame([{
        "target": target,
        "coverage_rate": _safe_float(coverage_rate),
        "fail_rate": _safe_float(fail_rate),
        "skip_rate": _safe_float(skip_rate),
        "retry_rate": _safe_float(retry_rate),

        "ROC_AUC": _safe_float(roc_auc),
        # DUD-style adjusted LogAUC (percent - random baseline)
        "LogAUC": _safe_float(logauc),
        "BEDROC": _safe_float(bedroc),

        "EF_1pct": _safe_float(ef_1),
        "EF_5pct": _safe_float(ef_5),
        "EF_10pct": _safe_float(ef_10),

        "nEF_1pct": _safe_float(nef_1),
        "nEF_5pct": _safe_float(nef_5),
        "nEF_10pct": _safe_float(nef_10),

        "clash_rate_all": _safe_float(clash_rate_all),
        "clash_rate_top1pct": _safe_float(clash_rate_top1pct),
        "pocket_in_rate_all": _safe_float(pocket_in_rate_all),
        "pocket_in_rate_top1pct": _safe_float(pocket_in_rate_top1pct),

        "mean_COMdist": _safe_float(mean_comdist),
    }])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    if write_config_json:
        cfg_path = out_csv.with_suffix(".config.json")
        cfg_obj = asdict(cfg)
        cfg_obj["comdist_scope"] = comdist_scope  # NEW: 기록

        payload = {
            "master_table_csv": str(master_table_csv),
            "err_status_csv": str(err_status_csv) if err_status_csv else "",
            "retry_csv": str(retry_csv) if retry_csv else "",
            "config": cfg_obj,
            "notes": {
                "LogAUC": "DUD-style adjusted LogAUC = dude_metrics.dude_logauc_adjusted(fpr_min, random_baseline)",
                "missing_policy": "summary computes metrics on success==1 only (missing already excluded upstream).",
                "mean_COMdist": "Computed from COMdist_A over scope controlled by comdist_scope.",
            },
        }
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return out


# =========================
# CLI
# =========================
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_table_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--err_status_csv", default="")
    ap.add_argument("--retry_csv", default="")

    # NEW: comdist scope
    ap.add_argument(
        "--comdist_scope",
        choices=["all_ok", "actives_ok"],
        default="all_ok",
        help="mean_COMdist scope: all_ok(success actives+decoys) or actives_ok(success actives only)",
    )

    # overrides
    ap.add_argument("--bedroc_alpha", type=float, default=20.0)
    ap.add_argument("--logauc_fpr_min", type=float, default=0.001)
    ap.add_argument("--dude_random_logauc_pct", type=float, default=14.462)
    ap.add_argument("--clash_count_ge", type=int, default=5)
    ap.add_argument("--top_frac", type=float, default=0.01)

    args = ap.parse_args()

    cfg = MetricsSummaryConfig(
        bedroc_alpha=float(args.bedroc_alpha),
        logauc_fpr_min=float(args.logauc_fpr_min),
        dude_random_logauc_pct=float(args.dude_random_logauc_pct),
        clash_count_ge=int(args.clash_count_ge),
        top_frac=float(args.top_frac),
    )

    build_metrics_summary(
        master_table_csv=Path(args.master_table_csv),
        out_csv=Path(args.out_csv),
        cfg=cfg,
        err_status_csv=Path(args.err_status_csv) if args.err_status_csv.strip() else None,
        retry_csv=Path(args.retry_csv) if args.retry_csv.strip() else None,
        comdist_scope=str(args.comdist_scope),
    )


if __name__ == "__main__":
    _cli()