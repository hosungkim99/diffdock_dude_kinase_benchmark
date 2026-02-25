# src/qc/rank1_bundle.py
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any


def run_rank1_qc(
    scores_csv: Path,
    metrics_dir: Path,
    topk: int = 100,
) -> Dict[str, Any]:

    df = pd.read_csv(scores_csv)
    ranking = pd.read_csv(metrics_dir / "ranking.csv")
    roc = pd.read_csv(metrics_dir / "roc.csv")
    metrics = json.load(open(metrics_dir / "metrics.json", "r", encoding="utf-8"))

    qc = {}

    # ---------- A) Data integrity ----------
    qc["n_rows_scores_csv"] = int(len(df))
    qc["n_unique_ligand_id_scores_csv"] = int(df["ligand_id"].nunique())
    qc["n_duplicates_scores_csv"] = int(len(df) - df["ligand_id"].nunique())
    qc["label_counts_scores_csv"] = df["label"].value_counts(dropna=False).to_dict()

    if "score" in df.columns:
        missing_src = df["score"].isna().sum() + (df["score"].astype(str).str.strip() == "").sum()
        qc["missing_score_in_scores_csv"] = int(missing_src)

    # ---------- B) Ranking sanity ----------
    base_rate = float((df["label"] == 1).mean())
    qc["base_rate_active"] = base_rate

    topk = min(topk, len(ranking))
    top = ranking.head(topk)

    qc["topk"] = int(topk)
    qc["topk_active_count"] = int(top["label"].sum())
    qc["topk_active_rate"] = float(top["label"].mean())
    qc["topk_enrichment_over_base"] = (
        float(top["label"].mean() / base_rate) if base_rate > 0 else float("nan")
    )

    qc["n_neg_inf_in_ranking"] = int((ranking["score"] == float("-inf")).sum())
    qc["tail20_neg_inf_count"] = int((ranking.tail(20)["score"] == float("-inf")).sum())

    # ---------- C) ROC sanity ----------
    qc["roc_first"] = {
        "fpr": float(roc.iloc[0]["fpr"]),
        "tpr": float(roc.iloc[0]["tpr"]),
    }
    qc["roc_last"] = {
        "fpr": float(roc.iloc[-1]["fpr"]),
        "tpr": float(roc.iloc[-1]["tpr"]),
    }

    qc["roc_monotonic_fpr"] = bool((roc["fpr"].diff().fillna(0) >= -1e-12).all())
    qc["roc_monotonic_tpr"] = bool((roc["tpr"].diff().fillna(0) >= -1e-12).all())

    # ---------- D) Metrics snapshot ----------
    qc["metrics_snapshot"] = {
        "actives": metrics["input"]["actives"],
        "decoys": metrics["input"]["decoys"],
        "missing_policy": metrics["input"]["missing_policy"],
        "roc_auc": metrics["standard"]["roc_auc"],
        "adjusted_logauc0.001": metrics["dude_style"]["adjusted_logauc0.001"],
        "roc_ef1_percent": metrics["dude_style"]["roc_ef1_percent"],
    }

    return qc
