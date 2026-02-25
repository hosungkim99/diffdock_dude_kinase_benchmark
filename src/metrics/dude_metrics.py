# src/metrics/dude_metrics.py
from __future__ import annotations

import csv, math, json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np


def load_csv(path: str | Path, missing_policy: str = "drop"):
    """
    missing_policy:
      - "drop"   : 기존 동작(점수 없으면 제외)
      - "bottom" : 점수 없으면 -inf로 넣어서 최하위 처리(보수적, 추천)
    """
    y = []
    s = []
    ligand_id = []
    missing = 0

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lid = row.get("ligand_id", "")
            label = int(float(row["label"]))
            score = row.get("score", "")
            has_rank1 = row.get("has_rank1", "1")

            # rank1 평가에서는 has_rank1==0을 missing으로 간주
            if has_rank1 != "" and int(float(has_rank1)) == 0:
                score = ""

            if score == "" or str(score).lower() == "none":
                missing += 1
                if missing_policy == "drop":
                    continue
                if missing_policy == "bottom":
                    y.append(label)
                    s.append(float("-inf"))
                    ligand_id.append(lid)
                    continue
                raise ValueError(f"Unknown missing_policy: {missing_policy}")

            y.append(label)
            s.append(float(score))
            ligand_id.append(lid)

    return np.array(y, dtype=int), np.array(s, dtype=float), ligand_id, missing


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos, dtype=int), np.zeros_like(neg, dtype=int)])

    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)

    # ties: average ranks
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def enrichment_factor(y_sorted: np.ndarray, top_frac: float):
    N = len(y_sorted)
    n_act = int(y_sorted.sum())
    if n_act == 0:
        return float("nan"), 0, 0, 0, N
    k = max(1, int(math.ceil(N * top_frac)))
    hits = int(y_sorted[:k].sum())
    ef = (hits / k) / (n_act / N)
    return float(ef), int(hits), int(k), int(n_act), int(N)


def log_auc(y_sorted: np.ndarray, alpha: float = 0.1) -> float:
    N = len(y_sorted)
    n_act = int(y_sorted.sum())
    if n_act == 0:
        return float("nan")

    tp = np.cumsum(y_sorted)
    x = np.arange(1, N + 1) / N
    tpr = tp / n_act

    mask = x <= alpha
    if not np.any(mask):
        return float("nan")

    x2 = x[mask]
    y2 = tpr[mask]
    lx = np.log10(x2)

    # normalize by log-range
    auc_log = np.trapz(y2, lx) / (math.log10(alpha) - lx[0])
    return float(auc_log)


import numpy as np

def bedroc(y_sorted: np.ndarray, alpha: float = 20.0) -> float:
    N = int(len(y_sorted))
    n = int(y_sorted.sum())
    if N == 0 or n == 0:
        return float("nan")

    ranks = np.where(y_sorted == 1)[0] + 1  # 1..N

    # 안정적 계산: 1 - exp(-x) = -expm1(-x)
    one_minus_e_alpha  = -np.expm1(-alpha)          # 1 - e^{-alpha}
    one_minus_e_alphaN = -np.expm1(-alpha / N)      # 1 - e^{-alpha/N}

    # 네가 쓰는 RIE 정의(스케일 포함)
    rie = (np.exp(-alpha * ranks / N).mean()) * (one_minus_e_alpha / one_minus_e_alphaN)

    # RIE_min (random baseline) - 너의 기존 정의 유지
    rie_min = one_minus_e_alpha / alpha

    # RIE_max: "active가 1..n에 모두 몰림" 케이스를 같은 정의로 계산
    e_alphaN = np.exp(-alpha / N)
    rie_max_meanexp = (e_alphaN * (1 - np.exp(-alpha * n / N))) / (n * (1 - e_alphaN))
    rie_max = rie_max_meanexp * (one_minus_e_alpha / one_minus_e_alphaN)

    denom = rie_max - rie_min
    if denom <= 0:
        return float("nan")
    
    print("rie:", rie)
    print("rie_min:", rie_min)
    print("rie_max:", rie_max)
    print("raw:", (rie - rie_min)/(rie_max - rie_min))
    val = (rie - rie_min) / denom
    val = (rie - rie_min) / denom
    # clip은 유지해도 되지만, 디버깅 단계에선 raw 값도 같이 로그로 보길 권장
    return float(np.clip(val, 0.0, 1.0))

    


def nef_from_ef(ef: float, top_frac: float, base_rate: float) -> float:
    """
    nEF 정의(현재 파일의 구현과 동일):
      EF_max(p) = min(1/p, 1/base_rate)
      nEF = clip((EF-1)/(EF_max-1), 0, 1)
    """
    if not np.isfinite(ef) or not np.isfinite(base_rate) or base_rate <= 0:
        return float("nan")
    ef_max = min(1.0 / top_frac, 1.0 / base_rate)
    if ef_max <= 1.0:
        return 0.0
    x = (ef - 1.0) / (ef_max - 1.0)
    return float(max(0.0, min(1.0, x)))


def roc_curve_sorted(y: np.ndarray, s: np.ndarray):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([float("nan"), float("nan")])

    order = np.argsort(-s)
    y_sorted = y[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = tp / pos
    fpr = fp / neg

    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])

    if fpr[-1] < 1.0:
        fpr = np.concatenate([fpr, [1.0]])
        tpr = np.concatenate([tpr, [1.0]])

    return fpr, tpr


def _interp_at(x, y, xq: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])
    return float(np.interp(xq, x, y))


def dude_logauc_adjusted(fpr, tpr, fpr_min: float = 0.001, random_logauc_pct: float = 14.462) -> float:
    if np.any(np.isnan(tpr)) or np.any(np.isnan(fpr)):
        return float("nan")

    fpr_min = float(fpr_min)
    if fpr_min <= 0.0 or fpr_min >= 1.0:
        raise ValueError("fpr_min must be in (0,1).")

    tpr_at_min = _interp_at(fpr, tpr, fpr_min)
    tpr_at_1 = _interp_at(fpr, tpr, 1.0)

    mask = (fpr >= fpr_min) & (fpr <= 1.0)
    f = fpr[mask]
    t = tpr[mask]

    if len(f) == 0 or f[0] > fpr_min:
        f = np.concatenate([[fpr_min], f])
        t = np.concatenate([[tpr_at_min], t])
    elif f[0] < fpr_min:
        f[0] = fpr_min
        t[0] = tpr_at_min

    if f[-1] < 1.0:
        f = np.concatenate([f, [1.0]])
        t = np.concatenate([t, [tpr_at_1]])
    else:
        f[-1] = 1.0
        t[-1] = tpr_at_1

    lx = np.log10(f)
    denom = (math.log10(1.0) - math.log10(fpr_min))  # log10(1/fpr_min)
    raw = np.trapz(t, lx) / denom
    raw_pct = 100.0 * raw
    return float(raw_pct - float(random_logauc_pct))


def dude_roc_ef_at_fpr(fpr, tpr, fpr_thresh: float = 0.01) -> float:
    if np.any(np.isnan(tpr)) or np.any(np.isnan(fpr)):
        return float("nan")
    tpr_at = _interp_at(fpr, tpr, float(fpr_thresh))
    return float(100.0 * tpr_at)


def save_outputs(outdir: Path, args, ligand_id, y, s, fpr, tpr, metrics: Dict[str, Any]):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # config.json
    config = {k: getattr(args, k) for k in vars(args).keys()}
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # metrics.json
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # roc.csv
    with (outdir / "roc.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr"])
        for x, yv in zip(fpr, tpr):
            w.writerow([float(x), float(yv)])

    # ranking.csv (descending by score)
    order = np.argsort(-s)
    with (outdir / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "ligand_id", "label", "score"])
        for r, i in enumerate(order, start=1):
            w.writerow([r, ligand_id[i], int(y[i]), float(s[i])])


def evaluate_from_scores_csv(
    scores_csv: str | Path,
    *,
    missing_policy: str = "drop",
    alpha_logauc: float = 0.1,
    alpha_bedroc: float = 20.0,
    dude_fpr_min: float = 0.001,
    dude_random_logauc_pct: float = 14.462,
    dude_ef_fpr: float = 0.01,
):
    """
    CLI(main)에서 쓰기 편하게 묶은 평가 함수.
    (원본 스크립트의 계산 순서/정의/출력 값과 동일)
    """
    y, s, ligand_id, missing = load_csv(scores_csv, missing_policy=missing_policy)

    order = np.argsort(-s)
    y_sorted = y[order]

    auc = roc_auc(y, s)

    ef1, hits1, k1, n_act, N = enrichment_factor(y_sorted, 0.01)
    ef5, hits5, k5, _, _ = enrichment_factor(y_sorted, 0.05)
    ef10, hits10, k10, _, _ = enrichment_factor(y_sorted, 0.10)

    base_rate = (n_act / N) if N > 0 else float("nan")
    nef1 = nef_from_ef(ef1, 0.01, base_rate)
    nef5 = nef_from_ef(ef5, 0.05, base_rate)
    nef10 = nef_from_ef(ef10, 0.10, base_rate)

    la_old = log_auc(y_sorted, alpha=alpha_logauc)
    bd = bedroc(y_sorted, alpha=alpha_bedroc)

    fpr, tpr = roc_curve_sorted(y, s)
    dude_logauc = dude_logauc_adjusted(fpr, tpr, fpr_min=dude_fpr_min, random_logauc_pct=dude_random_logauc_pct)
    dude_ef1_pct = dude_roc_ef_at_fpr(fpr, tpr, fpr_thresh=dude_ef_fpr)

    metrics = {
        "input": {
            "scores_csv": str(scores_csv),
            "missing_policy": missing_policy,
            "used_ligands": int(len(y)),
            "dropped_missing_score": int(missing) if missing_policy == "drop" else 0,
            "missing_score_total": int(missing),
            "actives": int((y == 1).sum()),
            "decoys": int((y == 0).sum()),
            "base_rate": float(base_rate),
        },
        "dude_style": {
            "adjusted_logauc_fpr_min": float(dude_fpr_min),
            "adjusted_logauc_random_baseline_percent": float(dude_random_logauc_pct),
            "adjusted_logauc0.001": float(dude_logauc),
            "roc_ef_fpr": float(dude_ef_fpr),
            "roc_ef1_percent": float(dude_ef1_pct),
        },
        "standard": {
            "roc_auc": float(auc),
        },
        "extra_legacy": {
            "ef_top1pct": float(ef1), "ef_top1pct_hits": int(hits1), "ef_top1pct_k": int(k1),
            "ef_top5pct": float(ef5), "ef_top5pct_hits": int(hits5), "ef_top5pct_k": int(k5),
            "ef_top10pct": float(ef10), "ef_top10pct_hits": int(hits10), "ef_top10pct_k": int(k10),

            "nef_top1pct": float(nef1),
            "nef_top5pct": float(nef5),
            "nef_top10pct": float(nef10),

            "logauc_rank_over_alpha": float(la_old),
            "logauc_alpha": float(alpha_logauc),
            "bedroc": float(bd),
            "bedroc_alpha": float(alpha_bedroc),
        }
    }

    extras = {
        "auc": auc,
        "ef1": ef1, "ef5": ef5, "ef10": ef10,
        "hits1": hits1, "hits5": hits5, "hits10": hits10,
        "k1": k1, "k5": k5, "k10": k10,
        "nef1": nef1, "nef5": nef5, "nef10": nef10,
        "la_old": la_old,
        "bd": bd,
        "dude_logauc": dude_logauc,
        "dude_ef1_pct": dude_ef1_pct,
        "missing": missing,
        "base_rate": base_rate,
        "ligand_id": ligand_id,
        "y": y,
        "s": s,
        "fpr": fpr,
        "tpr": tpr,
    }
    return metrics, extras
