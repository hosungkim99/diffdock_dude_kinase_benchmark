# src/qc/diffdock_scores.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


RANKCONF_RE = re.compile(r"^rank(\d+)_confidence([+-]?\d+(?:\.\d+)?)\.sdf$")


def _parse_rank_conf(complex_dir: Path, max_rank: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}

    for fp in complex_dir.glob("rank*_confidence*.sdf"):
        if not fp.is_file():
            continue

        m = RANKCONF_RE.match(fp.name)
        if not m:
            continue

        rank = int(m.group(1))
        if rank < 1 or rank > max_rank:
            continue

        conf = float(m.group(2))
        if rank not in scores or conf > scores[rank]:
            scores[rank] = conf

    return scores


def _choose_score(
    rank_conf: Dict[int, float],
    score_mode: str,
) -> Optional[Tuple[float, int]]:
    if not rank_conf:
        return None

    if score_mode == "rank1":
        if 1 not in rank_conf:
            return None
        return rank_conf[1], 1

    if score_mode == "max":
        rank_used = max(rank_conf.keys(), key=lambda k: rank_conf[k])
        return rank_conf[rank_used], rank_used

    raise ValueError(f"Unknown score_mode: {score_mode}")


def build_global_score_table(
    actives_root: Path,
    decoys_root: Path,
    score_mode: str = "rank1",
    max_rank: int = 10,
) -> List[dict]:

    actives_root = Path(actives_root)
    decoys_root = Path(decoys_root)

    if not actives_root.is_dir():
        raise FileNotFoundError(actives_root)
    if not decoys_root.is_dir():
        raise FileNotFoundError(decoys_root)

    rows: List[dict] = []

    def ingest(root: Path, label: int):
        for cdir in sorted(root.iterdir()):
            if not cdir.is_dir():
                continue

            ligand_id = cdir.name
            rank_conf = _parse_rank_conf(cdir, max_rank=max_rank)

            rep = _choose_score(rank_conf, score_mode)
            if rep is None:
                score = None
                rank_used = None
            else:
                score, rank_used = rep

            rows.append(
                {
                    "ligand_id": ligand_id,
                    "label": label,
                    "score": score,
                    "has_rank1": int(1 in rank_conf),
                    "n_ranks": len(rank_conf),
                    "rank_used": rank_used,
                }
            )

    ingest(actives_root, 1)
    ingest(decoys_root, 0)

    return rows
