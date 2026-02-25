# src/qc/postprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple
import csv
import re


# DiffDock inference.py가 저장하는 파일명 규칙에 맞춘 정규식
# 예: rank1_confidence0.32.sdf, rank10_confidence-1.25.sdf
RANKCONF_RE = re.compile(r"^rank(\d+)_confidence([+-]?\d+(?:\.\d+)?)\.sdf$")


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[List[str]], int]:
    """
    입력 CSV를 읽어 (complex_names, all_rows, idx_complex_name) 반환.

    - complex_names: CSV의 complex_name 컬럼 값 리스트(중복 가능)
    - all_rows: 헤더 포함 원본 rows (retry.csv 생성에 사용)
    - idx_complex_name: 헤더에서 complex_name 인덱스
    """
    csv_path = Path(csv_path)
    rows: List[List[str]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    header = rows[0]
    if "complex_name" not in header:
        raise ValueError(f"'complex_name' column not found. Header={header}")

    idx = header.index("complex_name")
    complex_names = []
    for r in rows[1:]:
        if len(r) <= idx:
            continue
        cname = r[idx].strip()
        if cname:
            complex_names.append(cname)

    return complex_names, rows, idx


def scan_best_rank_conf(results_dir: Path, max_rank: int = 10) -> Dict[str, Tuple[int, float]]:
    """
    results_dir/<complex_name>/rankK_confidenceX.sdf 를 스캔해
    complex_name -> (rank_used, confidence) 를 계산한다.

    정책:
    - rank1이 존재하면 무조건 rank1 사용
    - rank1이 없으면 rank2..max_rank 중 confidence 최대를 사용
    """
    results_dir = Path(results_dir)
    best: Dict[str, Tuple[int, float]] = {}

    if not results_dir.exists():
        return best

    # 구조를 명시적으로 강제: 1-depth complex dir 아래 rank 파일
    for sdf in results_dir.glob("*/rank*_confidence*.sdf"):
        if not sdf.is_file():
            continue

        m = RANKCONF_RE.match(sdf.name)
        if not m:
            continue

        rank = int(m.group(1))
        if rank < 1 or rank > max_rank:
            continue

        conf = float(m.group(2))
        cname = sdf.parent.name

        # rank1 우선
        if rank == 1:
            best[cname] = (1, conf)
            continue

        # 이미 rank1 있으면 유지
        if cname in best and best[cname][0] == 1:
            continue

        # fallback: confidence 최대
        if (cname not in best) or (conf > best[cname][1]):
            best[cname] = (rank, conf)

    return best


def write_list_txt(path: Path, items: List[str]) -> None:
    """
    txt에 한 줄에 하나씩 저장.
    """
    path = Path(path)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def write_retry_csv(path: Path, input_rows: List[List[str]], idx_cname: int, missing: Set[str]) -> int:
    """
    원본 CSV에서 missing complex만 골라 retry.csv 생성.
    반환: retry row 수(헤더 제외)
    """
    path = Path(path)
    out_rows = [input_rows[0]]  # header

    for r in input_rows[1:]:
        if len(r) <= idx_cname:
            continue
        cname = r[idx_cname].strip()
        if cname in missing:
            out_rows.append(r)

    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    return max(0, len(out_rows) - 1)


def write_scores_ok_csv(path: Path, ok_best: Dict[str, Tuple[int, float]], label: int) -> int:
    """
    ok-only score table 생성.
    스키마:
      complex_name, confidence, label, rank_used, has_rank1
    반환: row 수
    """
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["complex_name", "confidence", "label", "rank_used", "has_rank1"])
        for cname in sorted(ok_best.keys()):
            rank_used, conf = ok_best[cname]
            w.writerow([cname, conf, label, rank_used, 1 if rank_used == 1 else 0])

    return len(ok_best)
