#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- patterns (user-defined contract) ---
# Failed case:
#   ... WARNING -Failed on ['wee1_decoy_006137']: No edges and no nodes
RE_FAIL = re.compile(r"Failed on \['([^']+)'\]")

# Skipped-confidence dataset case:
#   ... WARNING -The confidence dataset did not contain ['csf1r_decoy_008645']...
RE_SKIP_CONF = re.compile(r"The confidence dataset did not contain \['([^']+)'\]")

# Skipped-test dataset case:
#   ... WARNING -The test dataset did not contain wee1_decoy_001450 for ...
RE_SKIP_TEST = re.compile(r"The test dataset did not contain\s+(\S+)\s+for\b")

# filename: diffdock_{target}_{split}_{jobid}.err
FNAME_RE = re.compile(
    r"diffdock_(?P<target>[^_]+)_(?P<split>actives|decoys)_(?P<jobid>\d+)\.err$"
)

PRIORITY = {
    "fail": 3,
    "skip_conf_missing": 2,
    "skip_test_missing": 1,
}


def classify_line(line: str) -> str | None:
    """Classify a single err line into one of (fail, skip_conf_missing, skip_test_missing)."""
    if RE_FAIL.search(line):
        # optional guard:
        # if "No edges and no nodes" not in line: return None
        return "fail"
    if RE_SKIP_CONF.search(line):
        return "skip_conf_missing"
    if RE_SKIP_TEST.search(line):
        return "skip_test_missing"
    return None


def fallback_token(line: str, target: str) -> str | None:
    """
    Fallback: search {target}_{active|decoy}_{digits} anywhere in the line.
    This is robust against prefix tqdm strings like: "1449it [11:29:08, 27.88s/it]..."
    """
    m = re.search(rf"\b{re.escape(target)}_(?:active|decoy)_\d+\b", line)
    return m.group(0) if m else None


def extract_ligand_id(line: str, st: str, target: str) -> str | None:
    """
    Extract ligand_id based on the classified status, using the correct regex for each case.
    Do NOT slice the line or take the first [...] block, because progress bars and timestamps also use [].
    """
    if st == "fail":
        m = RE_FAIL.search(line)
        if m:
            return m.group(1)
        return fallback_token(line, target)

    if st == "skip_conf_missing":
        m = RE_SKIP_CONF.search(line)
        if m:
            return m.group(1)
        return fallback_token(line, target)

    if st == "skip_test_missing":
        m = RE_SKIP_TEST.search(line)
        if m:
            return m.group(1)
        return fallback_token(line, target)

    return None


def update_status(cur: str | None, new: str) -> str:
    """Keep the higher-priority status if a ligand appears multiple times."""
    if cur is None:
        return new
    return new if PRIORITY[new] > PRIORITY.get(cur, 0) else cur


def parse_err_filename(path: Path) -> dict:
    m = FNAME_RE.search(path.name)
    if not m:
        return {"target": "", "split": "", "jobid": ""}
    return m.groupdict()


def build_err_glob(target_dir: Path, target: str, split: str) -> str:
    logs = target_dir / "logs"
    if split == "all":
        return str(logs / f"diffdock_{target}_*_*.err")
    if split in ("actives", "decoys"):
        return str(logs / f"diffdock_{target}_{split}_*.err")
    raise ValueError("--split must be one of: actives, decoys, all")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--target_dir",
        required=True,
        help="e.g. /home/.../dude_raw/wee1",
    )
    ap.add_argument(
        "--target",
        default="",
        help="default: basename(target_dir)",
    )
    ap.add_argument("--split", choices=["actives", "decoys", "all"], default="all")

    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--out_txt_dir",
        default="",
        help="if set, write fail/skip ligand_id lists here",
    )

    args = ap.parse_args()

    target_dir = Path(args.target_dir)
    target = args.target.strip() if args.target.strip() else target_dir.name
    err_glob = build_err_glob(target_dir, target, args.split)

    err_files = [Path(p) for p in glob.glob(err_glob)]
    err_files.sort()

    # ligand_id -> record(status + evidence)
    status: dict[str, dict] = {}

    for ef in err_files:
        meta = parse_err_filename(ef)

        try:
            lines = ef.read_text(errors="replace").splitlines()
        except Exception:
            continue

        for ln in lines:
            st = classify_line(ln)
            if st is None:
                continue

            lid = extract_ligand_id(ln, st, target)
            if lid is None:
                continue

            prev = status.get(lid, {}).get("status")
            new = update_status(prev, st)
            if new != prev:
                status[lid] = {
                    "ligand_id": lid,
                    "status": new,
                    "err_file": str(ef),
                    "matched_line": ln,
                    "target_from_fname": meta.get("target", ""),
                    "split_from_fname": meta.get("split", ""),
                    "jobid_from_fname": meta.get("jobid", ""),
                }

    df = pd.DataFrame(list(status.values()))
    if len(df) == 0:
        df = pd.DataFrame(
            columns=[
                "ligand_id",
                "status",
                "err_file",
                "matched_line",
                "target_from_fname",
                "split_from_fname",
                "jobid_from_fname",
            ]
        )

    # sort by priority (fail first), then ligand_id
    if len(df) > 0:
        df["priority"] = df["status"].map(PRIORITY).fillna(0).astype(int)
        df = df.sort_values(["priority", "ligand_id"], ascending=[False, True]).drop(
            columns=["priority"]
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if args.out_txt_dir:
        d = Path(args.out_txt_dir)
        d.mkdir(parents=True, exist_ok=True)
        for st in ["fail", "skip_test_missing", "skip_conf_missing"]:
            sub = df[df["status"] == st]
            (d / f"{st}.txt").write_text(
                "\n".join(sub["ligand_id"].tolist()) + "\n", encoding="utf-8"
            )

    print(f"[GLOB] {err_glob}")
    print(f"[SAVE] {out_csv} rows={len(df)}")
    if args.out_txt_dir:
        print(f"[SAVE] txt lists -> {args.out_txt_dir}")


if __name__ == "__main__":
    main()
    
'''
전체(actives + decoys) 파싱
TARGET="braf"
cd ./dataset/DUD-E
python ./dataset/DUD-E/scripts_2/postprocess/parse_inference_err.py \
  --target_dir ./dataset/DUD-E/dude_raw/$TARGET \
  --split all \
  --out_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/inference_status_err.csv \
  --out_txt_dir ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/inference_status_lists

only actives만
TARGET="braf"
cd ./dataset/DUD-E
TARGET="braf"
python ./dataset/DUD-E/scripts_2/postprocess/parse_inference_err.py \
  --target_dir ./dataset/DUD-E/dude_raw/$TARGET \
  --split actives \
  --out_csv ./dataset/DUD-E/dude_raw/$TARGET/eval/diffdock_2/inference_status_err_actives.csv

'''
