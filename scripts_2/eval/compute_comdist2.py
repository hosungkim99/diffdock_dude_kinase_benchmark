#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

# repo 루트에서 실행한다고 가정 (scripts_2/eval/ 기준 2단계 위)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.geometry.comdist import compute_comdist_table, summarize_comdist


def _resolve_results_dir(
    dude_root: Path,
    target: str,
    split: str,
    results_dir_arg: str,
) -> Path | None:
    """
    results_dir 오버라이드는 기존처럼 유지한다.
    - split=actives/decoys일 때: 사용자가 .../results/actives 같은 split까지 포함된 경로를 줄 수 있다.
    - split=all/both일 때:
        * results_dir_arg가 비어있으면 None(=기본 경로 사용)
        * results_dir_arg가 주어지면 'results 루트'로 간주하고 그 아래 actives/decoys를 붙인다.
          예: --results_dir /.../results  -> /.../results/actives, /.../results/decoys
        * 사용자가 이미 /.../results/actives 를 줬다면 all에서는 모호하므로 에러로 막는다.
    """
    if not results_dir_arg.strip():
        return None

    p = Path(results_dir_arg)

    # all/both에서는 split까지 포함된 경로를 받으면 실수 가능성이 커서 방지
    if split in {"all", "both"}:
        tail = p.name.lower()
        if tail in {"actives", "decoys"}:
            raise ValueError(
                f"--split {split}에서는 --results_dir에 split 하위(actives/decoys)까지 포함하지 말아야 한다: {p}"
            )
        return p

    # actives/decoys 단일 split은 사용자가 split까지 포함해도 허용(기존 동작 유지)
    return p


def _resolve_crystal_ligand(
    dude_root: Path,
    target: str,
    crystal_ligand_arg: str,
) -> Path | None:
    if not crystal_ligand_arg.strip():
        return None
    return Path(crystal_ligand_arg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dude_root",
        required=True,
        help="e.g. /home/.../dataset/DUD-E/dude_raw",
    )
    ap.add_argument("--target", required=True, help="target name, e.g. braf")

    # 요구사항: split=all 추가 + 기존 both도 호환 유지
    ap.add_argument(
        "--split",
        choices=["actives", "decoys", "both", "all"],
        required=True,
        help="actives/decoys: single split; both/all: compute both and write combined comdist_all.csv",
    )

    ap.add_argument("--cutoff_A", type=float, default=2.0)
    ap.add_argument("--out_csv", default=None)

    # 필요하면 results_dir을 split까지 직접 줄 수 있게 유지
    ap.add_argument(
        "--results_dir",
        default="",
        help="optional: override results dir. "
             "For split=actives/decoys: can be .../results/actives. "
             "For split=all/both: must be results root (e.g. .../results), NOT .../results/actives",
    )
    ap.add_argument(
        "--crystal_ligand",
        default="",
        help="optional: override crystal ligand mol2 path",
    )

    args = ap.parse_args()

    dude_root = Path(args.dude_root)
    target = args.target
    split = args.split.strip().lower()

    eval_com_dir = dude_root / target / "eval" / "diffdock_2" / "COM"
    eval_com_dir.mkdir(parents=True, exist_ok=True)

    results_dir_override = _resolve_results_dir(
        dude_root=dude_root,
        target=target,
        split=split,
        results_dir_arg=args.results_dir,
    )
    crystal_ligand = _resolve_crystal_ligand(
        dude_root=dude_root,
        target=target,
        crystal_ligand_arg=args.crystal_ligand,
    )

    cutoff_A = float(args.cutoff_A)

    # -----------------------
    # Case 1) single split
    # -----------------------
    if split in {"actives", "decoys"}:
        df = compute_comdist_table(
            dude_root=dude_root,
            target=target,
            split=split,
            cutoff_A=cutoff_A,
            results_dir=results_dir_override,
            crystal_ligand_path=crystal_ligand,
        )

        if args.out_csv is None:
            out_csv = eval_com_dir / f"{target}_comdist_{split}.csv"
        else:
            out_csv = Path(args.out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv} exists={out_csv.exists()} size={out_csv.stat().st_size if out_csv.exists() else 0}")

        summ = summarize_comdist(df, cutoff_A=cutoff_A)
        print(
            f"[{target} {split}] "
            f"N={summ['N']} success={summ['success']} "
            f"pass@{summ['cutoff_A']}A={summ['pass_rate']:.3f} "
            f"median_dist={summ['median_dist_A']:.3f}A"
        )
        return

    # -----------------------
    # Case 2) split=both/all
    # -----------------------
    # out_csv는 단일 파일 경로로 쓰기 어렵다. all/both에서는 무시하고 표준 출력 경로에 저장한다.
    if args.out_csv is not None and str(args.out_csv).strip():
        print(f"[WARN] --split {split}에서는 --out_csv를 무시하고 표준 경로(COM/)에 저장한다: --out_csv={args.out_csv}")

    # results_dir_override가 None이면 compute_comdist_table이 기본 경로를 사용한다.
    # override가 있을 때는 results root를 주는 것이므로, 아래에서 split을 붙여준다.
    if results_dir_override is None:
        results_dir_act = None
        results_dir_dec = None
    else:
        results_dir_act = results_dir_override / "actives"
        results_dir_dec = results_dir_override / "decoys"

    df_act = compute_comdist_table(
        dude_root=dude_root,
        target=target,
        split="actives",
        cutoff_A=cutoff_A,
        results_dir=results_dir_act,
        crystal_ligand_path=crystal_ligand,
    )
    df_dec = compute_comdist_table(
        dude_root=dude_root,
        target=target,
        split="decoys",
        cutoff_A=cutoff_A,
        results_dir=results_dir_dec,
        crystal_ligand_path=crystal_ligand,
    )

    out_act = eval_com_dir / f"{target}_comdist_actives.csv"
    out_dec = eval_com_dir / f"{target}_comdist_decoys.csv"
    df_act.to_csv(out_act, index=False)
    df_dec.to_csv(out_dec, index=False)

    print(f"[SAVE] {out_act} exists={out_act.exists()} size={out_act.stat().st_size if out_act.exists() else 0}")
    print(f"[SAVE] {out_dec} exists={out_dec.exists()} size={out_dec.stat().st_size if out_dec.exists() else 0}")

    summ_a = summarize_comdist(df_act, cutoff_A=cutoff_A)
    summ_d = summarize_comdist(df_dec, cutoff_A=cutoff_A)
    print(
        f"[{target} actives] "
        f"N={summ_a['N']} success={summ_a['success']} "
        f"pass@{summ_a['cutoff_A']}A={summ_a['pass_rate']:.3f} "
        f"median_dist={summ_a['median_dist_A']:.3f}A"
    )
    print(
        f"[{target} decoys ] "
        f"N={summ_d['N']} success={summ_d['success']} "
        f"pass@{summ_d['cutoff_A']}A={summ_d['pass_rate']:.3f} "
        f"median_dist={summ_d['median_dist_A']:.3f}A"
    )

    # (target 내부) actives+decoys 결합
    df_all = pd.concat([df_act, df_dec], ignore_index=True)

    out_all1 = eval_com_dir / f"{target}_comdist_all.csv"
    out_all2 = eval_com_dir / "comdist_all.csv"  # 요구사항: comdist_all.csv도 생성
    df_all.to_csv(out_all1, index=False)
    df_all.to_csv(out_all2, index=False)

    print(f"[SAVE] {out_all1} exists={out_all1.exists()} size={out_all1.stat().st_size if out_all1.exists() else 0}")
    print(f"[SAVE] {out_all2} exists={out_all2.exists()} size={out_all2.stat().st_size if out_all2.exists() else 0}")

    # 전체 요약(참고)
    summ_all = summarize_comdist(df_all, cutoff_A=cutoff_A)
    print(
        f"[{target} all] "
        f"N={summ_all['N']} success={summ_all['success']} "
        f"pass@{summ_all['cutoff_A']}A={summ_all['pass_rate']:.3f} "
        f"median_dist={summ_all['median_dist_A']:.3f}A"
    )


if __name__ == "__main__":
    main()

'''
DUDE_ROOT="./dataset/DUD-E/dude_raw"
TARGET="braf"

cd ./dataset/DUD-E

python ./dataset/DUD-E/scripts_2/eval/compute_comdist2.py \
  --dude_root $DUDE_ROOT \
  --target $TARGET \
  --split all \
  --cutoff_A 2.0
'''
