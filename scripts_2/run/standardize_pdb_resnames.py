#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# 프로젝트 루트 기준 import 가능하도록 조정 필요할 수 있음
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.io.pdb import standardize_inplace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize residue names in a receptor PDB file."
    )
    parser.add_argument(
        "--pdb",
        required=True,
        help="Path to receptor.pdb"
    )
    parser.add_argument(
        "--no-hetatm",
        action="store_true",
        help="Do not apply mapping to HETATM records."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    stats = standardize_inplace(
        pdb_path=pdb_path,
        apply_to_hetatm=not args.no_hetatm,
    )

    print("=== Standardization Summary ===")
    print(f"Checked lines: {stats['total_atomhetatm_lines_checked']}")
    if stats["changes"]:
        for k, v in stats["changes"].items():
            print(f"{k}: {v}")
    else:
        print("No residue name changes applied.")


if __name__ == "__main__":
    main()

'''
실행 예시
cd /home/deepfold/users/hosung/dataset/DUD-E
TARGET="csf1r"
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/run/standardize_pdb_resnames.py \
    --pdb /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/$TARGET/receptor.pdb
    
HETATM 제외하고 실행
python /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/run/standardize_pdb_resnames.py \
    --pdb /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/$TARGET/receptor.pdb \
    --no-hetatm

예상 출력
=== Standardization Summary ===
Checked lines: 4287
HID->HIS: 3
MSE->MET: 2

실행 후 파일 변화
receptor.pdb             ← 표준화된 파일
receptor_origin.pdb      ← 최초 1회 백업
'''