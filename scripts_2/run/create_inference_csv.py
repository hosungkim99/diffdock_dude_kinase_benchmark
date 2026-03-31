#!/usr/bin/env python3

import argparse
from pathlib import Path
import csv
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.io.dude import load_sdf_gz, mols_to_rows, CSV_HEADER


def write_csv(rows, out_path: Path, receptor_path: Path):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for name, smi in rows:
            writer.writerow([name, str(receptor_path), "", smi])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    base = Path(args.root) / args.target
    receptor = base / "receptor.pdb"
    actives = base / "actives_final.sdf.gz"
    decoys = base / "decoys_final.sdf.gz"

    if not receptor.exists():
        raise FileNotFoundError(receptor)

    act_rows = mols_to_rows(load_sdf_gz(actives), f"{args.target}_active")
    dec_rows = mols_to_rows(load_sdf_gz(decoys), f"{args.target}_decoy")

    write_csv(act_rows, base / f"{args.target}_actives.csv", receptor)
    write_csv(dec_rows, base / f"{args.target}_decoys.csv", receptor)

    print(f"Wrote {len(act_rows)} actives")
    print(f"Wrote {len(dec_rows)} decoys")


if __name__ == "__main__":
    main()

'''
실행 예시
TARGET="akt1"
cd ./dataset/DUD-E
python ./dataset/DUD-E/scripts_2/run/create_inference_csv.py \
    --root ./dataset/DUD-E/dude_raw \
    --target $TARGET

출력
akt1_actives.csv
akt1_decoys.csv

생성되는 csv 구조
complex_name,protein_path,protein_sequence,ligand_description
akt1_active_000001,/.../receptor.pdb,,CCO...
akt1_active_000002,/.../receptor.pdb,,CNC...
...

Github README에 넣기 좋은 섹션 예시
# 1. Standardize receptor
python scripts/run/standardize_pdb_resnames.py \
    --pdb $DUDE_ROOT/akt1/receptor.pdb

# 2. Generate DiffDock input CSV
python scripts/run/create_inference_csv.py \
    --root $DUDE_ROOT \
    --target akt1

'''
