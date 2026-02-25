from pathlib import Path
import gzip
from rdkit import Chem
from typing import List, Tuple


CSV_HEADER = [
    "complex_name",
    "protein_path",
    "protein_sequence",
    "ligand_description",
]


def load_sdf_gz(path: Path) -> List[Chem.Mol]:
    with gzip.open(path, "rb") as f:
        data = f.read().decode("utf-8", errors="ignore")

    suppl = Chem.SDMolSupplier()
    suppl.SetData(data, removeHs=False)

    mols = [m for m in suppl if m is not None]
    return mols


def mols_to_rows(mols, prefix: str) -> List[Tuple[str, str]]:
    rows = []
    for idx, mol in enumerate(mols, start=1):
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            continue
        rows.append((f"{prefix}_{idx:06d}", smi))
    return rows
