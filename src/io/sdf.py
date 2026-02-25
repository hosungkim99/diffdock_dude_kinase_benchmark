# src/io/sdf.py
from __future__ import annotations

from pathlib import Path
from typing import List, Iterable, Optional
import gzip
from rdkit import Chem


def load_sdf_gz_as_mols(path: Path, remove_hs: bool = False) -> List[Chem.Mol]:
    """
    .sdf.gz를 읽어 RDKit Mol 리스트로 반환.
    remove_hs=False를 기본으로 두면 explicit H를 보존한다.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with gzip.open(path, "rb") as f:
        data = f.read().decode("utf-8", errors="ignore")

    suppl = Chem.SDMolSupplier()
    suppl.SetData(data, removeHs=remove_hs)
    return [m for m in suppl if m is not None]
