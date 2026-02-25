# src/io/pdb.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import os
import shutil
from datetime import datetime


# ============================================================
# Residue Mapping Policy
# ============================================================

DEFAULT_RESNAME_MAP: Dict[str, str] = {
    # Histidine variants
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",

    # Phosphorylated residues
    "TPO": "THR",
    "SEP": "SER",
    "PTR": "TYR",

    # Selenomethionine
    "MSE": "MET",
}


# ============================================================
# Low-level utilities
# ============================================================

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_pdb_atom_line(line: str) -> bool:
    if len(line) < 20:
        return False
    rec = line[0:6].strip()
    return rec in ("ATOM", "HETATM")


def get_resname(line: str) -> str:
    return line[17:20]


def set_resname(line: str, new_resname: str) -> str:
    if len(new_resname) != 3:
        raise ValueError("Residue name must be exactly 3 characters.")
    return line[:17] + new_resname + line[20:]


# ============================================================
# Core standardization logic
# ============================================================

def standardize_resnames(
    input_pdb: Path,
    output_pdb: Path,
    resname_map: Dict[str, str] | None = None,
    apply_to_hetatm: bool = True,
) -> Dict:
    """
    Standardize residue names in a PDB file.

    Parameters
    ----------
    input_pdb : Path
    output_pdb : Path
    resname_map : mapping dictionary
    apply_to_hetatm : whether to apply mapping to HETATM records

    Returns
    -------
    dict with statistics
    """
    if resname_map is None:
        resname_map = DEFAULT_RESNAME_MAP

    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    total_checked = 0
    changes: Dict[Tuple[str, str], int] = {}

    with input_pdb.open("r") as fin, output_pdb.open("w") as fout:
        for line in fin:
            if is_pdb_atom_line(line):
                rec = line[0:6].strip()
                if rec == "HETATM" and not apply_to_hetatm:
                    fout.write(line)
                    continue

                total_checked += 1
                old = get_resname(line)

                if old in resname_map:
                    new = resname_map[old]
                    if new != old:
                        line = set_resname(line, new)
                        changes[(old, new)] = changes.get((old, new), 0) + 1

            fout.write(line)

    return {
        "total_atomhetatm_lines_checked": total_checked,
        "changes": {f"{o}->{n}": c for (o, n), c in changes.items()},
    }


# ============================================================
# Safe in-place editing
# ============================================================

def standardize_inplace(
    pdb_path: Path,
    resname_map: Dict[str, str] | None = None,
    apply_to_hetatm: bool = True,
    keep_origin_backup: bool = True,
) -> Dict:
    """
    Safely standardize a PDB file in-place.
    """

    pdb_path = Path(pdb_path)
    tmp_path = pdb_path.with_suffix(pdb_path.suffix + ".tmp")

    stats = standardize_resnames(
        input_pdb=pdb_path,
        output_pdb=tmp_path,
        resname_map=resname_map,
        apply_to_hetatm=apply_to_hetatm,
    )

    if keep_origin_backup:
        origin_backup = pdb_path.with_name("receptor_origin.pdb")
        if not origin_backup.exists():
            shutil.copy2(pdb_path, origin_backup)

    os.replace(tmp_path, pdb_path)

    return stats
