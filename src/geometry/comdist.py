# src/geometry/comdist.py
from __future__ import annotations

import os, glob, re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from rdkit import Chem


# ---------- I/O helpers (기존 스크립트 재사용) ----------

def load_first_sdf(sdf_path: str) -> Optional[Chem.Mol]:
    supp = Chem.SDMolSupplier(sdf_path, removeHs=False)
    if not supp or supp[0] is None:
        return None
    mol = supp[0]
    if mol.GetNumConformers() == 0:
        return None
    return mol

def load_mol2(mol2_path: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromMol2File(mol2_path, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return mol


# ---------- geometry helpers ----------

def centroid_heavy(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    heavy-atom centroid:
      c = (1/|H|) * sum_{i: Z_i>1} x_i
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(a.GetIdx())
        coords.append([p.x, p.y, p.z])
    if len(coords) == 0:
        return None
    coords = np.asarray(coords, dtype=float)
    return coords.mean(axis=0)

def find_rank1_sdf(lig_dir: str) -> Optional[str]:
    """
    우선순위(평가/랭킹 일관성):
      1) rank1.sdf
      2) rank1_confidence*.sdf
    """
    cand2 = sorted(glob.glob(os.path.join(lig_dir, "rank1.sdf")))
    if cand2:
        return cand2[0]
    cand = sorted(glob.glob(os.path.join(lig_dir, "rank1_confidence*.sdf")))
    if cand:
        return cand[0]
    return None

def parse_confidence_from_filename(path: str) -> float:
    """
    filename에서 confidence 파싱(기존 스크립트 재사용).
    예:
      rank1_confidence-2.41.sdf
      rank1_confidence0.50.sdf
    """
    fname = os.path.basename(path)
    m = re.search(r"confidence(-?\d+(?:\.\d+)?)", fname)
    if m is None:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


# ---------- main computation ----------

def compute_comdist_table(
    *,
    dude_root: str | Path,
    target: str,
    split: str,                 # "actives" | "decoys"
    cutoff_A: float = 2.0,
    results_dir: str | Path | None = None,     # default: <dude_root>/<target>/results/<split>
    crystal_ligand_path: str | Path | None = None,  # default: <dude_root>/<target>/crystal_ligand.mol2
) -> pd.DataFrame:
    """
    출력 DF 컬럼(기존과 동일):
      target, split, ligand_id, success, com_dist_A, pass_2A, confidence, sdf_path
    """
    dude_root = Path(dude_root)
    target_dir = dude_root / target

    if crystal_ligand_path is None:
        crystal_ligand_path = target_dir / "crystal_ligand.mol2"
    else:
        crystal_ligand_path = Path(crystal_ligand_path)

    xtal = load_mol2(str(crystal_ligand_path))
    if xtal is None:
        raise RuntimeError(f"crystal ligand load failed: {crystal_ligand_path}")

    c_xtal = centroid_heavy(xtal)
    if c_xtal is None:
        raise RuntimeError(f"crystal ligand centroid failed: {crystal_ligand_path}")

    if results_dir is None:
        results_dir = target_dir / "results" / split
    else:
        results_dir = Path(results_dir)

    lig_dirs = sorted(
        d for d in glob.glob(str(Path(results_dir) / "*"))
        if os.path.isdir(d)
    )

    rows: List[Dict[str, Any]] = []
    for d in lig_dirs:
        lig_id = os.path.basename(d)
        sdf = find_rank1_sdf(d)

        if sdf is None:
            rows.append(dict(
                target=target,
                split=split,
                ligand_id=lig_id,
                success=0,
                com_dist_A=float("nan"),
                pass_2A=0,
                confidence=float("nan"),
                sdf_path=""
            ))
            continue

        mol = load_first_sdf(sdf)
        if mol is None:
            rows.append(dict(
                target=target,
                split=split,
                ligand_id=lig_id,
                success=0,
                com_dist_A=float("nan"),
                pass_2A=0,
                confidence=parse_confidence_from_filename(sdf),
                sdf_path=sdf
            ))
            continue

        c_pred = centroid_heavy(mol)
        if c_pred is None:
            rows.append(dict(
                target=target,
                split=split,
                ligand_id=lig_id,
                success=0,
                com_dist_A=float("nan"),
                pass_2A=0,
                confidence=parse_confidence_from_filename(sdf),
                sdf_path=sdf
            ))
            continue

        dist = float(np.linalg.norm(c_pred - c_xtal))

        rows.append(dict(
            target=target,
            split=split,
            ligand_id=lig_id,
            success=1,
            com_dist_A=dist,
            pass_2A=int(dist <= float(cutoff_A)),
            confidence=parse_confidence_from_filename(sdf),
            sdf_path=sdf
        ))

    return pd.DataFrame(rows)


def summarize_comdist(df: pd.DataFrame, *, cutoff_A: float) -> Dict[str, Any]:
    ok = df[df["success"] == 1]
    out = {
        "N": int(len(df)),
        "success": int(len(ok)),
        "cutoff_A": float(cutoff_A),
    }
    if len(ok) > 0:
        out.update({
            "pass_rate": float(ok["pass_2A"].mean()),
            "median_dist_A": float(ok["com_dist_A"].median()),
            "mean_dist_A": float(ok["com_dist_A"].mean()),
        })
    else:
        out.update({
            "pass_rate": float("nan"),
            "median_dist_A": float("nan"),
            "mean_dist_A": float("nan"),
        })
    return out
