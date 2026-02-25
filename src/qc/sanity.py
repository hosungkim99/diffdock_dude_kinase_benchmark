# src/qc/sanity.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from rdkit import Chem


def _heavy_atom_coords_from_mol(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    coords = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(a.GetIdx())
        coords.append([p.x, p.y, p.z])
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def load_protein_heavy_coords_from_pdb(pdb_path: Path) -> np.ndarray:
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to read protein PDB with RDKit: {pdb_path}")
    xyz = _heavy_atom_coords_from_mol(mol)
    if xyz.shape[0] == 0:
        raise ValueError(f"No heavy atoms read from protein PDB: {pdb_path}")
    return xyz


def load_ligand_heavy_coords_from_sdf(sdf_path: Path) -> Optional[np.ndarray]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    if len(suppl) == 0:
        return None
    mol = suppl[0]
    if mol is None or mol.GetNumConformers() == 0:
        return None
    xyz = _heavy_atom_coords_from_mol(mol)
    return None if xyz.shape[0] == 0 else xyz


def find_rank1_conf_sdf(result_dir: Path) -> Optional[Path]:
    cands = sorted(result_dir.glob("rank1_confidence*.sdf"))
    return cands[0] if cands else None


def compute_metrics(
    prot_xyz: np.ndarray,
    lig_xyz: np.ndarray,
    clash_threshold: float = 1.0,
) -> Tuple[float, float, int]:
    diffs = lig_xyz[:, None, :] - prot_xyz[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    min_dist = float(np.sqrt(np.min(d2)))

    lig_com = np.mean(lig_xyz, axis=0)
    prot_com = np.mean(prot_xyz, axis=0)
    com_dist = float(np.linalg.norm(lig_com - prot_com))

    clash_pairs = int(np.sum(d2 < (clash_threshold ** 2)))
    return min_dist, com_dist, clash_pairs


def run_sanity_check(
    target: str,
    split: str,
    receptor_pdb: Path,
    results_dir: Path,
    scores_ok_csv: Path,
    out_dir: Path,
    pocket_out_min_dist: float = 6.0,
    clash_threshold: float = 1.0,
    clash_pairs_flag: int = 50,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    prot_xyz = load_protein_heavy_coords_from_pdb(receptor_pdb)

    df = pd.read_csv(scores_ok_csv)
    if "complex_name" not in df.columns or "confidence" not in df.columns:
        raise ValueError(f"scores_ok_csv must contain complex_name, confidence: {scores_ok_csv}")

    df["complex_name"] = df["complex_name"].astype(str).str.strip()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["complex_name", "confidence"])

    rows: List[Dict] = []
    n_missing_rank1 = 0
    n_bad_lig = 0

    for cname, conf in df[["complex_name", "confidence"]].itertuples(index=False, name=None):
        cdir = results_dir / cname
        if not cdir.exists():
            rows.append({
                "complex_name": cname, "confidence": float(conf), "rank1_sdf": "",
                "min_dist": np.nan, "com_dist": np.nan, "clash_pairs": np.nan,
                "status": "missing_complex_dir",
            })
            continue

        sdf = find_rank1_conf_sdf(cdir)
        if sdf is None:
            n_missing_rank1 += 1
            rows.append({
                "complex_name": cname, "confidence": float(conf), "rank1_sdf": "",
                "min_dist": np.nan, "com_dist": np.nan, "clash_pairs": np.nan,
                "status": "missing_rank1_conf_sdf",
            })
            continue

        lig_xyz = load_ligand_heavy_coords_from_sdf(sdf)
        if lig_xyz is None:
            n_bad_lig += 1
            rows.append({
                "complex_name": cname, "confidence": float(conf), "rank1_sdf": str(sdf),
                "min_dist": np.nan, "com_dist": np.nan, "clash_pairs": np.nan,
                "status": "bad_ligand_sdf",
            })
            continue

        min_dist, com_dist, clash_pairs = compute_metrics(prot_xyz, lig_xyz, clash_threshold=clash_threshold)

        pocket_out = (min_dist > pocket_out_min_dist)
        clash_flag = (clash_pairs >= clash_pairs_flag)

        status = "ok"
        if pocket_out and clash_flag:
            status = "pocket_out_and_clash"
        elif pocket_out:
            status = "pocket_out"
        elif clash_flag:
            status = "clash_suspect"

        rows.append({
            "complex_name": cname,
            "confidence": float(conf),
            "rank1_sdf": str(sdf),
            "min_dist": float(min_dist),
            "com_dist": float(com_dist),
            "clash_pairs": int(clash_pairs),
            "status": status,
        })

    out_df = pd.DataFrame(rows)

    summary_csv = out_dir / f"{target}_{split}_sanity_summary.csv"
    out_df.to_csv(summary_csv, index=False)

    # flagged lists (complex_name,confidence)
    pocket_out_df = out_df[out_df["status"].isin(["pocket_out", "pocket_out_and_clash"])][["complex_name", "confidence", "min_dist", "clash_pairs"]]
    clash_df = out_df[out_df["status"].isin(["clash_suspect", "pocket_out_and_clash"])][["complex_name", "confidence", "min_dist", "clash_pairs"]]

    (out_dir / f"{split}_pocket_out.txt").write_text(
        "\n".join([f"{r.complex_name},{r.confidence}" for r in pocket_out_df.itertuples(index=False)]) + ("\n" if len(pocket_out_df) else ""),
        encoding="utf-8",
    )
    (out_dir / f"{split}_clash_suspect.txt").write_text(
        "\n".join([f"{r.complex_name},{r.confidence}" for r in clash_df.itertuples(index=False)]) + ("\n" if len(clash_df) else ""),
        encoding="utf-8",
    )

    # 간단 콘솔 요약은 CLI에서 출력하는 게 깔끔하지만, 반환값으로 summary path만 준다.
    return summary_csv
