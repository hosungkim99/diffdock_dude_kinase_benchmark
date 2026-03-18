# src/aggregate/master.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from rdkit import Chem


# =========================
# Config / thresholds
# =========================
@dataclass(frozen=True)
class MasterTableConfig:
    # pocket definition
    pocket_radius_A: float = 6.0

    # pocket features
    frac_atoms_within_A: float = 4.0        # for frac_atoms_within_4A
    contact_cutoff_A: float = 4.0           # for n_contacts (pair count)
    in_dcenter_A: float = 8.0               # pocket_in criterion
    in_dmin_A: float = 4.0                  # pocket_in criterion
    in_contacts_ge: int = 10                # pocket_in criterion

    # clash definition (ligand-atom count)
    clash_cutoff_A: float = 2.0

    # status precedence (err parsing)
    # (낮을수록 “더 강한 상태”로 취급하는 방식으로 쓰지 말고, 명시적으로 우선순위 dict로 처리)
    status_priority: Tuple[str, ...] = ("fail", "skip_conf_missing", "skip_test_missing")


# =========================
# RDKit / geometry helpers
# =========================

def _protein_coords_from_pdb(path: Path) -> Optional[np.ndarray]:
    coords = []
    try:
        with open(path, "r") as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                atom_name = parts[2].upper()

                # hydrogen skip
                if atom_name.startswith("H"):
                    continue

                try:
                    x = float(parts[-3])
                    y = float(parts[-2])
                    z = float(parts[-1])
                except Exception:
                    continue

                coords.append((x, y, z))

        if not coords:
            return None

        return np.asarray(coords, dtype=float)

    except Exception:
        return None

def _mol_from_sdf(path: Path) -> Optional[Chem.Mol]:
    try:
        supp = Chem.SDMolSupplier(str(path), removeHs=False)
        for m in supp:
            if m is None:
                continue
            if m.GetNumConformers() == 0:
                continue
            return m
        return None
    except Exception:
        return None


def _mol_from_mol2(path: Path) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromMol2File(str(path), removeHs=False)
        if m is None or m.GetNumConformers() == 0:
            return None
        return m
    except Exception:
        return None


def _mol_from_pdb(path: Path) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        if m is None or m.GetNumConformers() == 0:
            return None
        return m
    except Exception:
        return None


def _get_heavy_coords(mol: Chem.Mol) -> Optional[np.ndarray]:
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    pts = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(a.GetIdx())
        pts.append((p.x, p.y, p.z))
    if not pts:
        return None
    return np.asarray(pts, dtype=float)


def _centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # (Na,3) and (Nb,3) -> (Na,Nb)
    d = A[:, None, :] - B[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


# =========================
# Resolve directories / poses
# =========================
def _resolve_ligand_dir(target_dir: Path, ligand_id: str) -> Optional[Path]:
    """
    지원하는 results 구조:
      1) <target>/results/<ligand_id>
      2) <target>/results/actives/<ligand_id>
      3) <target>/results/decoys/<ligand_id>
    """
    p0 = target_dir / "results" / ligand_id
    if p0.exists():
        return p0

    p1 = target_dir / "results" / "actives" / ligand_id
    if p1.exists():
        return p1

    p2 = target_dir / "results" / "decoys" / ligand_id
    if p2.exists():
        return p2

    return None


def _resolve_rank1_pose(ligand_dir: Path) -> Optional[Path]:
    """
    우선순위:
      1) rank1.sdf
      2) rank1_confidence*.sdf
      3) rank1*.sdf
    """
    p1 = ligand_dir / "rank1.sdf"
    if p1.exists():
        return p1

    cand = sorted(ligand_dir.glob("rank1_confidence*.sdf"))
    if cand:
        return cand[0]

    cand2 = sorted(ligand_dir.glob("rank1*.sdf"))
    if cand2:
        return cand2[0]

    return None


# =========================
# Pocket definition (once per target)
# =========================
def _define_pocket(
    *,
    receptor_pdb: Path,
    crystal_ligand_mol2: Path,
    pocket_radius_A: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      prot_coords    : (Np,3) protein heavy atom coords
      pocket_coords  : (Nk,3) protein heavy atom coords within pocket_radius of crystal ligand heavy atoms
      pocket_center  : (3,)  centroid of crystal ligand heavy atoms
    """
    # prot_mol = _mol_from_pdb(receptor_pdb)
    # if prot_mol is None:
    #     raise RuntimeError(f"Failed to read receptor PDB: {receptor_pdb}")
    # prot_coords = _get_heavy_coords(prot_mol)
    # if prot_coords is None:
    #     raise RuntimeError(f"No heavy coords in receptor PDB: {receptor_pdb}")

    prot_coords = _protein_coords_from_pdb(receptor_pdb)
    if prot_coords is None:
        raise RuntimeError(f"Failed to read receptor PDB: {receptor_pdb}")
    
    ref = _mol_from_mol2(crystal_ligand_mol2)
    if ref is None:
        raise RuntimeError(f"Failed to read crystal ligand MOL2: {crystal_ligand_mol2}")
    ref_heavy = _get_heavy_coords(ref)
    if ref_heavy is None:
        raise RuntimeError(f"No heavy coords in crystal ligand MOL2: {crystal_ligand_mol2}")

    pocket_center = _centroid(ref_heavy)
    d = _pairwise_dist(prot_coords, ref_heavy)  # (Np, Nref)
    in_pocket = (d.min(axis=1) <= float(pocket_radius_A))
    pocket_coords = prot_coords[in_pocket]
    return prot_coords, pocket_coords, pocket_center


# =========================
# Per-ligand QC features
# =========================
def _compute_qc_features_for_pose(
    *,
    lig_heavy: np.ndarray,
    prot_coords: np.ndarray,
    pocket_coords: np.ndarray,
    pocket_center: np.ndarray,
    cfg: MasterTableConfig,
) -> Dict[str, Any]:
    """
    Outputs needed for master_table:
      pocket_in, frac_atoms_within_4A, min_dist_to_pocket_A, clash_count
    """
    lig_center = _centroid(lig_heavy)
    d_center = float(np.linalg.norm(lig_center - pocket_center))

    if pocket_coords.shape[0] > 0:
        d_lp = _pairwise_dist(lig_heavy, pocket_coords)      # (N_lig, N_pocket)
        min_per_lig_atom_pocket = d_lp.min(axis=1)           # (N_lig,)
        min_dist_to_pocket_A = float(min_per_lig_atom_pocket.min())
        frac_atoms_within_4A = float((min_per_lig_atom_pocket <= float(cfg.frac_atoms_within_A)).mean())
        n_contacts = int((d_lp <= float(cfg.contact_cutoff_A)).sum())
    else:
        min_dist_to_pocket_A = float("nan")
        frac_atoms_within_4A = float("nan")
        n_contacts = 0

    pocket_in = (
        (d_center <= float(cfg.in_dcenter_A)) and
        (min_dist_to_pocket_A <= float(cfg.in_dmin_A)) and
        (n_contacts >= int(cfg.in_contacts_ge))
    )

    # clash_count: ligand atom count whose min distance to ANY protein heavy atom <= clash_cutoff
    d_lprot = _pairwise_dist(lig_heavy, prot_coords)
    min_per_lig_atom_prot = d_lprot.min(axis=1)
    clash_count = int((min_per_lig_atom_prot <= float(cfg.clash_cutoff_A)).sum())

    return {
        "pocket_in": bool(pocket_in),
        "frac_atoms_within_4A": frac_atoms_within_4A,
        "min_dist_to_pocket_A": min_dist_to_pocket_A,
        "clash_count": clash_count,
    }


# =========================
# Reading inputs
# =========================
def _read_scores(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)

    # tolerate either "score" or "confidence"
    if "confidence" not in df.columns and "score" in df.columns:
        df = df.rename(columns={"score": "confidence"})

    needed = {"ligand_id", "label", "confidence"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(
            f"scores_csv must contain columns {sorted(needed)} (or score instead of confidence). "
            f"Got={list(df.columns)}"
        )

    out = df[["ligand_id", "label", "confidence"]].copy()
    out["ligand_id"] = out["ligand_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    return out


def _read_comdist(comdist_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if comdist_csv is None or (not comdist_csv.exists()):
        return None

    df = pd.read_csv(comdist_csv)

    # tolerate old naming
    if "com_dist_A" in df.columns and "COMdist_A" not in df.columns:
        df = df.rename(columns={"com_dist_A": "COMdist_A"})

    if "ligand_id" not in df.columns or "COMdist_A" not in df.columns:
        return None

    out = df[["ligand_id", "COMdist_A"]].copy()
    out["ligand_id"] = out["ligand_id"].astype(str)
    out["COMdist_A"] = pd.to_numeric(out["COMdist_A"], errors="coerce")
    return out


def _read_err_status(err_status_csv: Optional[Path], cfg: MasterTableConfig) -> Dict[str, str]:
    """
    Expect from scripts_2/postprocess/parse_inference_err.py:
      ligand_id, status, ...
    """
    if err_status_csv is None or (not err_status_csv.exists()):
        return {}

    df = pd.read_csv(err_status_csv)
    if "ligand_id" not in df.columns or "status" not in df.columns:
        return {}

    df["ligand_id"] = df["ligand_id"].astype(str)
    df["status"] = df["status"].astype(str)

    # duplicates -> keep strongest status by cfg.status_priority order
    # (우선순위 튜플 앞쪽일수록 강함: fail > skip_conf > skip_test)
    pri = {s: i for i, s in enumerate(cfg.status_priority)}
    df["_pri"] = df["status"].map(lambda x: pri.get(x, 999))
    df = df.sort_values(["ligand_id", "_pri"]).drop_duplicates("ligand_id", keep="first")
    return dict(zip(df["ligand_id"], df["status"]))


# =========================
# Public API
# =========================
def build_master_table(
    *,
    dude_root: Path,
    target: str,
    scores_csv: Path,
    out_csv: Path,
    cfg: MasterTableConfig = MasterTableConfig(),
    receptor_pdb: Optional[Path] = None,
    crystal_ligand_mol2: Optional[Path] = None,
    comdist_csv: Optional[Path] = None,
    err_status_csv: Optional[Path] = None,
    cache_qc_csv: Optional[Path] = None,
    write_config_json: bool = True,
) -> pd.DataFrame:
    """
    Produce master_table with columns (schema 고정):
      target, ligand_id, label, success, status, confidence,
      pocket_in, frac_atoms_within_4A, min_dist_to_pocket_A, COMdist_A, clash_count
    """
    dude_root = Path(dude_root)
    target_dir = dude_root / target

    receptor_pdb = receptor_pdb or (target_dir / "receptor.pdb")
    crystal_ligand_mol2 = crystal_ligand_mol2 or (target_dir / "crystal_ligand.mol2")

    # 1) base: scores
    df = _read_scores(Path(scores_csv))
    df.insert(0, "target", target)

    # 2) optional joins: COMdist
    com = _read_comdist(Path(comdist_csv) if comdist_csv else None)
    if com is not None:
        df = df.merge(com, on="ligand_id", how="left")
    else:
        df["COMdist_A"] = np.nan

    # 3) err status map
    err_map = _read_err_status(Path(err_status_csv) if err_status_csv else None, cfg)

    # 4) QC feature cache
    qc_use: Optional[pd.DataFrame] = None
    if cache_qc_csv is not None:
        cache_qc_csv = Path(cache_qc_csv)
        if cache_qc_csv.exists():
            qc = pd.read_csv(cache_qc_csv)
            req = {"ligand_id", "pose_exists", "pocket_in", "frac_atoms_within_4A", "min_dist_to_pocket_A", "clash_count"}
            if req.issubset(set(qc.columns)):
                qc_use = qc[list(req)].copy()
                qc_use["ligand_id"] = qc_use["ligand_id"].astype(str)

    if qc_use is None:
        prot_coords, pocket_coords, pocket_center = _define_pocket(
            receptor_pdb=Path(receptor_pdb),
            crystal_ligand_mol2=Path(crystal_ligand_mol2),
            pocket_radius_A=float(cfg.pocket_radius_A),
        )

        qc_rows: List[Dict[str, Any]] = []
        for ligand_id in df["ligand_id"].tolist():
            lig_dir = _resolve_ligand_dir(target_dir, ligand_id)
            if lig_dir is None:
                qc_rows.append({
                    "ligand_id": ligand_id,
                    "pose_exists": False,
                    "pocket_in": False,
                    "frac_atoms_within_4A": np.nan,
                    "min_dist_to_pocket_A": np.nan,
                    "clash_count": 0,
                })
                continue

            pose = _resolve_rank1_pose(lig_dir)
            if pose is None or (not pose.exists()):
                qc_rows.append({
                    "ligand_id": ligand_id,
                    "pose_exists": False,
                    "pocket_in": False,
                    "frac_atoms_within_4A": np.nan,
                    "min_dist_to_pocket_A": np.nan,
                    "clash_count": 0,
                })
                continue

            mol = _mol_from_sdf(pose)
            if mol is None:
                qc_rows.append({
                    "ligand_id": ligand_id,
                    "pose_exists": False,
                    "pocket_in": False,
                    "frac_atoms_within_4A": np.nan,
                    "min_dist_to_pocket_A": np.nan,
                    "clash_count": 0,
                })
                continue

            lig_heavy = _get_heavy_coords(mol)
            if lig_heavy is None:
                qc_rows.append({
                    "ligand_id": ligand_id,
                    "pose_exists": False,
                    "pocket_in": False,
                    "frac_atoms_within_4A": np.nan,
                    "min_dist_to_pocket_A": np.nan,
                    "clash_count": 0,
                })
                continue

            feats = _compute_qc_features_for_pose(
                lig_heavy=lig_heavy,
                prot_coords=prot_coords,
                pocket_coords=pocket_coords,
                pocket_center=pocket_center,
                cfg=cfg,
            )
            qc_rows.append({
                "ligand_id": ligand_id,
                "pose_exists": True,
                **feats,
            })

        qc_use = pd.DataFrame(qc_rows)

        if cache_qc_csv is not None:
            cache_qc_csv.parent.mkdir(parents=True, exist_ok=True)
            qc_use.to_csv(cache_qc_csv, index=False)

    df = df.merge(qc_use, on="ligand_id", how="left")

    # 5) status / success 결정 규칙(명시적)
    #    - err_status_csv에 있으면: 무조건 해당 status + success=0
    #    - 아니면 pose_exists=False: status=no_rank1 + success=0
    #    - 아니면: status=ok + success=1
    status_list: List[str] = []
    success_list: List[int] = []
    for ligand_id, pose_exists in zip(df["ligand_id"].tolist(), df["pose_exists"].fillna(False).astype(bool).tolist()):
        if ligand_id in err_map:
            status_list.append(err_map[ligand_id])
            success_list.append(0)
        else:
            if not pose_exists:
                status_list.append("no_rank1")
                success_list.append(0)
            else:
                status_list.append("ok")
                success_list.append(1)

    df["status"] = status_list
    df["success"] = success_list

    # 6) finalize schema + dtype normalize
    out = df[[
        "target",
        "ligand_id",
        "label",
        "success",
        "status",
        "confidence",
        "pocket_in",
        "frac_atoms_within_4A",
        "min_dist_to_pocket_A",
        "COMdist_A",
        "clash_count",
    ]].copy()

    out["label"] = out["label"].astype(int)
    out["success"] = out["success"].astype(int)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["pocket_in"] = out["pocket_in"].fillna(False).astype(bool)
    out["frac_atoms_within_4A"] = pd.to_numeric(out["frac_atoms_within_4A"], errors="coerce")
    out["min_dist_to_pocket_A"] = pd.to_numeric(out["min_dist_to_pocket_A"], errors="coerce")
    out["COMdist_A"] = pd.to_numeric(out["COMdist_A"], errors="coerce")
    out["clash_count"] = pd.to_numeric(out["clash_count"], errors="coerce").fillna(0).astype(int)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    if write_config_json:
        cfg_path = out_csv.with_suffix(".config.json")
        payload = {
            "target": target,
            "dude_root": str(dude_root),
            "scores_csv": str(scores_csv),
            "comdist_csv": str(comdist_csv) if comdist_csv else "",
            "err_status_csv": str(err_status_csv) if err_status_csv else "",
            "receptor_pdb": str(receptor_pdb),
            "crystal_ligand_mol2": str(crystal_ligand_mol2),
            "cache_qc_csv": str(cache_qc_csv) if cache_qc_csv else "",
            "config": asdict(cfg),
        }
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return out