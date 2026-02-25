# src/qc/pocket_rank1.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem


# -----------------------------
# Geometry helpers
# -----------------------------
def mol_from_sdf(path: Path) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if len(mols) == 0:
        raise RuntimeError(f"Failed to read SDF: {path}")
    m = mols[0]
    if m.GetNumConformers() == 0:
        raise RuntimeError(f"SDF has no conformer/3D coords: {path}")
    return m


def mol_from_mol2(path: Path) -> Chem.Mol:
    m = Chem.MolFromMol2File(str(path), removeHs=False)
    if m is None:
        raise RuntimeError(f"Failed to read MOL2: {path}")
    if m.GetNumConformers() == 0:
        raise RuntimeError(f"MOL2 has no conformer/3D coords: {path}")
    return m


def get_heavy_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    pts = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(a.GetIdx())
        pts.append((p.x, p.y, p.z))
    if len(pts) == 0:
        raise RuntimeError("No heavy atoms found.")
    return np.array(pts, dtype=float)


def centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = A[:, None, :] - B[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


# -----------------------------
# IO helpers
# -----------------------------
def resolve_rank1_pose(ligand_dir: Path) -> Optional[Path]:
    """
    Priority:
      1) rank1.sdf
      2) rank1_confidence*.sdf
      3) rank1*.sdf
    """
    p1 = ligand_dir / "rank1.sdf"
    if p1.exists():
        return p1

    cand = sorted(ligand_dir.glob("rank1_confidence*.sdf"))
    if len(cand) > 0:
        return cand[0]

    cand2 = sorted(ligand_dir.glob("rank1*.sdf"))
    if len(cand2) > 0:
        return cand2[0]

    return None


def label_from_ligand_id(ligand_id: str) -> Optional[int]:
    if "_active_" in ligand_id:
        return 1
    if "_decoy_" in ligand_id:
        return 0
    return None


def load_topk_ranking_mixed(
    *,
    ranking_csv: Optional[Path],
    scores_csv: Optional[Path],
    topk: int,
) -> Tuple[List[int], List[str], List[float], List[Optional[int]]]:
    """
    Mixed TopK selection (actives+decoys).
    Returns: ranks, ligand_ids, scores, labels

    ranking_csv preferred: columns ligand_id, score, (label optional), (rank optional)
    scores_csv fallback  : columns ligand_id, label, score
    """
    if ranking_csv:
        rk = pd.read_csv(ranking_csv)
        if not {"ligand_id", "score"}.issubset(set(rk.columns)):
            raise ValueError("ranking_csv must have columns: ligand_id, score (and label optional)")
        rk = rk.head(int(topk)).copy()
        ligand_ids = rk["ligand_id"].tolist()
        scores = rk["score"].astype(float).tolist()
        labels = rk["label"].tolist() if "label" in rk.columns else [label_from_ligand_id(x) for x in ligand_ids]
        ranks = rk["rank"].tolist() if "rank" in rk.columns else list(range(1, len(ligand_ids) + 1))
        return ranks, ligand_ids, scores, labels

    if scores_csv:
        sc = pd.read_csv(scores_csv)
        if not {"ligand_id", "score", "label"}.issubset(set(sc.columns)):
            raise ValueError("scores_csv must have columns: ligand_id, label, score")
        sc = sc.copy()
        sc["score"] = sc["score"].astype(float)
        sc = sc.sort_values("score", ascending=False).head(int(topk)).copy()
        ligand_ids = sc["ligand_id"].astype(str).tolist()
        scores = sc["score"].tolist()
        labels = sc["label"].tolist()
        ranks = list(range(1, len(ligand_ids) + 1))
        return ranks, ligand_ids, scores, labels

    raise ValueError("Provide either ranking_csv or scores_csv for mixed mode")


def load_topk_actives_scores_ok(
    *,
    actives_scores_ok_csv: Path,
    topk: int,
) -> Tuple[List[int], List[str], List[float], List[Optional[int]]]:
    """
    Active-only TopK selection from *_actives_scores_ok.csv.

    Expected columns:
      complex_name, confidence, label, rank_used, has_rank1

    Sort by confidence desc, take TopK.
    Returns: ranks, ligand_ids(=complex_name), scores(=confidence), labels
    """
    df = pd.read_csv(actives_scores_ok_csv)
    need = {"complex_name", "confidence", "label"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"actives_scores_ok_csv must have columns: {sorted(list(need))}")

    df = df.copy()
    df["confidence"] = df["confidence"].astype(float)
    df = df.sort_values("confidence", ascending=False).head(int(topk)).copy()

    ligand_ids = df["complex_name"].astype(str).tolist()
    scores = df["confidence"].tolist()
    labels = df["label"].tolist()
    ranks = list(range(1, len(ligand_ids) + 1))
    return ranks, ligand_ids, scores, labels


def resolve_ligand_dir(results_root: Path, ligand_id: str, label: Optional[int]) -> Path:
    """
    Robustly resolve ligand directory for mixed mode.

    Priority:
      1) results_root/ligand_id exists
      2) results_root/actives/ligand_id or results_root/decoys/ligand_id (by label or ligand_id pattern)
    """
    direct = results_root / ligand_id
    if direct.exists():
        return direct

    inferred = label if label is not None else label_from_ligand_id(ligand_id)
    if inferred == 1:
        return results_root / "actives" / ligand_id
    if inferred == 0:
        return results_root / "decoys" / ligand_id

    cand_a = results_root / "actives" / ligand_id
    if cand_a.exists():
        return cand_a
    return results_root / "decoys" / ligand_id


# -----------------------------
# Pocket definition
# -----------------------------
def load_protein_heavy_coords_from_pdb(pdb_path: Path) -> np.ndarray:
    m = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if m is None or m.GetNumConformers() == 0:
        raise RuntimeError(f"Failed to read receptor PDB or no coords: {pdb_path}")
    return get_heavy_coords(m)


def define_pocket_atoms(
    receptor_pdb: Path,
    pocket_ligand_mol2: Path,
    pocket_radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Returns:
      prot_coords   : (Np,3) protein heavy atoms
      pocket_coords : (Nk,3) pocket heavy atoms (within radius of crystal ligand heavy atoms)
      pocket_center : (3,)  centroid of crystal ligand heavy atoms
      pocket_warn   : bool  (pocket atoms too few)
    """
    prot_coords = load_protein_heavy_coords_from_pdb(receptor_pdb)

    ref_lig = mol_from_mol2(pocket_ligand_mol2)
    ref_heavy = get_heavy_coords(ref_lig)
    pocket_center = centroid(ref_heavy)

    d = pairwise_dist(prot_coords, ref_heavy)  # (Np, Nlig)
    in_pocket = (d.min(axis=1) <= float(pocket_radius))
    pocket_coords = prot_coords[in_pocket]

    pocket_warn = bool(pocket_coords.shape[0] < 30)
    return prot_coords, pocket_coords, pocket_center, pocket_warn


# -----------------------------
# QC thresholds
# -----------------------------
@dataclass(frozen=True)
class PocketQCThresholds:
    contact_cutoff: float = 4.0
    clash_cutoff: float = 2.0
    in_dcenter: float = 8.0
    in_dmin: float = 4.0
    in_contacts: int = 10
    out_dcenter: float = 15.0
    out_dmin: float = 8.0

    clash_n_clashes_ge: int = 5
    clash_min_prot_le: float = 1.6


# -----------------------------
# Main QC runner
# -----------------------------
def run_pocket_qc_rank1(
    *,
    target_dir: Path,
    outdir: Path,
    topk: int,
    mode: str = "mixed",  # "mixed" or "actives_only"
    results_root: Optional[Path] = None,
    ranking_csv: Optional[Path] = None,
    scores_csv: Optional[Path] = None,
    actives_scores_ok_csv: Optional[Path] = None,
    pocket_ligand: Optional[Path] = None,
    receptor_pdb: Optional[Path] = None,
    pocket_radius: float = 6.0,
    thr: PocketQCThresholds = PocketQCThresholds(),
) -> Dict[str, Any]:
    """
    mode:
      - "mixed": TopK from ranking_csv (preferred) or scores_csv; pose lookup under results_root with actives/decoys split.
      - "actives_only": TopK from actives_scores_ok_csv (confidence); pose lookup under results_root/actives.

    Writes:
      - pocket_qc_top{topk}.csv
      - flagged_outside.csv
      - flagged_clash.csv
      - pocket_qc_summary.json
      - pocket_qc_summary.txt
    """
    target_dir = Path(target_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_root = Path(results_root) if results_root else (target_dir / "results")
    pocket_ligand = Path(pocket_ligand) if pocket_ligand else (target_dir / "crystal_ligand.mol2")
    receptor_pdb = Path(receptor_pdb) if receptor_pdb else (target_dir / "receptor.pdb")

    if not pocket_ligand.exists():
        raise FileNotFoundError(f"Pocket ligand not found: {pocket_ligand}")
    if not receptor_pdb.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {receptor_pdb}")
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    mode = str(mode).strip().lower()
    if mode not in {"mixed", "actives_only"}:
        raise ValueError("mode must be one of: mixed, actives_only")

    # ---- select TopK list
    if mode == "mixed":
        ranks, ligand_ids, scores, labels = load_topk_ranking_mixed(
            ranking_csv=Path(ranking_csv) if ranking_csv else None,
            scores_csv=Path(scores_csv) if scores_csv else None,
            topk=int(topk),
        )
    else:
        if actives_scores_ok_csv is None:
            raise ValueError("actives_only mode requires actives_scores_ok_csv")
        ranks, ligand_ids, scores, labels = load_topk_actives_scores_ok(
            actives_scores_ok_csv=Path(actives_scores_ok_csv),
            topk=int(topk),
        )

    # ---- pocket definition
    prot_coords, pocket_coords, pocket_center, pocket_warn = define_pocket_atoms(
        receptor_pdb=receptor_pdb,
        pocket_ligand_mol2=pocket_ligand,
        pocket_radius=float(pocket_radius),
    )
    n_pocket_atoms = int(pocket_coords.shape[0])

    # ---- per-row QC
    rows: List[Dict[str, Any]] = []
    for rank, ligand_id, score, label in zip(ranks, ligand_ids, scores, labels):
        if mode == "actives_only":
            ligand_dir = results_root / "actives" / str(ligand_id)
        else:
            ligand_dir = resolve_ligand_dir(results_root, str(ligand_id), label)

        pose_path = resolve_rank1_pose(ligand_dir)

        if pose_path is None or (not pose_path.exists()):
            rows.append({
                "rank": int(rank),
                "ligand_id": str(ligand_id),
                "label": int(label) if label is not None else None,
                "score": float(score),
                "ligand_dir": str(ligand_dir),
                "pose_path": "" if pose_path is None else str(pose_path),
                "status": "missing_pose",
                "d_center": np.nan,
                "d_min_pocket": np.nan,
                "n_contacts": 0,
                "min_protein_dist": np.nan,
                "clash_count": 0,
                "pocket_in": False,
                "pocket_out_strong": False,
                "clash_flag": False,
            })
            continue

        try:
            lig = mol_from_sdf(pose_path)
            lig_heavy = get_heavy_coords(lig)
        except Exception as e:
            rows.append({
                "rank": int(rank),
                "ligand_id": str(ligand_id),
                "label": int(label) if label is not None else None,
                "score": float(score),
                "ligand_dir": str(ligand_dir),
                "pose_path": str(pose_path),
                "status": f"bad_pose: {type(e).__name__}",
                "d_center": np.nan,
                "d_min_pocket": np.nan,
                "n_contacts": 0,
                "min_protein_dist": np.nan,
                "clash_count": 0,
                "pocket_in": False,
                "pocket_out_strong": False,
                "clash_flag": False,
            })
            continue

        lig_center = centroid(lig_heavy)
        d_center = float(np.linalg.norm(lig_center - pocket_center))

        # ligand-to-pocket
        if n_pocket_atoms > 0:
            d_lp = pairwise_dist(lig_heavy, pocket_coords)
            d_min_pocket = float(d_lp.min())
            n_contacts = int((d_lp <= float(thr.contact_cutoff)).sum())
        else:
            d_min_pocket = float("nan")
            n_contacts = 0

        # ligand-to-protein (clash)
        d_lprot = pairwise_dist(lig_heavy, prot_coords)
        min_per_lig_atom = d_lprot.min(axis=1)
        min_prot = float(min_per_lig_atom.min())
        clash_count = int((min_per_lig_atom <= float(thr.clash_cutoff)).sum())

        pocket_in = bool(
            (d_center <= float(thr.in_dcenter)) and
            (d_min_pocket <= float(thr.in_dmin)) and
            (n_contacts >= int(thr.in_contacts))
        )
        pocket_out_strong = bool(
            (d_center >= float(thr.out_dcenter)) or
            (d_min_pocket >= float(thr.out_dmin))
        )
        clash_flag = bool(
            (clash_count >= int(thr.clash_n_clashes_ge)) or
            (min_prot <= float(thr.clash_min_prot_le))
        )

        rows.append({
            "rank": int(rank),
            "ligand_id": str(ligand_id),
            "label": int(label) if label is not None else None,
            "score": float(score),
            "ligand_dir": str(ligand_dir),
            "pose_path": str(pose_path),
            "status": "ok",
            "d_center": float(d_center),
            "d_min_pocket": float(d_min_pocket),
            "n_contacts": int(n_contacts),
            "min_protein_dist": float(min_prot),
            "clash_count": int(clash_count),
            "pocket_in": bool(pocket_in),
            "pocket_out_strong": bool(pocket_out_strong),
            "clash_flag": bool(clash_flag),
        })

    df = pd.DataFrame(rows)

    # outputs
    out_csv = outdir / f"pocket_qc_top{int(topk)}.csv"
    df.to_csv(out_csv, index=False)

    df_ok = (df["status"] == "ok")
    flagged_outside = df[df_ok & (df["pocket_out_strong"] == True)].copy()
    flagged_clash = df[df_ok & (df["clash_flag"] == True)].copy()
    flagged_outside.to_csv(outdir / "flagged_outside.csv", index=False)
    flagged_clash.to_csv(outdir / "flagged_clash.csv", index=False)

    # summary
    n_total = int(len(df))
    n_ok = int((df["status"] == "ok").sum())
    n_missing = int((df["status"] == "missing_pose").sum())
    n_bad = int(df["status"].astype(str).str.startswith("bad_pose").sum())

    ok = (df["status"] == "ok")
    denom_ok = max(1, int(ok.sum()))

    summary: Dict[str, Any] = {
        "inputs": {
            "target_dir": str(target_dir),
            "results_root": str(results_root),
            "receptor_pdb": str(receptor_pdb),
            "pocket_ligand": str(pocket_ligand),
            "topk": int(topk),
            "mode": mode,
            "ranking_csv": str(ranking_csv) if ranking_csv else "",
            "scores_csv": str(scores_csv) if scores_csv else "",
            "actives_scores_ok_csv": str(actives_scores_ok_csv) if actives_scores_ok_csv else "",
        },
        "pocket_definition": {
            "pocket_radius": float(pocket_radius),
            "pocket_atoms": int(n_pocket_atoms),
            "pocket_center": [float(x) for x in pocket_center.tolist()],
            "pocket_warn_small_atom_count": bool(pocket_warn),
        },
        "thresholds": {
            "contact_cutoff": float(thr.contact_cutoff),
            "clash_cutoff": float(thr.clash_cutoff),
            "in_dcenter": float(thr.in_dcenter),
            "in_dmin": float(thr.in_dmin),
            "in_contacts": int(thr.in_contacts),
            "out_dcenter": float(thr.out_dcenter),
            "out_dmin": float(thr.out_dmin),
            "clash_n_clashes_ge": int(thr.clash_n_clashes_ge),
            "clash_min_prot_le": float(thr.clash_min_prot_le),
        },
        "counts": {
            "n_total": int(n_total),
            "n_ok": int(n_ok),
            "n_missing_pose": int(n_missing),
            "n_bad_pose": int(n_bad),
            "n_pocket_in": int(df[ok]["pocket_in"].sum()),
            "n_pocket_out_strong": int(df[ok]["pocket_out_strong"].sum()),
            "n_clash_flag": int(df[ok]["clash_flag"].sum()),
        },
        "rates": {
            "pocket_in_rate": float(df[ok]["pocket_in"].sum() / denom_ok),
            "pocket_out_strong_rate": float(df[ok]["pocket_out_strong"].sum() / denom_ok),
            "clash_flag_rate": float(df[ok]["clash_flag"].sum() / denom_ok),
        },
    }

    # -----------------------------
    # per-label medians on ok only
    # -----------------------------
    # 요구사항:
    # - actives_only 모드에서는 by_label_active만 만들고 by_label_decoy는 만들지 않는다.
    # - mixed 모드에서도 ok 표본이 0개인 label 블록은 만들지 않는다(= NaN 방지).
    if "label" in df.columns:
        if mode == "actives_only":
            label_specs = [(1, "active")]
        else:
            label_specs = [(1, "active"), (0, "decoy")]

        for lab, name in label_specs:
            sub = df[ok & (df["label"] == lab)].copy()
            n_sub = int(len(sub))
            if n_sub == 0:
                continue  # omit empty group

            summary[f"by_label_{name}"] = {
                "n": n_sub,
                "pocket_in_rate": float(sub["pocket_in"].mean()),
                "pocket_out_strong_rate": float(sub["pocket_out_strong"].mean()),
                "clash_flag_rate": float(sub["clash_flag"].mean()),
                "median_d_center": float(sub["d_center"].median()),
                "median_d_min_pocket": float(sub["d_min_pocket"].median()),
                "median_n_contacts": float(sub["n_contacts"].median()),
                "median_clash_count": float(sub["clash_count"].median()),
            }

    (outdir / "pocket_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"Pocket atoms (within {float(pocket_radius)}Å of crystal ligand): {n_pocket_atoms}")
    lines.append(f"Processed Top{int(topk)}: ok={n_ok}, missing_pose={n_missing}, bad_pose={n_bad}")
    lines.append(f"Pocket-in rate (ok only): {summary['rates']['pocket_in_rate']:.3f}")
    lines.append(f"Pocket-out strong rate (ok only): {summary['rates']['pocket_out_strong_rate']:.3f}")
    lines.append(f"Clash-flag rate (ok only): {summary['rates']['clash_flag_rate']:.3f}")

    if "by_label_active" in summary:
        lines.append(f"Active pocket-in rate: {summary['by_label_active']['pocket_in_rate']:.3f}")
    if "by_label_decoy" in summary:
        lines.append(f"Decoy  pocket-in rate: {summary['by_label_decoy']['pocket_in_rate']:.3f}")

    (outdir / "pocket_qc_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary["_outputs"] = {
        "csv": str(out_csv),
        "flagged_outside": str(outdir / "flagged_outside.csv"),
        "flagged_clash": str(outdir / "flagged_clash.csv"),
        "summary_json": str(outdir / "pocket_qc_summary.json"),
        "summary_txt": str(outdir / "pocket_qc_summary.txt"),
    }
    return summary