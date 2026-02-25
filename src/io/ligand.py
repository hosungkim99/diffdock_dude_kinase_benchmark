# src/domain/ligand.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from rdkit import Chem


# split 표준값 (프로젝트 전역 정책)
SPLIT_ACTIVE = "active"
SPLIT_DECOY = "decoy"


def build_complex_name(target: str, split: str, idx: int) -> str:
    """
    target/split/index로 complex_name 생성.
    예: akt1_active_000001, akt1_decoy_000123
    """
    if split not in (SPLIT_ACTIVE, SPLIT_DECOY):
        raise ValueError(f"split must be '{SPLIT_ACTIVE}' or '{SPLIT_DECOY}', got: {split}")
    if idx < 1:
        raise ValueError(f"idx must be >= 1, got: {idx}")
    return f"{target}_{split}_{idx:06d}"


def infer_label_from_complex_name(complex_name: str) -> int:
    """
    complex_name에서 label 추론.
    active -> 1, decoy -> 0
    """
    if f"_{SPLIT_ACTIVE}_" in complex_name:
        return 1
    if f"_{SPLIT_DECOY}_" in complex_name:
        return 0
    raise ValueError(f"Cannot infer label from complex_name: {complex_name}")


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    """
    RDKit Mol -> SMILES 변환. sanitize 실패/None은 호출측에서 걸러야 한다.
    """
    # Chem.MolToSmiles는 기본 canonical=True로 동작하지만 명시해둔다.
    return Chem.MolToSmiles(mol, canonical=canonical)
