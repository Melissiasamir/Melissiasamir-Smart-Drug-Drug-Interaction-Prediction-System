"""Feature extraction helpers for GNN drug embeddings."""

from __future__ import annotations

import re
from typing import Any

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

SMILES_PATTERN = re.compile(r"^[BCNOFPSclArn0-9@+\-\[\]\(\)=#$]+$")


def is_smiles(drug_input: str) -> bool:
    value = drug_input.strip()
    if len(value) == 0:
        return False

    if Chem is not None:
        try:
            mol = Chem.MolFromSmiles(value)
            return mol is not None
        except Exception:
            return False

    return bool(SMILES_PATTERN.match(value)) and any(ch.isdigit() for ch in value)


def _text_statistics(drug_name: str) -> dict[str, float]:
    normalized = drug_name.strip().lower()
    letters = [ch for ch in normalized if ch.isalpha()]
    digits = [ch for ch in normalized if ch.isdigit()]
    vowels = [ch for ch in letters if ch in "aeiou"]
    unique_chars = len(set(normalized))
    word_count = len([token for token in normalized.split() if token])
    return {
        "name_length": float(len(normalized)),
        "unique_char_ratio": float(unique_chars) / max(1, len(normalized)),
        "alpha_ratio": float(len(letters)) / max(1, len(normalized)),
        "digit_ratio": float(len(digits)) / max(1, len(normalized)),
        "vowel_ratio": float(len(vowels)) / max(1, len(letters)) if letters else 0.0,
        "word_count": float(word_count),
    }


def _smiles_features(smiles: str) -> dict[str, float]:
    if Chem is None:
        return _text_statistics(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _text_statistics(smiles)

    return {
        "molecular_weight": float(Chem.Descriptors.MolWt(mol)),
        "logp": float(Chem.Crippen.MolLogP(mol)),
        "atom_count": float(mol.GetNumAtoms()),
        **_text_statistics(smiles),
    }


def extract_features(drug_input: str) -> dict[str, float]:
    """Extract a small feature vector for a drug from SMILES or drug name."""
    if not isinstance(drug_input, str) or not drug_input.strip():
        return _text_statistics(str(drug_input))

    drug_value = drug_input.strip()
    if is_smiles(drug_value):
        features = _smiles_features(drug_value)
    else:
        features = _text_statistics(drug_value)

    print(f"[Advanced AI] Features extracted for '{drug_input}': {features}")
    return features
