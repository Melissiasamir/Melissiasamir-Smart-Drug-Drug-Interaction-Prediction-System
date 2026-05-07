"""Lightweight RAG retrieval for borderline drug-pair predictions.

The index uses sklearn NearestNeighbors over concatenated vectors:

    retrieval_vector = [scaled_feature_vector, shap_vector]

It is intentionally small and local; no external vector database is required.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


RAG_INDEX_VERSION = 1


@dataclass(frozen=True)
class RetrievalConfig:
    """Thresholds for deciding whether retrieved neighbors are usable evidence."""

    min_similarity: float = 0.65
    min_evidence_cases: int = 2
    low_confidence_threshold: float = 0.60


def _as_2d(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def build_case_matrix(feature_vectors: Any, shap_vectors: Any) -> np.ndarray:
    """Build the RAG vector matrix from features plus SHAP explanations."""
    features = _as_2d(feature_vectors)
    shap = _as_2d(shap_vectors)
    n = min(len(features), len(shap))
    if n == 0:
        return np.empty((0, 0), dtype=float)
    return np.hstack([features[:n], shap[:n]])


def build_rag_index(
    feature_vectors: Any,
    shap_vectors: Any,
    labels: list[str],
    feature_names: list[str],
    path: Path,
    n_neighbors: int = 8,
) -> dict[str, Any] | None:
    """Fit and persist a NearestNeighbors index for historical cases."""
    matrix = build_case_matrix(feature_vectors, shap_vectors)
    if matrix.size == 0 or len(matrix) < 2:
        return None

    neighbors = NearestNeighbors(n_neighbors=min(n_neighbors, len(matrix)), metric="cosine")
    neighbors.fit(matrix)
    payload = {
        "version": RAG_INDEX_VERSION,
        "created_at": datetime.now().isoformat(),
        "feature_names": list(feature_names),
        "matrix": matrix,
        "labels": [str(label) for label in labels[: len(matrix)]],
        "neighbors": neighbors,
        "n_cases": int(len(matrix)),
        "retrieval_config": RetrievalConfig().__dict__,
    }
    save_rag_index(payload, path)
    return payload


def save_rag_index(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_rag_index(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("version") != RAG_INDEX_VERSION:
        return None
    return payload


def vote_cases(cases: list[dict[str, Any]], low_confidence_threshold: float = 0.60) -> dict[str, Any]:
    """Majority vote with tie, no-evidence, and low-confidence detection."""
    if not cases:
        return {
            "status": "no_evidence",
            "majority_label": None,
            "vote_confidence": 0.0,
            "counts": {},
            "is_tie": False,
            "is_low_confidence": True,
            "evidence_count": 0,
        }

    counts: dict[str, int] = {}
    for case in cases:
        label = str(case["label"])
        counts[label] = counts.get(label, 0) + 1

    max_votes = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_votes]
    is_tie = len(winners) > 1
    majority_label = None if is_tie else winners[0]
    vote_confidence = float(max_votes / len(cases))
    is_low_confidence = vote_confidence < low_confidence_threshold
    status = "tie" if is_tie else "low_confidence" if is_low_confidence else "ok"

    return {
        "status": status,
        "majority_label": majority_label,
        "vote_confidence": vote_confidence,
        "counts": counts,
        "is_tie": is_tie,
        "is_low_confidence": is_low_confidence,
        "evidence_count": len(cases),
    }


def filter_weak_neighbors(
    cases: list[dict[str, Any]],
    min_similarity: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply similarity filtering after nearest-neighbor retrieval.

    Weak neighbors are kept in ``rejected_cases`` for debugging, but they do not
    participate in majority voting.
    """
    strong_cases = [case for case in cases if float(case.get("similarity", 0.0)) >= min_similarity]
    weak_cases = [case for case in cases if float(case.get("similarity", 0.0)) < min_similarity]
    return strong_cases, weak_cases


def retrieve_similar_cases(
    rag_index: dict[str, Any] | None,
    row_scaled: pd.DataFrame,
    shap_vector: np.ndarray,
    top_k: int = 5,
    min_similarity: float | None = None,
    min_evidence_cases: int | None = None,
    low_confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Retrieve top-k historical cases and summarize their vote."""
    index_config = (rag_index or {}).get("retrieval_config") or {}
    config = RetrievalConfig(
        min_similarity=float(min_similarity if min_similarity is not None else index_config.get("min_similarity", 0.65)),
        min_evidence_cases=int(min_evidence_cases if min_evidence_cases is not None else index_config.get("min_evidence_cases", 2)),
        low_confidence_threshold=float(
            low_confidence_threshold
            if low_confidence_threshold is not None
            else index_config.get("low_confidence_threshold", 0.60)
        ),
    )

    if not rag_index or "neighbors" not in rag_index:
        return {"status": "no_evidence", "cases": [], "rejected_cases": [], "vote": vote_cases([]), "config": config.__dict__}

    query = build_case_matrix(row_scaled.to_numpy(dtype=float), np.asarray(shap_vector, dtype=float))
    if query.size == 0:
        return {"status": "no_evidence", "cases": [], "rejected_cases": [], "vote": vote_cases([]), "config": config.__dict__}

    neighbors = rag_index["neighbors"]
    distances, indices = neighbors.kneighbors(query, n_neighbors=min(top_k, int(rag_index.get("n_cases", top_k))))
    labels = rag_index.get("labels") or []
    cases = []
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if int(idx) >= len(labels):
            continue
        cases.append(
            {
                "rank": rank,
                "label": str(labels[int(idx)]),
                "distance": float(distance),
                "similarity": float(1.0 - distance),
            }
        )

    strong_cases, weak_cases = filter_weak_neighbors(cases, config.min_similarity)
    if len(strong_cases) < config.min_evidence_cases:
        vote = vote_cases([], config.low_confidence_threshold)
        vote["reason"] = "not_enough_neighbors_above_similarity_threshold"
        return {
            "status": "no_evidence",
            "cases": strong_cases,
            "rejected_cases": weak_cases,
            "raw_cases": cases,
            "vote": vote,
            "config": config.__dict__,
            "evidence_strength": 0.0,
        }

    vote = vote_cases(strong_cases, config.low_confidence_threshold)
    evidence_strength = float(np.mean([case["similarity"] for case in strong_cases])) if strong_cases else 0.0
    return {
        "status": vote["status"],
        "cases": strong_cases,
        "rejected_cases": weak_cases,
        "raw_cases": cases,
        "vote": vote,
        "config": config.__dict__,
        "evidence_strength": evidence_strength,
    }
