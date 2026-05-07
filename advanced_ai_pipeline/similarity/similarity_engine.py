"""(similarity_engine.py) Similarity search engine for drug embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _normalize(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(vector), dtype=float)
    norm = np.linalg.norm(array)
    return array / max(norm, 1e-9)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


@dataclass
class SimilarDrug:
    name: str
    score: float


class SimilarityEngine:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = {name: _normalize(values) for name, values in embeddings.items()}

    def get_similar_drugs(self, embedding: list[float], top_n: int = 5) -> list[dict[str, object]]:
        if not self.embeddings:
            return []

        query = _normalize(embedding)
        scores = []
        for name, vector in self.embeddings.items():
            similarity = _cosine_similarity(query, vector)
            scores.append(SimilarDrug(name=name, score=similarity))

        scores.sort(key=lambda item: item.score, reverse=True)
        return [
            {"drug": item.name, "score": round(item.score, 4)}
            for item in scores[:top_n]
            if item.score > 0.0
        ]
