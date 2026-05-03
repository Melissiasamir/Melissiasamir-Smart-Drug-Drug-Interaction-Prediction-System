"""Embedding-based clustering helpers for the advanced AI pipeline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None


class EmbeddingClusterer:
    def __init__(self, n_clusters: int = 8) -> None:
        self.n_clusters = n_clusters
        self.cluster_centers_: list[np.ndarray] = []
        self.labels_: dict[str, int] = {}
        self.model: Any | None = None

    def fit(self, embeddings: dict[str, list[float]]) -> None:
        if not embeddings:
            self.cluster_centers_ = []
            self.labels_ = {}
            self.model = None
            return

        vectors = np.vstack([embeddings[name] for name in sorted(embeddings)])
        names = sorted(embeddings)

        if KMeans is not None and len(embeddings) >= self.n_clusters:
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.model.fit(vectors)
            self.cluster_centers_ = [center for center in self.model.cluster_centers_]
            self.labels_ = {name: int(label) for name, label in zip(names, self.model.labels_)}
            return

        self.cluster_centers_ = [vectors[i] for i in range(min(self.n_clusters, len(vectors)))]
        self.labels_ = {name: int(i % self.n_clusters) for i, name in enumerate(names)}

    def predict(self, embedding: list[float]) -> int:
        if self.model is not None:
            try:
                return int(self.model.predict([embedding])[0])
            except Exception:
                pass

        if not self.cluster_centers_:
            return 0

        target = np.asarray(embedding, dtype=float)
        best_index = 0
        best_similarity = -math.inf
        for index, center in enumerate(self.cluster_centers_):
            similarity = float(np.dot(target, center) / max(np.linalg.norm(target) * np.linalg.norm(center), 1e-9))
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        return int(best_index)
