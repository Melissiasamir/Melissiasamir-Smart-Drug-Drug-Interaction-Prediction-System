"""Cluster assignment helpers for embedding-based drug clusters."""

from __future__ import annotations

from .embedding_cluster import EmbeddingClusterer


def assign_cluster(embedding: list[float], clusterer: EmbeddingClusterer) -> int:
    return int(clusterer.predict(embedding))
