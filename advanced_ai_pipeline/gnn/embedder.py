"""Embedding manager for GNN-based drug representations."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .features import extract_features
from .graph_builder import GraphData
from .model import build_gnn_model, has_gnn_support

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "gnn_embedding_cache.pkl"


def _serialize_payload(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(payload, path)
        return
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _deserialize_payload(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        if joblib is not None:
            return joblib.load(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _feature_dict_to_vector(features: dict[str, float], size: int = 16) -> np.ndarray:
    values = [float(value) for _, value in sorted(features.items())]
    vector = np.array(values, dtype=float)
    if vector.shape[0] >= size:
        return vector[:size]
    padded = np.zeros(size, dtype=float)
    padded[: vector.shape[0]] = vector
    return padded


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-9)


class GNNEmbedder:
    def __init__(self, cache_path: Path | None = None, output_dim: int = 32) -> None:
        self.cache_path = cache_path or CACHE_PATH
        self.output_dim = output_dim
        self.cache: dict[str, list[float]] = self._load_cache() or {}
        self.model = None

    def _load_cache(self) -> dict[str, list[float]] | None:
        payload = _deserialize_payload(self.cache_path)
        if isinstance(payload, dict):
            return payload
        return None

    def _save_cache(self) -> None:
        _serialize_payload(self.cache, self.cache_path)

    def _build_feature_matrix(self, graph: GraphData) -> np.ndarray:
        rows = []
        for drug_name in graph.node_names:
            features = graph.node_features.get(drug_name) or extract_features(drug_name)
            rows.append(_feature_dict_to_vector(features, size=16))
        return np.vstack(rows).astype(float)

    def _build_edge_index(self, graph: GraphData):
        try:
            import torch
        except Exception:
            return None
        if not graph.edge_index:
            return None
        edges = list(zip(*graph.edge_index))
        return torch.tensor(edges, dtype=torch.long)

    def _ensure_model(self, graph: GraphData) -> bool:
        if self.model is not None:
            return True
        if not has_gnn_support() or not graph.node_names:
            return False
        input_dim = 16
        self.model = build_gnn_model(input_dim=input_dim, hidden_dim=64, output_dim=self.output_dim)
        return self.model is not None

    def _compute_embeddings(self, graph: GraphData) -> dict[str, np.ndarray]:
        feature_matrix = self._build_feature_matrix(graph)
        if self._ensure_model(graph):
            import torch
            self.model.eval()
            x = torch.tensor(feature_matrix, dtype=torch.float32)
            edge_index = self._build_edge_index(graph)
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            with torch.no_grad():
                embedded = self.model(x, edge_index)
            return {drug: _normalize_vector(embedded[idx].cpu().numpy()) for idx, drug in enumerate(graph.node_names)}

        return {drug: _normalize_vector(feature_matrix[idx]) for idx, drug in enumerate(graph.node_names)}

    def get_embedding(self, drug_name: str) -> list[float] | None:
        return self.cache.get(drug_name)

    def generate_embedding(self, drug_name: str, features: dict[str, float], graph: GraphData) -> list[float]:
        normalized_name = drug_name.strip().title()
        if normalized_name in self.cache:
            return self.cache[normalized_name]

        graph.node_features[normalized_name] = features
        if normalized_name not in graph.node_names:
            graph.node_names.append(normalized_name)
            graph.node_to_index = {name: idx for idx, name in enumerate(graph.node_names)}

        embeddings = self._compute_embeddings(graph)
        result = embeddings.get(normalized_name)
        if result is None:
            result = _normalize_vector(_feature_dict_to_vector(features, size=16))
        self.cache[normalized_name] = result.tolist()
        self._save_cache()
        print(f"[Advanced AI] Embeddings generated for '{normalized_name}'")
        return self.cache[normalized_name]
