"""Drug processing workflow for new doctor-submitted drugs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..clustering.cluster_assigner import assign_cluster
from ..clustering.embedding_cluster import EmbeddingClusterer
from ..gnn.embedder import GNNEmbedder
from ..gnn.features import extract_features
from ..gnn.graph_builder import GraphData
from ..similarity.similarity_engine import SimilarityEngine


STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "doctor_drug_store.json"


def normalize_drug_name(drug_input: str) -> str:
    return drug_input.strip().title()


class DrugStore:
    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or STORE_PATH
        self.drugs: dict[str, dict[str, Any]] = self._load_store()

    def _load_store(self) -> dict[str, dict[str, Any]]:
        if not self.storage_path.exists():
            return {}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as handle:
            json.dump(self.drugs, handle, indent=2)

    def exists(self, drug_name: str) -> bool:
        return normalize_drug_name(drug_name) in self.drugs

    def get(self, drug_name: str) -> dict[str, Any] | None:
        return self.drugs.get(normalize_drug_name(drug_name))

    def add(self, drug_name: str, drug_input: str, features: dict[str, float], embedding: list[float], cluster: int) -> dict[str, Any]:
        normalized = normalize_drug_name(drug_name)
        info = {
            "name": normalized,
            "input": drug_input,
            "features": features,
            "embedding": embedding,
            "cluster": int(cluster),
        }
        self.drugs[normalized] = info
        self.save()
        return info

    def all_embeddings(self) -> dict[str, list[float]]:
        return {name: data["embedding"] for name, data in self.drugs.items() if "embedding" in data}


class DrugProcessor:
    def __init__(self, graph: GraphData, store: DrugStore | None = None) -> None:
        self.graph = graph
        self.store = store or DrugStore()
        self.embedder = GNNEmbedder()
        self.clusterer = EmbeddingClusterer(n_clusters=8)

    def _retrain_clusterer(self) -> None:
        all_embeddings = self.store.all_embeddings()
        if all_embeddings:
            self.clusterer.fit(all_embeddings)

    def process_drug(self, drug_input: str) -> dict[str, Any]:
        normalized = normalize_drug_name(drug_input)
        existing = self.store.get(normalized)
        if existing is not None:
            print(f"[Advanced AI] Drug '{normalized}' already exists in store")
            return existing

        features = extract_features(drug_input)
        embedding = self.embedder.generate_embedding(normalized, features, self.graph)
        all_embeddings = self.store.all_embeddings()
        all_embeddings[normalized] = embedding
        self.clusterer.fit(all_embeddings)
        cluster = assign_cluster(embedding, self.clusterer)
        stored = self.store.add(normalized, drug_input, features, embedding, cluster)
        print(f"[Advanced AI] Cluster assigned for '{normalized}': {cluster}")
        return stored

    def similarity_engine(self) -> SimilarityEngine:
        all_embeddings = self.store.all_embeddings()
        return SimilarityEngine(all_embeddings)
