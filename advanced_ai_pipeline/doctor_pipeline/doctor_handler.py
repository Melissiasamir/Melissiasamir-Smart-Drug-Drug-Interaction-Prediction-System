"""Doctor-facing pipeline integration for the advanced AI extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..clustering.embedding_cluster import EmbeddingClusterer
from .drug_processor import DrugProcessor, DrugStore
from .interaction_processor import InteractionProcessor
from ..gnn.graph_builder import GraphData
from ..similarity.similarity_engine import SimilarityEngine


def normalize_drug_name(drug_input: str) -> str:
    return drug_input.strip().title()


class DoctorPipeline:
    def __init__(self) -> None:
        self.interaction_processor = InteractionProcessor()
        self.store = DrugStore()
        self.graph = self.interaction_processor.build_graph(initial_features={})
        self.processor = DrugProcessor(self.graph, self.store)
        self.clusterer = EmbeddingClusterer(n_clusters=8)
        self._retrain_clusters()

    def _retrain_clusters(self) -> None:
        self.clusterer.fit(self.store.all_embeddings())

    def _build_similarity_engine(self) -> SimilarityEngine:
        embeddings = self.store.all_embeddings()
        return SimilarityEngine(embeddings)

    def add_interaction(self, drug_1: str, drug_2: str, description: str) -> Path:
        return self.interaction_processor.save_doctor_interaction(drug_1, drug_2, description)

    def process_drug(self, drug_input: str) -> dict[str, Any]:
        drug_info = self.processor.process_drug(drug_input)
        self._retrain_clusters()
        cluster = self.clusterer.predict(drug_info["embedding"])
        drug_info["cluster"] = int(cluster)
        self.store.add(drug_info["name"], drug_info["input"], drug_info["features"], drug_info["embedding"], cluster)
        return drug_info

    def update_graph(self, drug_1: str, drug_2: str) -> None:
        self.graph = self.interaction_processor.build_graph(initial_features={})
        self.processor.graph = self.graph

    def get_similar_drugs(self, drug_input: str, top_n: int = 5) -> list[dict[str, Any]]:
        engine = self._build_similarity_engine()
        drug_info = self.store.get(normalize_drug_name(drug_input))
        if not drug_info:
            return []
        return engine.get_similar_drugs(drug_info["embedding"], top_n=top_n)

    def handle_doctor_input(self, drug_1: str, drug_2: str, description: str) -> dict[str, Any]:
        self.add_interaction(drug_1, drug_2, description)
        self.update_graph(drug_1, drug_2)
        drug1_info = self.process_drug(drug_1)
        drug2_info = self.process_drug(drug_2)
        drug1_similar = self.get_similar_drugs(drug_1)
        drug2_similar = self.get_similar_drugs(drug_2)
        print(f"[Advanced AI] Processed doctor input for '{drug_1}' and '{drug_2}'")
        return {
            "drug1": drug1_info["name"],
            "drug2": drug2_info["name"],
            "drug1_cluster": int(drug1_info["cluster"]),
            "drug2_cluster": int(drug2_info["cluster"]),
            "similar_drugs_1": drug1_similar,
            "similar_drugs_2": drug2_similar,
            "status": "processed",
        }
