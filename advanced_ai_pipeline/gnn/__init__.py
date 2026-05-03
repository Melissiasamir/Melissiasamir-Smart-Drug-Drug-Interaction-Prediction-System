"""GNN module package for advanced AI pipeline."""

from .embedder import GNNEmbedder
from .features import extract_features
from .graph_builder import GraphData, build_graph_data, load_interaction_csv, merge_interaction_frames
from .model import build_gnn_model, has_gnn_support

__all__ = [
    "GNNEmbedder",
    "extract_features",
    "GraphData",
    "build_graph_data",
    "load_interaction_csv",
    "merge_interaction_frames",
    "build_gnn_model",
    "has_gnn_support",
]
