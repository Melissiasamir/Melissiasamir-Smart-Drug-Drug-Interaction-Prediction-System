"""Build graph structures from drug interaction datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class GraphData:
    node_names: list[str]
    node_to_index: dict[str, int]
    edge_index: list[tuple[int, int]]
    node_features: dict[str, dict[str, float]]


def normalize_drug_name(drug_name: str) -> str:
    return drug_name.strip().title()


def build_graph_data(interactions: pd.DataFrame, initial_features: dict[str, dict[str, float]] | None = None) -> GraphData:
    if interactions is None or interactions.empty:
        return GraphData(node_names=[], node_to_index={}, edge_index=[], node_features={})

    drug_1 = interactions.get("Drug 1", pd.Series(dtype="object")).fillna("")
    drug_2 = interactions.get("Drug 2", pd.Series(dtype="object")).fillna("")
    nodes = sorted({normalize_drug_name(str(name)) for name in drug_1.tolist() + drug_2.tolist() if str(name).strip()})
    node_to_index = {name: index for index, name in enumerate(nodes)}
    edge_index: list[tuple[int, int]] = []

    for left, right in zip(drug_1.tolist(), drug_2.tolist()):
        left_name = normalize_drug_name(str(left))
        right_name = normalize_drug_name(str(right))
        if left_name and right_name and left_name in node_to_index and right_name in node_to_index:
            edge_index.append((node_to_index[left_name], node_to_index[right_name]))
            edge_index.append((node_to_index[right_name], node_to_index[left_name]))

    node_features = {}
    if initial_features is None:
        initial_features = {}

    for name in nodes:
        if name in initial_features:
            node_features[name] = initial_features[name]
        else:
            node_features[name] = {"name_length": float(len(name)), "name_token_count": float(len(name.split()))}

    return GraphData(node_names=nodes, node_to_index=node_to_index, edge_index=edge_index, node_features=node_features)


def load_interaction_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_interaction_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames_list = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames_list:
        return pd.DataFrame(columns=["Drug 1", "Drug 2", "Interaction Description"])
    return pd.concat(frames_list, ignore_index=True).drop_duplicates(subset=["Drug 1", "Drug 2", "Interaction Description"]).reset_index(drop=True)
