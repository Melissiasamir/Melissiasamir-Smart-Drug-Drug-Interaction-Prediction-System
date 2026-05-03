"""Interaction persistence and graph-building for doctor-submitted cases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..gnn.graph_builder import GraphData, build_graph_data, load_interaction_csv, merge_interaction_frames


class InteractionProcessor:
    def __init__(self, base_path: Path | None = None, doctor_excel_path: Path | None = None) -> None:
        self.base_path = base_path or Path(__file__).resolve().parents[2] / "data" / "drug_drug_interactions.csv"
        self.doctor_excel_path = doctor_excel_path or Path(__file__).resolve().parents[2] / "data" / "doctor_added_interactions.xlsx"

    def load_base_interactions(self) -> pd.DataFrame:
        return load_interaction_csv(self.base_path)

    def load_doctor_interactions(self) -> pd.DataFrame:
        if not self.doctor_excel_path.exists():
            return pd.DataFrame(columns=["Submitted At", "Drug 1", "Drug 2", "Interaction Description"])
        return pd.read_excel(self.doctor_excel_path)

    def save_doctor_interaction(self, drug_1: str, drug_2: str, description: str) -> Path:
        self.doctor_excel_path.parent.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame(
            [
                {
                    "Submitted At": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Drug 1": drug_1,
                    "Drug 2": drug_2,
                    "Interaction Description": description,
                }
            ]
        )
        existing = self.load_doctor_interactions()
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_excel(self.doctor_excel_path, index=False, sheet_name="Doctor Interactions")
        return self.doctor_excel_path

    def build_graph(self, include_doctor_interactions: bool = True, initial_features: dict[str, dict[str, float]] | None = None) -> GraphData:
        base = self.load_base_interactions()
        if include_doctor_interactions:
            doctor = self.load_doctor_interactions()
            if not doctor.empty and "Drug 1" in doctor.columns and "Drug 2" in doctor.columns:
                doctor = doctor[["Drug 1", "Drug 2", "Interaction Description"]]
            interactions = merge_interaction_frames([base, doctor])
        else:
            interactions = base
        return build_graph_data(interactions, initial_features=initial_features)
