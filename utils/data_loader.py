"""# 1. Project Overview

Smart Drug Risk Agent loads two complementary datasets:
1. Drug-Drug Interactions: known interacting drug pairs and text descriptions.
2. Drug Classification: patient/drug category records that provide clinical context.

The system predicts risk for Drug A + Drug B, explains the decision with SHAP,
forecasts future dangerous cases with SARIMA, and can trigger real SMTP alerts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DDI_PATH = PROJECT_ROOT / "data" / "drug_drug_interactions.csv"
CLASSIFICATION_PATH = PROJECT_ROOT / "data" / "drug_classification.csv"


def load_datasets(
    ddi_path: Path = DDI_PATH,
    classification_path: Path = CLASSIFICATION_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the DDI and drug classification datasets from disk."""
    ddi_df = pd.read_csv(ddi_path)
    class_df = pd.read_csv(classification_path)
    return ddi_df, class_df


def dataset_description() -> str:
    """Return the required dataset explanation for the dashboard."""
    return (
        "The Drug-Drug Interactions dataset contains Drug 1, Drug 2, and an "
        "interaction description. It captures relational evidence: which drugs "
        "co-occur in known interaction events and how the interaction is described. "
        "The Drug Classification dataset contains demographic and clinical fields "
        "such as Age, Sex, BP, Cholesterol, Na_to_K, and prescribed Drug category. "
        "Both are used because interaction data explains pair-level behavior, while "
        "classification data adds clinical context and drug-family priors. They "
        "complement each other by combining network-style interaction patterns with "
        "patient/drug classification signals."
    )

