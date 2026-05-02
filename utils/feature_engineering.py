"""# 4. Feature Engineering

Creates numeric features suitable for clustering, SVM classification, SHAP, and
SARIMA aggregation. Features include interaction frequency, co-occurrence, and
similarity score as requested.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


RISK_KEYWORDS = {
    "increase": 1.2,
    "decrease": 0.7,
    "serum concentration": 1.6,
    "adverse": 1.8,
    "toxicity": 2.3,
    "metabolism": 1.1,
    "bleeding": 2.4,
    "qt": 2.2,
    "prolong": 1.9,
    "hypertension": 1.5,
    "hypotension": 1.5,
    "photosensitizing": 1.3,
}


def _severity_from_text(text: str) -> float:
    lower = text.lower()
    score = 0.5
    for keyword, weight in RISK_KEYWORDS.items():
        if keyword in lower:
            score += weight
    return float(score)


def _build_classification_priors(class_df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Aggregate clinical classification rows into drug-category risk priors."""
    bp_map = {"LOW": 0.5, "NORMAL": 1.0, "HIGH": 1.5}
    chol_map = {"NORMAL": 0.8, "HIGH": 1.4}
    temp = class_df.copy()
    temp["bp_score"] = temp["BP"].map(bp_map).fillna(1.0)
    temp["chol_score"] = temp["Cholesterol"].map(chol_map).fillna(1.0)
    temp["clinical_prior"] = (
        temp["Age"] / max(float(temp["Age"].max()), 1.0)
        + temp["Na_to_K"] / max(float(temp["Na_to_K"].max()), 1.0)
        + temp["bp_score"]
        + temp["chol_score"]
    )
    priors = temp.groupby("Drug")["clinical_prior"].mean().to_dict()
    return {str(k).lower(): float(v) for k, v in priors.items()}, float(temp["clinical_prior"].mean())


def build_features(ddi_df: pd.DataFrame, class_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create pair-level numeric features from DDI and classification data."""
    pair_rows = ddi_df.copy()
    drug_counts = Counter(pair_rows["Drug 1"].str.lower()) + Counter(pair_rows["Drug 2"].str.lower())
    pair_counts = Counter(pair_rows["pair_key"])

    neighbors: dict[str, set[str]] = defaultdict(set)
    for _, row in pair_rows.iterrows():
        d1, d2 = row["Drug 1"].lower(), row["Drug 2"].lower()
        neighbors[d1].add(d2)
        neighbors[d2].add(d1)

    clinical_priors, global_prior = _build_classification_priors(class_df)

    records: list[dict[str, float]] = []
    for _, row in pair_rows.iterrows():
        d1 = row["Drug 1"].lower()
        d2 = row["Drug 2"].lower()
        n1 = neighbors[d1]
        n2 = neighbors[d2]
        union = n1 | n2
        shared = n1 & n2
        description = row["Interaction Description"]
        name_similarity = SequenceMatcher(None, d1, d2).ratio()

        records.append(
            {
                "drug1_frequency": float(drug_counts[d1]),
                "drug2_frequency": float(drug_counts[d2]),
                "interaction_frequency": float(pair_counts[row["pair_key"]]),
                "co_occurrence": float(len(shared)),
                "similarity_score": float(len(shared) / len(union)) if union else 0.0,
                "name_similarity": float(name_similarity),
                "description_severity": _severity_from_text(description),
                "description_length": float(len(description)),
                "clinical_prior_drug1": clinical_priors.get(d1, global_prior),
                "clinical_prior_drug2": clinical_priors.get(d2, global_prior),
                "clinical_prior_mean": float((clinical_priors.get(d1, global_prior) + clinical_priors.get(d2, global_prior)) / 2),
            }
        )

    features = pd.DataFrame(records)
    enriched = pd.concat([pair_rows.reset_index(drop=True), features], axis=1)
    return features, enriched


def build_single_pair_features(drug_a: str, drug_b: str, enriched_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Build numeric features for a user-entered drug pair using training priors."""
    a = drug_a.strip().lower()
    b = drug_b.strip().lower()
    direct = enriched_df[
        ((enriched_df["Drug 1"].str.lower() == a) & (enriched_df["Drug 2"].str.lower() == b))
        | ((enriched_df["Drug 1"].str.lower() == b) & (enriched_df["Drug 2"].str.lower() == a))
    ]
    if not direct.empty:
        return direct[feature_columns].mean().to_frame().T

    drug1_rows = enriched_df[(enriched_df["Drug 1"].str.lower() == a) | (enriched_df["Drug 2"].str.lower() == a)]
    drug2_rows = enriched_df[(enriched_df["Drug 1"].str.lower() == b) | (enriched_df["Drug 2"].str.lower() == b)]
    base = enriched_df[feature_columns].median()
    if not drug1_rows.empty:
        base["drug1_frequency"] = drug1_rows[["drug1_frequency", "drug2_frequency"]].mean().mean()
        base["clinical_prior_drug1"] = drug1_rows[["clinical_prior_drug1", "clinical_prior_drug2"]].mean().mean()
    if not drug2_rows.empty:
        base["drug2_frequency"] = drug2_rows[["drug1_frequency", "drug2_frequency"]].mean().mean()
        base["clinical_prior_drug2"] = drug2_rows[["clinical_prior_drug1", "clinical_prior_drug2"]].mean().mean()
    base["name_similarity"] = SequenceMatcher(None, a, b).ratio()
    base["clinical_prior_mean"] = np.mean([base["clinical_prior_drug1"], base["clinical_prior_drug2"]])
    return base[feature_columns].to_frame().T

