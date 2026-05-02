"""# 3. Data Preprocessing

This module handles missing values, data cleaning, categorical encoding, and
feature scaling with StandardScaler. Each step is intentionally explicit because
medical AI systems must be auditable and reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_drug_name(value: object) -> str:
    """Normalize drug names so matching is consistent across datasets."""
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.strip())
    return text


def clean_ddi_data(ddi_df: pd.DataFrame) -> pd.DataFrame:
    """Clean DDI rows, fill missing descriptions, and remove duplicate pairs."""
    df = ddi_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["Drug 1"] = df["Drug 1"].apply(clean_drug_name)
    df["Drug 2"] = df["Drug 2"].apply(clean_drug_name)
    df["Interaction Description"] = df["Interaction Description"].fillna("No description available")
    df["Interaction Description"] = df["Interaction Description"].astype(str).str.strip()
    df = df[(df["Drug 1"] != "") & (df["Drug 2"] != "")]
    df["pair_key"] = df.apply(lambda r: "||".join(sorted([r["Drug 1"].lower(), r["Drug 2"].lower()])), axis=1)
    df = df.drop_duplicates(subset=["pair_key", "Interaction Description"]).reset_index(drop=True)
    return df


@dataclass
class ClassificationPreprocessor:
    """Stores encoders used for the drug classification dataset."""

    encoders: dict[str, LabelEncoder]


def clean_classification_data(class_df: pd.DataFrame) -> tuple[pd.DataFrame, ClassificationPreprocessor]:
    """Handle missing values and encode categorical columns numerically."""
    df = class_df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Numeric missing values use medians because medians are robust to outliers.
    for col in ["Age", "Na_to_K"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Categorical missing values use mode because it preserves valid categories.
    for col in ["Sex", "BP", "Cholesterol", "Drug"]:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0]).astype(str).str.strip()

    encoders: dict[str, LabelEncoder] = {}
    for col in ["Sex", "BP", "Cholesterol", "Drug"]:
        encoder = LabelEncoder()
        df[f"{col}_encoded"] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    return df, ClassificationPreprocessor(encoders=encoders)


def scale_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric ML features with StandardScaler for SVM/DBSCAN stability."""
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )
    return scaled, scaler

