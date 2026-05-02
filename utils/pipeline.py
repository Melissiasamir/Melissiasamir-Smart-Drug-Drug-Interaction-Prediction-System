"""End-to-end training and inference pipeline for Smart Drug Risk Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
from utils.classifier import ClassifierArtifacts, predict_risk, train_svm_classifier
from utils.clustering import generate_pseudo_labels
from utils.data_loader import load_datasets
from utils.feature_engineering import build_features, build_single_pair_features
from utils.forecasting import create_daily_risk_series, forecast_is_increasing, train_sarima_forecast
from utils.preprocessing import clean_classification_data, clean_ddi_data, scale_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
PIPELINE_ARTIFACT_PATH = MODEL_DIR / "pipeline_artifacts.joblib"
PIPELINE_CACHE_VERSION = 1


@dataclass
class PipelineArtifacts:
    raw_ddi: object
    raw_classification: object
    ddi: object
    classification: object
    features: object
    scaled_features: object
    scaler: object
    enriched: object
    pseudo_labels: object
    clustering_metadata: dict
    classifier: ClassifierArtifacts
    risk_series: object
    forecast: object
    actual_recent: object
    forecast_error: float
    forecast_increasing: bool
    feature_columns: list[str]


def save_pipeline_artifacts(artifacts: PipelineArtifacts, path: Path = PIPELINE_ARTIFACT_PATH) -> None:
    """Persist trained pipeline artifacts so Streamlit can start quickly later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PIPELINE_CACHE_VERSION,
        "max_train_rows": int(os.getenv("MAX_TRAIN_ROWS", "4000")),
        "artifacts": artifacts,
    }
    joblib.dump(payload, path)


def load_pipeline_artifacts(path: Path = PIPELINE_ARTIFACT_PATH) -> PipelineArtifacts | None:
    """Load saved pipeline artifacts if a compatible cache exists."""
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != PIPELINE_CACHE_VERSION:
        return None
    if payload.get("max_train_rows") != int(os.getenv("MAX_TRAIN_ROWS", "4000")):
        return None
    artifacts = payload.get("artifacts")
    return artifacts if isinstance(artifacts, PipelineArtifacts) else None


def load_or_train_pipeline(force_retrain: bool = False) -> PipelineArtifacts:
    """Load persisted artifacts, or train and save them when missing."""
    if not force_retrain:
        artifacts = load_pipeline_artifacts()
        if artifacts is not None:
            return artifacts

    artifacts = train_pipeline()
    save_pipeline_artifacts(artifacts)
    return artifacts


def train_pipeline() -> PipelineArtifacts:
    """Train preprocessing, clustering, classification, and forecasting artifacts."""
    raw_ddi, raw_classification = load_datasets()
    ddi = clean_ddi_data(raw_ddi)
    classification, _ = clean_classification_data(raw_classification)
    features, full_enriched = build_features(ddi, classification)

    # Production dashboards need predictable startup latency. We keep the full
    # cleaned interaction table for pair lookup, then train the expensive
    # DBSCAN/FCM/SVM steps on a reproducible sample. Override with MAX_TRAIN_ROWS.
    max_train_rows = int(os.getenv("MAX_TRAIN_ROWS", "4000"))
    if len(features) > max_train_rows:
        train_features = features.sample(max_train_rows, random_state=42)
        train_enriched = full_enriched.loc[train_features.index].reset_index(drop=True)
        train_features = train_features.reset_index(drop=True)
    else:
        train_features = features.reset_index(drop=True)
        train_enriched = full_enriched.reset_index(drop=True)

    scaled_features, scaler = scale_features(train_features)
    pseudo_labels, clustering_metadata = generate_pseudo_labels(scaled_features, train_features)
    labeled_enriched = train_enriched.join(pseudo_labels)

    classifier = train_svm_classifier(scaled_features, pseudo_labels["pseudo_label"])
    risk_series = create_daily_risk_series(labeled_enriched, pseudo_labels["pseudo_label"])
    forecast, actual_recent, forecast_error = train_sarima_forecast(risk_series)
    increasing = forecast_is_increasing(forecast, actual_recent)

    # 9. Dynamic Reclustering: if the forecast error is too high, adapt clusters.
    # This keeps the model responsive when future risk behavior drifts.
    threshold = max(1.0, float(actual_recent.mean() * 0.35))
    if forecast_error > threshold:
        pseudo_labels, clustering_metadata = generate_pseudo_labels(scaled_features, train_features)
        labeled_enriched = labeled_enriched.drop(columns=[c for c in pseudo_labels.columns if c in labeled_enriched.columns], errors="ignore").join(pseudo_labels)
        classifier = train_svm_classifier(scaled_features, pseudo_labels["pseudo_label"])
        clustering_metadata["dynamic_reclustering"] = "Re-run DBSCAN + FCM because forecast_error > threshold"
    else:
        clustering_metadata["dynamic_reclustering"] = "No reclustering needed; forecast_error <= threshold"
    clustering_metadata["forecast_error_threshold"] = float(threshold)

    return PipelineArtifacts(
        raw_ddi=raw_ddi,
        raw_classification=raw_classification,
        ddi=ddi,
        classification=classification,
        features=train_features,
        scaled_features=scaled_features,
        scaler=scaler,
        enriched=full_enriched,
        pseudo_labels=pseudo_labels,
        clustering_metadata=clustering_metadata,
        classifier=classifier,
        risk_series=risk_series,
        forecast=forecast,
        actual_recent=actual_recent,
        forecast_error=forecast_error,
        forecast_increasing=increasing,
        feature_columns=list(features.columns),
    )


def infer_pair(artifacts: PipelineArtifacts, drug_a: str, drug_b: str) -> tuple[object, object, dict[str, object]]:
    """Create features for a user pair and predict risk."""
    row = build_single_pair_features(drug_a, drug_b, artifacts.enriched, artifacts.feature_columns)
    row_scaled = row.copy()
    row_scaled.loc[:, :] = artifacts.scaler.transform(row)
    prediction = predict_risk(artifacts.classifier.model, row_scaled)
    return row, row_scaled, prediction
