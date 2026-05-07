"""End-to-end training and inference pipeline for Smart Drug Risk Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from utils.classifier import ClassifierArtifacts, predict_risk, train_svm_classifier
from utils.clustering import fuzzy_membership_for_row, generate_pseudo_labels
from utils.data_loader import load_datasets
from utils.drift_detection import evaluate_self_learning_drift
from utils.explainability import compute_shap_importance
from utils.feature_engineering import build_features, build_single_pair_features
from utils.forecasting import create_daily_risk_series, forecast_is_increasing, train_sarima_forecast
from utils.preprocessing import clean_classification_data, clean_ddi_data, scale_features
from utils.rag_retrieval import build_rag_index, load_rag_index, retrieve_similar_cases
from utils.react_engine import build_observation, run_react_loop
from utils.scoring import compute_arima_multiplier, compute_scores, compute_svm_gap
from utils.shap_memory import build_shap_cluster_memory, load_shap_memory, shap_cluster_reliability, shap_importance_to_vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
PIPELINE_ARTIFACT_PATH = MODEL_DIR / "pipeline_artifacts.joblib"
SHAP_MEMORY_PATH = MODEL_DIR / "shap_cluster_memory.joblib"
RAG_INDEX_PATH = MODEL_DIR / "rag_case_index.joblib"
PIPELINE_CACHE_VERSION = 3


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
    shap_memory: dict | None
    rag_index: dict | None
    baseline_svm_gap: float
    drift_status: dict | None


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

    artifacts = payload.get("artifacts")

    return artifacts 

# def load_or_train_pipeline(force_retrain: bool = False) -> PipelineArtifacts:
#     """Load persisted artifacts, or train and save them when missing."""
#     if not force_retrain:
#         artifacts = load_pipeline_artifacts()
#         if artifacts is not None:
#             return artifacts

#     artifacts = train_pipeline()
#     save_pipeline_artifacts(artifacts)
#     return artifacts

def load_or_train_pipeline(force_retrain: bool = False) -> PipelineArtifacts:
    """Load persisted artifacts, or train and save them when missing."""

    print("DEBUG: entered load_or_train_pipeline")

    if not force_retrain:
        print("DEBUG: trying cached artifacts")
        artifacts = load_pipeline_artifacts()

        if artifacts is not None:
            print("DEBUG: LOADED FROM CACHE")
            return artifacts

    print("DEBUG: TRAINING PIPELINE")

    artifacts = train_pipeline()

    print("DEBUG: SAVING PIPELINE")

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

    baseline_svm_gap = _mean_svm_gap(classifier.model, scaled_features)
    shap_memory = build_shap_cluster_memory(
        classifier.model,
        scaled_features,
        scaled_features,
        pseudo_labels,
        list(features.columns),
        SHAP_MEMORY_PATH,
        max_rows=int(os.getenv("SHAP_MEMORY_ROWS", "120")),
    )
    rag_index = build_rag_index(
        shap_memory["case_feature_vectors"],
        shap_memory["case_shap_vectors"],
        shap_memory["case_labels"],
        list(features.columns),
        RAG_INDEX_PATH,
    )
    drift_status = evaluate_self_learning_drift(
        baseline_svm_gap=baseline_svm_gap,
        recent_gaps=[baseline_svm_gap],
        old_clustering_metadata=clustering_metadata,
        new_clustering_metadata=clustering_metadata,
    )

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
        shap_memory=shap_memory,
        rag_index=rag_index,
        baseline_svm_gap=baseline_svm_gap,
        drift_status=drift_status,
    )


def infer_pair(
    artifacts: PipelineArtifacts,
    drug_a: str,
    drug_b: str,
    run_agentic_decision: bool = True,
) -> tuple[object, object, dict[str, object]]:
    """Create features for a user pair and return prediction plus agentic decision.

    Integration order:
    prediction -> SHAP explanation -> SHAP reliability -> FCM membership ->
    intelligent scoring -> RAG retrieval only when borderline -> ReAct routing.

    The tuple return is preserved for Streamlit/report compatibility. New
    outputs are attached to the prediction dictionary.
    """
    row = build_single_pair_features(drug_a, drug_b, artifacts.enriched, artifacts.feature_columns)
    row_scaled = row.copy()
    row_scaled.loc[:, :] = artifacts.scaler.transform(row)
    prediction = predict_risk(artifacts.classifier.model, row_scaled)
    prediction["svm_gap"] = compute_svm_gap(prediction.get("probabilities"), float(prediction["confidence"]))
    prediction["fuzzy_memberships"] = fuzzy_membership_for_row(row_scaled, artifacts.clustering_metadata)
    prediction["fuzzy_membership"] = float(prediction["fuzzy_memberships"].get(str(prediction["label"]), 0.5))
    prediction["arima_multiplier"] = forecast_trend_multiplier(artifacts.forecast, artifacts.actual_recent, artifacts.forecast_increasing)
    if artifacts.shap_memory is None:
        artifacts.shap_memory = load_shap_memory(SHAP_MEMORY_PATH)
    if artifacts.rag_index is None:
        artifacts.rag_index = load_rag_index(RAG_INDEX_PATH)
    prediction["rag_available"] = artifacts.rag_index is not None
    if run_agentic_decision:
        prediction["agentic_decision"] = build_agentic_decision(artifacts, row_scaled, prediction)
        prediction.update(
            {
                "shap_importance": prediction["agentic_decision"]["shap_importance"],
                "shap_reliability": prediction["agentic_decision"]["shap_reliability"],
                "scores": prediction["agentic_decision"]["scores"],
                "rag_result": prediction["agentic_decision"]["rag_result"],
                "react_decision": prediction["agentic_decision"]["react_decision"],
                "final_decision": prediction["agentic_decision"]["final_decision"],
            }
        )
    return row, row_scaled, prediction


def build_agentic_decision(
    artifacts: PipelineArtifacts,
    row_scaled,
    prediction: dict[str, object],
) -> dict[str, object]:
    """Build the final structured decision object for one inference."""
    shap_importance = compute_shap_importance(
        artifacts.classifier.model,
        artifacts.scaled_features,
        row_scaled,
        artifacts.feature_columns,
    )
    shap_vector = shap_importance_to_vector(shap_importance, artifacts.feature_columns)
    shap_reliability = shap_cluster_reliability(shap_vector, str(prediction["label"]), artifacts.shap_memory)
    scores = compute_scores(
        float(prediction["confidence"]),
        shap_importance,
        bool(artifacts.forecast_increasing),
        str(prediction["label"]),
        prediction_probabilities=prediction.get("probabilities"),
        fuzzy_membership=prediction.get("fuzzy_memberships"),
        shap_cosine_similarity=shap_reliability["cosine_similarity"],
        arima_multiplier=prediction.get("arima_multiplier"),
    )

    rag_cache: dict[str, object] = {}

    def retrieve_current_cases() -> dict[str, object]:
        if not rag_cache:
            rag_cache.update(retrieve_similar_cases(artifacts.rag_index, row_scaled, shap_vector, top_k=5))
        return rag_cache

    dominant_feature = str(shap_importance.iloc[0]["feature"]) if not shap_importance.empty else None
    observation = build_observation(
        final_score=float(scores["final_agentic_score"]),
        shap_dominant_feature=dominant_feature,
        rag_vote=None,
        arima_multiplier=float(scores["arima_multiplier"]),
        arima_signal=float(scores["arima_signal"]),
        reliability_score=float(scores["shap_reliability_signal"]),
        prediction_label=str(prediction["label"]),
    )
    react_decision = run_react_loop(observation, retrieval_fn=retrieve_current_cases)
    rag_result = rag_cache or {"status": "not_requested", "cases": [], "vote": {"status": "not_requested"}}

    return {
        "prediction_label": str(prediction["label"]),
        "shap_importance": shap_importance,
        "shap_vector": shap_vector,
        "shap_reliability": shap_reliability,
        "scores": scores,
        "rag_result": rag_result,
        "react_decision": react_decision,
        "final_decision": react_decision["final_decision"],
        "integration_order": [
            "prediction",
            "shap_explanation",
            "shap_reliability_lookup",
            "fuzzy_membership_lookup",
            "intelligent_scoring",
            "rag_retrieval_borderline_only",
            "react_reasoning_loop",
            "final_decision",
        ],
    }


def _mean_svm_gap(model, scaled_features) -> float:
    probabilities = model.predict_proba(scaled_features)
    sorted_probs = -np.sort(-probabilities, axis=1)
    gaps = sorted_probs[:, 0] - sorted_probs[:, 1]
    return float(gaps.mean())


def forecast_trend_multiplier(forecast, actual_recent, forecast_increasing: bool) -> float:
    """Compute a bounded SARIMA multiplier from future-vs-recent risk intensity."""
    try:
        future_half = forecast.iloc[len(forecast) // 2 :]
        recent_mean = float(actual_recent.mean())
        future_mean = float(future_half.mean())
        ratio = future_mean / max(recent_mean, 1e-6)
        multiplier = 1.0 + (0.15 * float(np.tanh(ratio - 1.0)))
        return compute_arima_multiplier(forecast_increasing, multiplier)
    except Exception:
        return compute_arima_multiplier(forecast_increasing)
