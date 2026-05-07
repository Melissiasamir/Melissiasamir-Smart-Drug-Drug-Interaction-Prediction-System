"""Agentic composite scoring for Smart Drug Risk Agent.

The score intentionally uses five independent signals:

1. S: class severity prior for the predicted label.
2. Gap(SVM): top probability minus second probability.
3. mu(predicted): Fuzzy C-Means membership of the predicted risk cluster.
4. ARIMA/SARIMA trend signal.
5. SHAP reliability: cosine similarity to saved cluster SHAP memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEVERITY_WEIGHTS = {"Low Risk": 0.25, "Medium Risk": 0.60, "High Risk": 1.00, "Dangerous": 1.00}
SCORE_WEIGHTS = {
    "severity": 0.30,
    "svm_gap": 0.20,
    "fuzzy_membership": 0.20,
    "shap_reliability": 0.15,
    "arima_trend": 0.15,
}


def compute_svm_gap(probabilities: dict[str, float] | None, fallback_confidence: float) -> float:
    """Gap(SVM) = max(P(class)) - second_max(P(class)).

    A small gap means the SVM decision boundary is close or ambiguous. A large
    gap means the top class is well separated from its nearest competitor.
    """
    if not probabilities:
        return float(np.clip(fallback_confidence, 0.0, 1.0))

    sorted_probs = sorted((float(value) for value in probabilities.values()), reverse=True)
    top_probability = sorted_probs[0] if sorted_probs else float(fallback_confidence)
    second_probability = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    svm_gap = top_probability - second_probability
    return float(np.clip(svm_gap, 0.0, 1.0))


def normalize_shap_cosine_similarity(cosine_similarity: float | None) -> float:
    """Convert cosine similarity from [-1, 1] into reliability in [0, 1]."""
    if cosine_similarity is None or not np.isfinite(cosine_similarity):
        return 0.50
    return float(np.clip((float(cosine_similarity) + 1.0) / 2.0, 0.0, 1.0))


def compute_arima_multiplier(
    forecast_increasing: bool,
    arima_multiplier: float | None = None,
) -> float:
    """ARIMA/SARIMA trend multiplier.

    The forecasting signal is multiplicative rather than another confidence:
    rising future dangerous cases amplify the composite risk, while flat or
    falling forecasts dampen it slightly.
    """
    if arima_multiplier is not None and np.isfinite(arima_multiplier):
        return float(np.clip(arima_multiplier, 0.85, 1.15))
    return 1.10 if forecast_increasing else 0.95


def compute_arima_signal(
    forecast_increasing: bool,
    arima_multiplier: float | None = None,
) -> float:
    """Convert the SARIMA/ARIMA trend into an independent [0, 1] signal.

    This is separate from the legacy multiplier. If a bounded multiplier is
    available, map 0.85..1.15 onto 0..1. Otherwise use conservative defaults:
    rising forecast = stronger risk signal, flat/falling forecast = weaker risk.
    """
    if arima_multiplier is not None and np.isfinite(arima_multiplier):
        bounded = float(np.clip(arima_multiplier, 0.85, 1.15))
        return float(np.clip((bounded - 0.85) / 0.30, 0.0, 1.0))
    return 0.70 if forecast_increasing else 0.40


def _resolve_fuzzy_membership(
    fuzzy_membership: float | dict[str, float] | None,
    predicted_label: str,
) -> float:
    if isinstance(fuzzy_membership, dict):
        return float(np.clip(fuzzy_membership.get(predicted_label, fuzzy_membership.get("predicted", 0.5)), 0.0, 1.0))
    if fuzzy_membership is None or not np.isfinite(fuzzy_membership):
        return 0.50
    return float(np.clip(fuzzy_membership, 0.0, 1.0))


def compute_scores(
    confidence: float,
    shap_importance: pd.DataFrame,
    forecast_increasing: bool,
    label: str,
    prediction_probabilities: dict[str, float] | None = None,
    fuzzy_membership: float | dict[str, float] | None = None,
    shap_cosine_similarity: float | None = None,
    arima_multiplier: float | None = None,
) -> dict[str, float]:
    """Compute a five-signal intelligent composite score.

    final_agentic_score =
        100 * (
            0.30*S
          + 0.20*Gap(SVM)
          + 0.20*mu(predicted)
          + 0.15*SHAP_reliability
          + 0.15*ARIMA_signal
        )

    The old keys are preserved so existing dashboard/report code keeps working.
    """
    severity = float(SEVERITY_WEIGHTS.get(label, 0.50))
    svm_gap = compute_svm_gap(prediction_probabilities, confidence)
    fuzzy_predicted = _resolve_fuzzy_membership(fuzzy_membership, label)
    shap_reliability = normalize_shap_cosine_similarity(shap_cosine_similarity)
    trend_multiplier = compute_arima_multiplier(forecast_increasing, arima_multiplier)
    arima_signal = compute_arima_signal(forecast_increasing, trend_multiplier)

    # SHAP magnitude is retained as a transparent diagnostic, but it is not the
    # reliability signal. Reliability comes from cosine similarity against the
    # saved cluster SHAP mean vector.
    shap_magnitude_score = float(np.clip(shap_importance["abs_importance"].head(5).sum() * 100.0, 0.0, 100.0))

    base_agentic_score = (
        (SCORE_WEIGHTS["severity"] * severity)
        + (SCORE_WEIGHTS["svm_gap"] * svm_gap)
        + (SCORE_WEIGHTS["fuzzy_membership"] * fuzzy_predicted)
        + (SCORE_WEIGHTS["shap_reliability"] * shap_reliability)
        + (SCORE_WEIGHTS["arima_trend"] * arima_signal)
    )
    final_agentic_score = float(np.clip(100.0 * base_agentic_score, 0.0, 100.0))

    return {
        "class_severity_signal": severity,
        "svm_gap": svm_gap,
        "fuzzy_membership": fuzzy_predicted,
        "shap_cosine_similarity": float(shap_cosine_similarity if shap_cosine_similarity is not None else 0.0),
        "shap_reliability_signal": shap_reliability,
        "arima_signal": arima_signal,
        "arima_multiplier": trend_multiplier,
        "score_weight_severity": SCORE_WEIGHTS["severity"],
        "score_weight_svm_gap": SCORE_WEIGHTS["svm_gap"],
        "score_weight_fuzzy_membership": SCORE_WEIGHTS["fuzzy_membership"],
        "score_weight_shap_reliability": SCORE_WEIGHTS["shap_reliability"],
        "score_weight_arima_trend": SCORE_WEIGHTS["arima_trend"],
        "base_agentic_score": float(100.0 * base_agentic_score),
        "final_agentic_score": final_agentic_score,
        "confidence_x_feature_importance": float(confidence * float(shap_importance["abs_importance"].mean())),
        "svm_score": float(100.0 * severity * svm_gap),
        "shap_score": shap_magnitude_score,
        "forecast_score": float(100.0 * arima_signal),
        "total_score": final_agentic_score,
    }
