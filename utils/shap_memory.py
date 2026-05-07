"""Persistent SHAP cluster memory.

The memory stores mean SHAP vectors per predicted cluster/label. Inference can
compare the current SHAP vector with the saved cluster mean using cosine
similarity, which becomes the SHAP reliability signal in the composite score.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


SHAP_MEMORY_VERSION = 1


@dataclass(frozen=True)
class ReliabilityThresholds:
    """Dashboard-friendly thresholds for normalized reliability in [0, 1]."""

    high: float = 0.80
    medium: float = 0.60


def shap_importance_to_vector(shap_importance: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Convert a sorted SHAP importance table back into feature-column order."""
    lookup = dict(zip(shap_importance["feature"], shap_importance["shap_value"]))
    return np.array([float(lookup.get(feature, 0.0)) for feature in feature_names], dtype=float)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """cos(theta) = (a dot b) / (||a|| * ||b||)."""
    a = np.asarray(vector_a, dtype=float).reshape(-1)
    b = np.asarray(vector_b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def normalized_reliability(cosine_value: float | None) -> float:
    """Map cosine similarity from [-1, 1] to a dashboard signal in [0, 1]."""
    if cosine_value is None or not np.isfinite(cosine_value):
        return 0.0
    return float(np.clip((float(cosine_value) + 1.0) / 2.0, 0.0, 1.0))


def reliability_level(reliability_signal: float, thresholds: ReliabilityThresholds | None = None) -> str:
    """Classify SHAP memory alignment as high, medium, or low."""
    thresholds = thresholds or ReliabilityThresholds()
    if reliability_signal >= thresholds.high:
        return "high"
    if reliability_signal >= thresholds.medium:
        return "medium"
    return "low"


def _sensitivity_vectors(model: Any, background_scaled: pd.DataFrame, rows_scaled: pd.DataFrame) -> np.ndarray:
    """Deterministic SHAP fallback based on class-probability perturbation."""
    feature_names = list(rows_scaled.columns)
    baseline = background_scaled[feature_names].median().to_frame().T
    base_prob = model.predict_proba(baseline)[0]
    pred_indices = np.argmax(model.predict_proba(rows_scaled), axis=1)
    vectors = []
    for _, row in rows_scaled.iterrows():
        values = []
        for feature in feature_names:
            perturbed = baseline.copy()
            perturbed[feature] = row[feature]
            probabilities = model.predict_proba(perturbed)[0]
            pred_idx = int(pred_indices[len(vectors)])
            values.append(float(probabilities[pred_idx] - base_prob[pred_idx]))
        vectors.append(values)
    return np.asarray(vectors, dtype=float)


def _normalize_batch_shap_values(shap_values: Any, pred_indices: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(shap_values, dtype=object if isinstance(shap_values, list) else float)
    vectors = []
    if isinstance(shap_values, list):
        for row_idx, pred_idx in enumerate(pred_indices):
            vectors.append(np.asarray(shap_values[int(pred_idx)][row_idx], dtype=float).reshape(-1))
        return np.asarray(vectors, dtype=float)

    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3 and values.shape[1] == n_features:
        for row_idx, pred_idx in enumerate(pred_indices):
            vectors.append(values[row_idx, :, int(pred_idx)])
        return np.asarray(vectors, dtype=float)
    if values.ndim == 3 and values.shape[2] == n_features:
        for row_idx, pred_idx in enumerate(pred_indices):
            vectors.append(values[int(pred_idx), row_idx, :])
        return np.asarray(vectors, dtype=float)
    if values.ndim == 2 and values.shape[1] == n_features:
        return values
    raise ValueError(f"Unexpected batch SHAP value shape: {values.shape}")


def compute_training_shap_vectors(
    model: Any,
    background_scaled: pd.DataFrame,
    rows_scaled: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    """Compute SHAP vectors for training memory, with a deterministic fallback."""
    feature_names = list(rows_scaled.columns)
    try:
        import shap

        background = shap.sample(background_scaled[feature_names], min(40, len(background_scaled)), random_state=42)

        def predict_proba_with_names(values):
            frame = pd.DataFrame(values, columns=feature_names)
            return model.predict_proba(frame)

        explainer = shap.KernelExplainer(predict_proba_with_names, background)
        shap_values = explainer.shap_values(rows_scaled[feature_names], nsamples=80)
        pred_indices = np.argmax(model.predict_proba(rows_scaled[feature_names]), axis=1)
        return _normalize_batch_shap_values(shap_values, pred_indices, len(feature_names)), "shap_kernel"
    except Exception:
        return _sensitivity_vectors(model, background_scaled[feature_names], rows_scaled[feature_names]), "sensitivity_fallback"


def build_shap_cluster_memory(
    model: Any,
    background_scaled: pd.DataFrame,
    scaled_features: pd.DataFrame,
    pseudo_labels: pd.DataFrame,
    feature_names: list[str],
    path: Path,
    max_rows: int = 120,
) -> dict[str, Any]:
    """Compute, group, and persist cluster mean SHAP vectors."""
    n_rows = min(max_rows, len(scaled_features))
    sampled = scaled_features[feature_names].sample(n_rows, random_state=42) if len(scaled_features) > n_rows else scaled_features[feature_names].copy()
    labels = pseudo_labels.loc[sampled.index, "pseudo_label"].astype(str)
    fcm_clusters = pseudo_labels.loc[sampled.index, "fcm_cluster"].astype(int)
    shap_vectors, vector_source = compute_training_shap_vectors(model, background_scaled[feature_names], sampled)

    cluster_mean_vectors: dict[str, list[float]] = {}
    cluster_counts: dict[str, int] = {}
    fcm_mean_vectors: dict[str, list[float]] = {}
    for label in sorted(labels.unique()):
        mask = labels.values == label
        cluster_mean_vectors[label] = shap_vectors[mask].mean(axis=0).astype(float).tolist()
        cluster_counts[label] = int(mask.sum())
    for cluster_id in sorted(fcm_clusters.unique()):
        mask = fcm_clusters.values == cluster_id
        fcm_mean_vectors[str(cluster_id)] = shap_vectors[mask].mean(axis=0).astype(float).tolist()

    payload = {
        "version": SHAP_MEMORY_VERSION,
        "created_at": datetime.now().isoformat(),
        "feature_names": list(feature_names),
        "cluster_mean_vectors": cluster_mean_vectors,
        "fcm_mean_vectors": fcm_mean_vectors,
        "cluster_counts": cluster_counts,
        "reliability_thresholds": ReliabilityThresholds().__dict__,
        "sample_count": int(len(sampled)),
        "vector_source": vector_source,
        "case_indices": sampled.index.astype(int).tolist(),
        "case_feature_vectors": sampled.to_numpy(dtype=float),
        "case_shap_vectors": shap_vectors,
        "case_labels": labels.tolist(),
    }
    save_shap_memory(payload, path)
    return payload


def save_shap_memory(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_shap_memory(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("version") != SHAP_MEMORY_VERSION:
        return None
    return payload


def shap_cluster_reliability(
    shap_vector: np.ndarray,
    predicted_label: str,
    memory: dict[str, Any] | None,
    high_threshold: float | None = None,
    medium_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare a new SHAP vector to saved cluster mean vectors."""
    saved_thresholds = (memory or {}).get("reliability_thresholds") or {}
    thresholds = ReliabilityThresholds(
        high=float(high_threshold if high_threshold is not None else saved_thresholds.get("high", 0.80)),
        medium=float(medium_threshold if medium_threshold is not None else saved_thresholds.get("medium", 0.60)),
    )
    if not memory:
        return {
            "cosine_similarity": 0.0,
            "reliability_signal": 0.0,
            "reliability_level": "low",
            "matched_cluster": None,
            "available": False,
            "thresholds": thresholds.__dict__,
            "explanation": "No SHAP memory is available for reliability comparison.",
        }

    means = memory.get("cluster_mean_vectors") or {}
    if predicted_label in means:
        matched_label = predicted_label
    elif means:
        matched_label = max(means, key=lambda label: cosine_similarity(shap_vector, np.asarray(means[label], dtype=float)))
    else:
        return {
            "cosine_similarity": 0.0,
            "reliability_signal": 0.0,
            "reliability_level": "low",
            "matched_cluster": None,
            "available": False,
            "thresholds": thresholds.__dict__,
            "explanation": "SHAP memory contains no cluster mean vectors.",
        }

    similarity = cosine_similarity(shap_vector, np.asarray(means[matched_label], dtype=float))
    signal = normalized_reliability(similarity)
    level = reliability_level(signal, thresholds)
    return {
        "cosine_similarity": similarity,
        "reliability_signal": signal,
        "reliability_level": level,
        "matched_cluster": matched_label,
        "available": True,
        "thresholds": thresholds.__dict__,
        "explanation": f"Current SHAP vector has {level} alignment with the {matched_label} cluster memory.",
    }
