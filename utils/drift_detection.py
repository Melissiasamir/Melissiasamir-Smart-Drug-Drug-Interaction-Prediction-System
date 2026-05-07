"""Drift detection and rollback helpers for self-learning previews."""

from __future__ import annotations

from logging import config
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class DriftConfig:
    """Drift gates tuned to ignore single-sample noise."""

    min_gap_drop: float = 0.12
    centroid_movement_threshold: float = 0.35
    rolling_window: int = 20
    min_recent_points: int = 5
    min_f1_delta: float = -0.02


def svm_gap_degradation(
    baseline_gap: float | None,
    recent_gaps: list[float],
    min_drop: float = 0.12,
) -> dict[str, Any]:
    """Detect degradation in SVM confidence separation.

    degradation = baseline_mean_gap - recent_mean_gap
    """
    clean_gaps = [float(gap) for gap in recent_gaps if gap is not None and np.isfinite(gap)]
    if baseline_gap is None or not clean_gaps:
        return {"triggered": False, "baseline_gap": baseline_gap, "recent_gap": None, "drop": 0.0}

    recent_gap = float(np.mean(clean_gaps))
    drop = float(baseline_gap - recent_gap)
    return {
        "triggered": bool(drop >= min_drop),
        "baseline_gap": float(baseline_gap),
        "recent_gap": recent_gap,
        "drop": drop,
        "threshold": float(min_drop),
    }


def rolling_svm_gap_degradation(
    baseline_gap: float | None,
    recent_gaps: list[float],
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    """Track degradation over a rolling window to avoid isolated noise."""
    config = config or DriftConfig()
    clean_gaps = [float(gap) for gap in recent_gaps if gap is not None and np.isfinite(gap)]
    window_gaps = clean_gaps[-config.rolling_window :]
    if len(window_gaps) < config.min_recent_points:
        return {
            "triggered": False,
            "baseline_gap": baseline_gap,
            "recent_gap": float(np.mean(window_gaps)) if window_gaps else None,
            "drop": 0.0,
            "points": len(window_gaps),
            "reason": "not_enough_recent_gap_points",
            "threshold": config.min_gap_drop,
        }
    signal = svm_gap_degradation(baseline_gap, window_gaps, config.min_gap_drop)
    signal["points"] = len(window_gaps)
    signal["window"] = config.rolling_window
    signal["reason"] = "gap_degraded_over_rolling_window" if signal["triggered"] else "gap_stable"
    return signal


def centroid_movement(
    old_metadata: dict[str, Any] | None,
    new_metadata: dict[str, Any] | None,
    movement_threshold: float = 0.35,
) -> dict[str, Any]:
    """Compute centroid drift between old and new FCM centers.

    centroid_drift = mean_i(||old_center_i - new_center_i||_2)
    """
    if not old_metadata or not new_metadata:
        return {"triggered": False, "centroid_drift": 0.0, "threshold": float(movement_threshold)}

    old_centers = np.asarray(old_metadata.get("fcm_centers"), dtype=float)
    new_centers = np.asarray(new_metadata.get("fcm_centers"), dtype=float)
    if old_centers.ndim != 2 or new_centers.ndim != 2 or old_centers.size == 0 or new_centers.size == 0:
        return {"triggered": False, "centroid_drift": 0.0, "threshold": float(movement_threshold)}

    n = min(len(old_centers), len(new_centers))
    distance_matrix = np.linalg.norm(old_centers[:n, None, :] - new_centers[None, :n, :], axis=2)
    matched_distances = []
    used_new: set[int] = set()
    for old_idx in range(n):
        available = [(distance_matrix[old_idx, new_idx], new_idx) for new_idx in range(n) if new_idx not in used_new]
        distance, new_idx = min(available, key=lambda item: item[0])
        matched_distances.append(float(distance))
        used_new.add(int(new_idx))
    drift = float(np.mean(matched_distances))
    return {"triggered": bool(drift >= movement_threshold), "centroid_drift": drift, "threshold": float(movement_threshold)}


def should_trigger_retraining(
    gap_signal: dict[str, Any],
    centroid_signal: dict[str, Any],
) -> dict[str, Any]:
    """Both signals must trigger together before retraining is allowed."""
    triggered = bool(gap_signal.get("triggered") and centroid_signal.get("triggered"))
    return {
        "triggered": triggered,
        "reason": "svm_gap_degradation_and_centroid_movement" if triggered else "waiting_for_both_drift_signals",
        "svm_gap_signal": gap_signal,
        "centroid_signal": centroid_signal,
    }


def build_drift_metadata(
    decision: dict[str, Any],
    old_clustering_metadata: dict[str, Any] | None,
    new_clustering_metadata: dict[str, Any] | None,
    config: DriftConfig,
) -> dict[str, Any]:
    """Metadata worth persisting with a promoted or previewed model."""
    return {
        "created_at": datetime.now().isoformat(),
        "config": config.__dict__,
        "decision": decision,
        "old_fcm_centers": None if old_clustering_metadata is None else old_clustering_metadata.get("fcm_centers"),
        "new_fcm_centers": None if new_clustering_metadata is None else new_clustering_metadata.get("fcm_centers"),
        "rollback_policy": "backup current artifact, validate new model, restore backup if validation fails",
    }


def compare_model_quality(
    old_model: Any,
    new_model: Any,
    validation_x: Any,
    validation_y: Any,
    min_f1_delta: float = -0.02,
) -> dict[str, Any]:
    """Approve a new model only if validation quality is not meaningfully worse."""
    old_pred = old_model.predict(validation_x)
    new_pred = new_model.predict(validation_x)
    old_f1 = float(f1_score(validation_y, old_pred, average="weighted", zero_division=0))
    new_f1 = float(f1_score(validation_y, new_pred, average="weighted", zero_division=0))
    delta = new_f1 - old_f1
    return {
        "approved": bool(delta >= min_f1_delta),
        "old_f1": old_f1,
        "new_f1": new_f1,
        "delta": float(delta),
        "min_delta": float(min_f1_delta),
    }


def validate_retraining_candidate(
    old_model: Any,
    new_model: Any,
    validation_x: Any,
    validation_y: Any,
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    """Rollback gate for model promotion."""
    config = config or DriftConfig()
    quality = compare_model_quality(old_model, new_model, validation_x, validation_y, config.min_f1_delta)
    quality["rollback_required"] = not quality["approved"]
    quality["reason"] = "quality_accepted" if quality["approved"] else "new_model_quality_below_rollback_threshold"
    return quality


def backup_artifact(path: Path) -> Path | None:
    """Create a timestamped rollback copy before model promotion."""
    if not path.exists():
        return None
    backup_path = path.with_suffix(path.suffix + f".rollback-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(path, backup_path)
    return backup_path


def rollback_artifact(backup_path: Path | None, destination_path: Path) -> bool:
    """Restore the previous artifact if a promotion fails validation."""
    if backup_path is None or not backup_path.exists():
        return False
    shutil.copy2(backup_path, destination_path)
    return True


def promote_with_rollback(
    old_model: Any,
    new_model: Any,
    validation_x: Any,
    validation_y: Any,
    artifact_payload: Any,
    artifact_path: Path,
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    """Persist a candidate artifact only after validation, with safe restore."""
    config = config or DriftConfig()
    backup_path = backup_artifact(artifact_path)
    quality = validate_retraining_candidate(old_model, new_model, validation_x, validation_y, config)
    if not quality["approved"]:
        restored = rollback_artifact(backup_path, artifact_path)
        return {"promoted": False, "quality": quality, "backup_path": str(backup_path) if backup_path else None, "rolled_back": restored}

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_payload, artifact_path)
    return {"promoted": True, "quality": quality, "backup_path": str(backup_path) if backup_path else None, "rolled_back": False}


def evaluate_self_learning_drift(
    baseline_gap: float | None,    
    recent_gaps: list[float],
    old_clustering_metadata: dict[str, Any] | None,
    new_clustering_metadata: dict[str, Any] | None,
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    """Single entry point for self-learning previews."""
    config = config or DriftConfig()
    gap_signal = rolling_svm_gap_degradation(baseline_gap, recent_gaps, config)    
    centroid_signal = centroid_movement(old_clustering_metadata, new_clustering_metadata, config.centroid_movement_threshold)
    decision = should_trigger_retraining(gap_signal, centroid_signal)
    decision["rollback_protection"] = "enabled: backup old artifact, validate new model, rollback if quality drops"
    decision["metadata_to_persist"] = build_drift_metadata(
        decision,
        old_clustering_metadata,
        new_clustering_metadata,
        config,
    )
    return decision
