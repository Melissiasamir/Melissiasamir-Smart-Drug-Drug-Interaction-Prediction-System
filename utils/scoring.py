"""# 10. Scoring System

Score = Confidence x Feature Importance
Total Score = SVM + SHAP + Forecast
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_scores(confidence: float, shap_importance: pd.DataFrame, forecast_increasing: bool, label: str) -> dict[str, float]:
    """Compute transparent SVM, SHAP, forecast, and total scores."""
    label_weight = {"Low Risk": 0.25, "Medium Risk": 0.6, "High Risk": 1.0}.get(label, 0.5)
    svm_score = confidence * label_weight * 100.0
    shap_score = float(np.clip(shap_importance["abs_importance"].head(5).sum() * 100.0, 0, 100))
    forecast_score = 100.0 if forecast_increasing else 35.0
    confidence_feature_score = confidence * float(shap_importance["abs_importance"].mean())
    total_score = float(np.clip((0.45 * svm_score) + (0.35 * shap_score) + (0.20 * forecast_score), 0, 100))
    return {
        "confidence_x_feature_importance": confidence_feature_score,
        "svm_score": float(svm_score),
        "shap_score": shap_score,
        "forecast_score": forecast_score,
        "total_score": total_score,
    }

