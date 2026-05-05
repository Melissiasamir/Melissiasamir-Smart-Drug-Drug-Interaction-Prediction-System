"""# 7. Explainability (SHAP)

SHAP is used for interpretability because it attributes a prediction back to
individual feature contributions. This is essential in healthcare-adjacent AI:
users need a reason, not only a class label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px


def _prediction_class_index(model, row_scaled: pd.DataFrame) -> int:
    """Return the class index selected by the model for one scaled row."""
    return int(np.argmax(model.predict_proba(row_scaled)[0]))


def _single_row_shap_values(shap_values, pred_idx: int, n_features: int) -> np.ndarray:
    """Normalize SHAP outputs across versions to one value per feature."""
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[pred_idx], dtype=float)
        if values.ndim == 2:
            values = values[0]
        return values.reshape(-1)

    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        if values.shape[0] == 1 and values.shape[1] == n_features:
            return values[0, :, pred_idx]
        if values.shape[0] > pred_idx and values.shape[2] == n_features:
            return values[pred_idx, 0, :]
    if values.ndim == 2:
        if values.shape[0] == 1:
            return values[0]
        if values.shape[0] == n_features:
            return values[:, pred_idx] if values.shape[1] > pred_idx else values[:, 0]
    if values.ndim == 1:
        return values

    raise ValueError(f"Unexpected SHAP value shape: {values.shape}")


def _sensitivity_importance(
    model,
    background_scaled: pd.DataFrame,
    row_scaled: pd.DataFrame,
    feature_names: list[str],
    pred_idx: int,
) -> np.ndarray:
    """Estimate feature impact when SHAP cannot provide aligned values."""
    baseline = background_scaled.median().to_frame().T
    base_prob = model.predict_proba(baseline)[0]
    values = []
    for feature in feature_names:
        perturbed = baseline.copy()
        perturbed[feature] = row_scaled.iloc[0][feature]
        values.append(float(model.predict_proba(perturbed)[0][pred_idx] - base_prob[pred_idx]))
    return np.array(values, dtype=float)


def _as_feature_frame(values, feature_names: list[str]) -> pd.DataFrame:
    """Keep feature names attached when SHAP supplies NumPy arrays."""
    if isinstance(values, pd.DataFrame):
        return values.loc[:, feature_names]
    return pd.DataFrame(values, columns=feature_names)


def compute_shap_importance(model, background_scaled: pd.DataFrame, row_scaled: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Compute SHAP values for a single prediction, with a deterministic fallback."""
    feature_names = list(feature_names)
    n_features = len(feature_names)
    pred_idx = _prediction_class_index(model, row_scaled)

    try:
        import shap

        background = shap.sample(background_scaled, min(40, len(background_scaled)), random_state=42)
        def predict_proba_with_names(values):
            return model.predict_proba(_as_feature_frame(values, feature_names))

        explainer = shap.KernelExplainer(predict_proba_with_names, background)
        shap_values = explainer.shap_values(row_scaled, nsamples=100)
        values = _single_row_shap_values(shap_values, pred_idx, n_features)
        if len(values) != n_features:
            raise ValueError(f"Expected {n_features} SHAP values, got {len(values)}")
    except Exception:
        # Fallback keeps the app functional if SHAP cannot initialize locally.
        # It still uses model sensitivity, but production environments should
        # install shap from requirements.txt for true SHAP explanations.
        values = _sensitivity_importance(model, background_scaled, row_scaled, feature_names, pred_idx)

    importance = pd.DataFrame({"feature": feature_names, "shap_value": values})
    importance["abs_importance"] = importance["shap_value"].abs()
    return importance.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def shap_plot(importance: pd.DataFrame):
    """Create a Plotly SHAP feature-importance chart."""
    top = importance.head(10).sort_values("abs_importance", ascending=True)
    fig = px.bar(
        top,
        x="abs_importance",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale=["#22c55e", "#facc15", "#ef4444"],
        title="SHAP Feature Importance",
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig
