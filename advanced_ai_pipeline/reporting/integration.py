"""Wire doctor submission to reporting, persistence, and optional HIGH-risk user email."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from advanced_ai_pipeline.doctor_pipeline.doctor_handler import DoctorPipeline

from utils.explainability import compute_shap_importance
from utils.pipeline import infer_pair, load_or_train_pipeline
from utils.scoring import compute_scores

from .email_service import send_email
from .report_generator import generate_report
from .user_profile import get_user_email

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORT_PATH = _PROJECT_ROOT / "data" / "last_clinical_report.json"


def report_storage_path() -> Path:
    return _REPORT_PATH


def load_persisted_report() -> dict[str, Any] | None:
    if not _REPORT_PATH.exists():
        return None
    try:
        raw = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def persist_report(report: dict[str, Any]) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")


def build_context(
    drug_1: str,
    drug_2: str,
    description: str,
    artifacts: Any,
    prediction: dict[str, Any],
    shap_importance: pd.DataFrame,
    scores: dict[str, float],
    advanced_ai: dict[str, Any],
    *,
    force_high_risk: bool = False,
) -> dict[str, Any]:
    shap_records = shap_importance.to_dict("records")
    return {
        "drug_1": drug_1,
        "drug_2": drug_2,
        "description": description,
        "svm": {
            "label": prediction["label"],
            "confidence": float(prediction["confidence"]),
            "probabilities": {str(k): float(v) for k, v in (prediction.get("probabilities") or {}).items()},
        },
        "shap": shap_records,
        "scores": {str(k): float(v) for k, v in scores.items()},
        "forecast_increasing": bool(artifacts.forecast_increasing),
        "production_clustering": dict(artifacts.clustering_metadata),
        "advanced_ai": advanced_ai,
        "force_high_risk": force_high_risk,
    }


def _advanced_ai_snapshot(drug_1: str, drug_2: str) -> dict[str, Any]:
    """Run embedding cluster + similarity path without persisting another doctor Excel row."""
    pipeline = DoctorPipeline()
    pipeline.update_graph(drug_1, drug_2)
    drug1_info = pipeline.process_drug(drug_1)
    drug2_info = pipeline.process_drug(drug_2)
    drug1_similar = pipeline.get_similar_drugs(drug_1)
    drug2_similar = pipeline.get_similar_drugs(drug_2)
    return {
        "drug1": drug1_info["name"],
        "drug2": drug2_info["name"],
        "drug1_cluster": int(drug1_info["cluster"]),
        "drug2_cluster": int(drug2_info["cluster"]),
        "similar_drugs_1": drug1_similar,
        "similar_drugs_2": drug2_similar,
        "status": "processed",
    }


def process_doctor_submission(drug_1: str, drug_2: str, description: str) -> dict[str, Any]:
    """
    After the doctor record is saved to Excel, run SVM/SHAP/SARIMA signals plus
    advanced similarity/cluster snapshot, build the report, persist it, and send
    user email on HIGH risk when an email is on file.
    """
    force_high = os.environ.get("DDI_SIMULATE_HIGH_RISK", "").strip() == "1"

    artifacts = load_or_train_pipeline()
    _, row_scaled, prediction = infer_pair(artifacts, drug_1, drug_2)
    shap_importance = compute_shap_importance(
        artifacts.classifier.model,
        artifacts.scaled_features,
        row_scaled,
        artifacts.feature_columns,
    )
    scores = compute_scores(
        float(prediction["confidence"]),
        shap_importance,
        bool(artifacts.forecast_increasing),
        str(prediction["label"]),
    )
    advanced = _advanced_ai_snapshot(drug_1, drug_2)

    context = build_context(
        drug_1,
        drug_2,
        description,
        artifacts,
        prediction,
        shap_importance,
        scores,
        advanced,
        force_high_risk=force_high,
    )
    report = generate_report(context)
    persist_report(report)

    email = get_user_email()
    if report.get("risk_level") == "HIGH" and email:
        send_email(report, email)
    elif report.get("risk_level") == "HIGH" and not email:
        print("High risk - no saved user email; alert not sent")
    else:
        print("Low risk - no email sent")

    return report


def build_pair_report(
    drug_1: str,
    drug_2: str,
    artifacts: Any,
    prediction: dict[str, Any],
    shap_importance: pd.DataFrame,
    scores: dict[str, float],
) -> dict[str, Any]:
    """Build the same structured report used elsewhere, for User → Analyze Risk (no Excel write)."""
    advanced = _advanced_ai_snapshot(drug_1, drug_2)
    context = build_context(
        drug_1,
        drug_2,
        "",
        artifacts,
        prediction,
        shap_importance,
        scores,
        advanced,
        force_high_risk=False,
    )
    return generate_report(context)
