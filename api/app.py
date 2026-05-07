"""
FastAPI wrapper for Smart Drug Risk Agent.

This module provides a REST API interface to the existing drug interaction
prediction pipeline, reusing all existing ML components without modification.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.schemas import (
    AuditLogEntry,
    DriftStatusResponse,
    DrugPairRequest,
    HealthResponse,
    PredictionResponse,
    ReactResponse,
)
from utils.pipeline import PipelineArtifacts, infer_pair, load_or_train_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
AUDIT_LOG_PATH = LOGS_DIR / "api_audit.jsonl"

# Global pipeline artifacts (loaded once on startup)
pipeline_artifacts: PipelineArtifacts | None = None
startup_timestamp: datetime | None = None


def load_pipeline() -> None:
    """Load ML pipeline artifacts on application startup."""
    global pipeline_artifacts
    try:
        logger.info("Loading ML pipeline artifacts...")
        pipeline_artifacts = load_or_train_pipeline()
        logger.info("Pipeline artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline artifacts: {e}")
        raise


def log_audit_entry(entry: AuditLogEntry) -> None:
    """Log API request to JSONL audit file."""
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")


def check_system_health() -> Dict[str, Any]:
    """Check availability of all system components."""
    if pipeline_artifacts is None:
        return {
            "model_loaded": False,
            "shap_memory_available": False,
            "rag_available": False,
            "drift_monitoring_enabled": False,
        }

    return {
        "model_loaded": pipeline_artifacts.classifier.model is not None,
        "shap_memory_available": hasattr(pipeline_artifacts, 'shap_memory') and pipeline_artifacts.shap_memory is not None,
        "rag_available": hasattr(pipeline_artifacts, 'rag_index') and pipeline_artifacts.rag_index is not None,
        "drift_monitoring_enabled": hasattr(pipeline_artifacts, 'drift_status') and pipeline_artifacts.drift_status is not None,
    }


# Create FastAPI application
app = FastAPI(
    title="Smart Drug Risk Agent API",
    description="""
    REST API for drug-drug interaction risk prediction using advanced ML pipeline.

    This API provides endpoints for predicting drug interaction risks using a comprehensive
    machine learning pipeline that includes SVM classification, fuzzy clustering, SHAP explanations,
    RAG retrieval, and ReAct reasoning for decision routing.

    ## Features
    - Full drug interaction prediction with all ML components
    - ReAct reasoning engine for decision routing
    - Drift monitoring and model health checks
    - Comprehensive audit logging
    - Production-ready error handling
    """,
    version="1.0.0",
    openapi_tags=[
        {
            "name": "predictions",
            "description": "Drug interaction prediction endpoints",
        },
        {
            "name": "monitoring",
            "description": "System health and monitoring endpoints",
        },
    ],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with clean JSON response."""
    logger.warning(f"Validation error for {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid request data",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors with clean JSON response."""
    logger.error(f"Unexpected error for {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
        },
    )


@app.on_event("startup")
async def startup_event():
    """Load pipeline artifacts during application startup."""
    global pipeline_artifacts, startup_timestamp
    startup_timestamp = datetime.now()
    logger.info("API server starting up...")
    load_pipeline()


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check() -> HealthResponse:
    """
    Check system health and component availability.

    Returns comprehensive health status including:
    - ML pipeline loading status
    - Component availability (SHAP, RAG, drift monitoring)
    - System uptime and version information

    **Example Response:**
    ```json
    {
      "status": "healthy",
      "model_loaded": true,
      "shap_memory_available": true,
      "rag_available": true,
      "drift_monitoring_enabled": true,
      "pipeline_version": "3.0",
      "model_type": "SVM + FCM Clustering + SARIMA Forecasting",
      "startup_timestamp": "2024-01-01T12:00:00",
      "uptime_seconds": 3600.0,
      "timestamp": "2024-01-01T13:00:00"
    }
    ```
    """
    health_status = check_system_health()
    current_time = datetime.now()

    # Determine overall status
    all_components_ok = all(health_status.values())
    status = "healthy" if all_components_ok else "degraded"

    # Calculate uptime
    uptime_seconds = (current_time - startup_timestamp).total_seconds() if startup_timestamp else 0.0

    return HealthResponse(
        status=status,
        model_loaded=health_status["model_loaded"],
        shap_memory_available=health_status["shap_memory_available"],
        rag_available=health_status["rag_available"],
        drift_monitoring_enabled=health_status["drift_monitoring_enabled"],
        pipeline_version="3.0",  # From PIPELINE_CACHE_VERSION
        model_type="SVM + FCM Clustering + SARIMA Forecasting",
        startup_timestamp=startup_timestamp.isoformat() if startup_timestamp else "",
        uptime_seconds=uptime_seconds,
        timestamp=current_time.isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
async def predict_drug_interaction(request: DrugPairRequest) -> PredictionResponse:
    """
    Run full drug interaction prediction with all ML components.

    This endpoint performs comprehensive drug interaction analysis using:
    - SVM classification for risk prediction
    - Fuzzy clustering for uncertainty quantification
    - SHAP explanations for model interpretability
    - RAG retrieval for similar case analysis
    - ReAct reasoning for decision routing

    **Example Request:**
    ```json
    {
      "drug_a": "aspirin",
      "drug_b": "warfarin"
    }
    ```

    **Example Response:**
    ```json
    {
      "drug_a": "aspirin",
      "drug_b": "warfarin",
      "prediction_label": "High Risk",
      "confidence": 0.89,
      "probabilities": {"High Risk": 0.89, "Moderate Risk": 0.11},
      "scores": {...},
      "shap_reliability": {...},
      "fuzzy_membership": 0.95,
      "rag_results": {...},
      "react_decision": {...},
      "final_decision": "High Risk - Requires Review",
      "reasoning_trace": [...]
    }
    ```
    """
    if pipeline_artifacts is None:
        raise HTTPException(status_code=503, detail="ML pipeline not loaded")

    start_time = datetime.now()
    try:
        # Run full inference pipeline
        _, _, prediction = infer_pair(
            pipeline_artifacts,
            request.drug_a,
            request.drug_b,
            run_agentic_decision=True
        )

        # Extract results from prediction dict
        response = PredictionResponse(
            drug_a=request.drug_a,
            drug_b=request.drug_b,
            prediction_label=str(prediction.get("prediction_label", prediction.get("label", "Unknown"))),
            confidence=float(prediction.get("confidence", 0.0)),
            probabilities=prediction.get("probabilities", {}),
            scores=prediction.get("scores", {}),
            shap_reliability=prediction.get("shap_reliability", {}),
            fuzzy_membership=prediction.get("fuzzy_membership"),
            rag_results=prediction.get("rag_result", {}),
            react_decision=prediction.get("react_decision", {}),
            final_decision=str(prediction.get("final_decision", "Unknown")),
            reasoning_trace=prediction.get("react_decision", {}).get("trace", []),
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log audit entry
        audit_entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            endpoint="/predict",
            drug_a=request.drug_a,
            drug_b=request.drug_b,
            route_taken=response.react_decision.get("route", "unknown"),
            final_decision=response.final_decision,
            prediction_label=response.prediction_label,
            confidence=response.confidence,
            scores=response.scores,
            processing_time_ms=processing_time,
            response_status=200,
            human_review_required=bool(response.react_decision.get("human_review_required", False)),
            drift_status={"available": check_system_health()["drift_monitoring_enabled"]},
        )
        log_audit_entry(audit_entry)

        return response

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Prediction failed for {request.drug_a} vs {request.drug_b}: {e}")

        # Log failed request
        audit_entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            endpoint="/predict",
            drug_a=request.drug_a,
            drug_b=request.drug_b,
            route_taken="error",
            final_decision="error",
            prediction_label="error",
            confidence=0.0,
            scores={},
            processing_time_ms=processing_time,
            response_status=500,
            human_review_required=True,
            drift_status={"available": check_system_health()["drift_monitoring_enabled"]},
        )
        log_audit_entry(audit_entry)

        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/react", response_model=ReactResponse, tags=["predictions"])
async def react_decision_only(request: DrugPairRequest) -> ReactResponse:
    """
    Run only the ReAct reasoning engine for decision routing.

    This endpoint focuses on the reasoning and decision-making component,
    providing fast analysis for decision routing without full ML pipeline execution.

    **Example Request:**
    ```json
    {
      "drug_a": "aspirin",
      "drug_b": "warfarin"
    }
    ```

    **Example Response:**
    ```json
    {
      "drug_a": "aspirin",
      "drug_b": "warfarin",
      "route_taken": "high_risk_path",
      "action_confidence": 0.92,
      "decision_explanation": "High risk interaction detected",
      "human_review_required": true,
      "reasoning_trace": [...],
      "final_action": "escalate_to_pharmacist"
    }
    ```
    """
    if pipeline_artifacts is None:
        raise HTTPException(status_code=503, detail="ML pipeline not loaded")

    start_time = datetime.now()
    try:
        # Run inference but focus only on ReAct decision
        _, _, prediction = infer_pair(
            pipeline_artifacts,
            request.drug_a,
            request.drug_b,
            run_agentic_decision=True
        )

        react_decision = prediction.get("react_decision", {})

        response = ReactResponse(
            drug_a=request.drug_a,
            drug_b=request.drug_b,
            route_taken=str(react_decision.get("route", "unknown")),
            action_confidence=float(react_decision.get("action_confidence", 0.0)),
            decision_explanation=str(react_decision.get("decision_explanation", "")),
            human_review_required=bool(react_decision.get("human_review_required", False)),
            reasoning_trace=react_decision.get("trace", []),
            final_action=str(react_decision.get("final_decision", "unknown")),
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log audit entry
        audit_entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            endpoint="/react",
            drug_a=request.drug_a,
            drug_b=request.drug_b,
            route_taken=response.route_taken,
            final_decision=response.final_action,
            prediction_label=response.final_action,
            confidence=response.action_confidence,
            scores={"action_confidence": response.action_confidence},
            processing_time_ms=processing_time,
            response_status=200,
            human_review_required=response.human_review_required,
            drift_status={"available": check_system_health()["drift_monitoring_enabled"]},
        )
        log_audit_entry(audit_entry)

        return response

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"ReAct decision failed for {request.drug_a} vs {request.drug_b}: {e}")

        # Log failed request
        audit_entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            endpoint="/react",
            drug_a=request.drug_a,
            drug_b=request.drug_b,
            route_taken="error",
            final_decision="error",
            prediction_label="error",
            confidence=0.0,
            scores={},
            processing_time_ms=processing_time,
            response_status=500,
            human_review_required=True,
            drift_status={"available": check_system_health()["drift_monitoring_enabled"]},
        )
        log_audit_entry(audit_entry)

        raise HTTPException(status_code=500, detail="ReAct decision failed")


@app.get("/drift/status", response_model=DriftStatusResponse, tags=["monitoring"])
async def get_drift_status() -> DriftStatusResponse:
    """
    Get current drift monitoring status.

    Returns information about model drift detection including:
    - Whether drift has been triggered
    - Centroid drift magnitude
    - SVM gap degradation signals
    - Retraining gate status

    **Example Response:**
    ```json
    {
      "drift_triggered": false,
      "centroid_drift": 0.05,
      "svm_gap_degradation": 0.02,
      "retraining_gate_status": "stable",
      "rollback_protection_status": "available"
    }
    ```
    """
    if pipeline_artifacts is None:
        raise HTTPException(status_code=503, detail="ML pipeline not loaded")

    try:
        drift_status = getattr(pipeline_artifacts, 'drift_status', {}) or {}

        return DriftStatusResponse(
            drift_triggered=bool(drift_status.get("triggered", False)),
            centroid_drift=float(drift_status.get("centroid_signal", {}).get("centroid_drift", 0.0)),
            svm_gap_degradation=float(drift_status.get("svm_gap_signal", {}).get("drop", 0.0)),
            retraining_gate_status=str(drift_status.get("retraining_gate", "unknown")),
            rollback_protection_status=str(drift_status.get("rollback_protection", "available")),
        )

    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)