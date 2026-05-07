"""
Pydantic schemas for the Smart Drug Risk Agent API.

This module defines request/response models for the FastAPI endpoints,
ensuring type safety and validation for drug interaction predictions.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DrugPairRequest(BaseModel):
    """Request model for drug pair analysis endpoints."""

    drug_a: str = Field(..., description="First drug name", min_length=1)
    drug_b: str = Field(..., description="Second drug name", min_length=1)


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="System status")
    model_loaded: bool = Field(..., description="Whether ML pipeline is loaded")
    shap_memory_available: bool = Field(..., description="SHAP memory store status")
    rag_available: bool = Field(..., description="RAG system availability")
    drift_monitoring_enabled: bool = Field(..., description="Drift detection status")
    pipeline_version: str = Field(..., description="Pipeline version identifier")
    model_type: str = Field(..., description="ML model type")
    startup_timestamp: str = Field(..., description="API startup timestamp")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    timestamp: str = Field(..., description="Health check timestamp")


class PredictionResponse(BaseModel):
    """Response model for full prediction endpoint."""

    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")
    prediction_label: str = Field(..., description="Predicted risk level")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Risk level probabilities")
    scores: Dict[str, Any] = Field(..., description="Intelligent scoring results")
    shap_reliability: Dict[str, Any] = Field(..., description="SHAP explanation reliability")
    fuzzy_membership: Optional[float] = Field(None, description="Fuzzy clustering membership")
    rag_results: Dict[str, Any] = Field(..., description="RAG retrieval results")
    react_decision: Dict[str, Any] = Field(..., description="ReAct reasoning decision")
    final_decision: str = Field(..., description="Final system decision")
    reasoning_trace: List[Dict[str, Any]] = Field(..., description="Complete reasoning trace")


class ReactResponse(BaseModel):
    """Response model for ReAct-only endpoint."""

    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")
    route_taken: str = Field(..., description="Decision route selected")
    action_confidence: float = Field(..., description="Confidence in action (0-1)")
    decision_explanation: str = Field(..., description="Human-readable explanation")
    human_review_required: bool = Field(..., description="Whether human review is needed")
    reasoning_trace: List[Dict[str, Any]] = Field(..., description="ReAct reasoning trace")
    final_action: str = Field(..., description="Final recommended action")


class DriftStatusResponse(BaseModel):
    """Response model for drift monitoring status."""

    drift_triggered: bool = Field(..., description="Whether drift has been detected")
    centroid_drift: float = Field(..., description="Centroid drift magnitude")
    svm_gap_degradation: float = Field(..., description="SVM gap degradation signal")
    retraining_gate_status: str = Field(..., description="Retraining gate state")
    rollback_protection_status: str = Field(..., description="Rollback protection availability")


class AuditLogEntry(BaseModel):
    """Model for API audit logging."""

    timestamp: str = Field(..., description="Log entry timestamp")
    endpoint: str = Field(..., description="API endpoint called")
    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")
    route_taken: str = Field(..., description="Decision route selected")
    final_decision: str = Field(..., description="Final system decision")
    prediction_label: str = Field(..., description="Predicted risk label")
    confidence: float = Field(..., description="Prediction confidence")
    scores: Dict[str, Any] = Field(..., description="Scoring results")
    processing_time_ms: float = Field(..., description="Request processing time in milliseconds")
    response_status: int = Field(..., description="HTTP response status code")
    human_review_required: bool = Field(..., description="Whether human review is required")
    drift_status: Dict[str, Any] = Field(..., description="Drift monitoring status")