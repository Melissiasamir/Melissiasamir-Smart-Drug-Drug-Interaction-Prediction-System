# Smart Drug Risk Agent API

Lightweight FastAPI wrapper for the Smart Drug Risk Agent ML pipeline.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python api/app.py

# Or with uvicorn directly
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will automatically load the ML pipeline artifacts on startup.

## Endpoints

### GET /health
Check system status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "shap_memory_available": true,
  "rag_available": true,
  "drift_monitoring_enabled": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /predict
Run full drug interaction prediction with all components.

**Request:**
```json
{
  "drug_a": "Aspirin",
  "drug_b": "Warfarin"
}
```

**Response:**
```json
{
  "drug_a": "Aspirin",
  "drug_b": "Warfarin",
  "prediction_label": "High Risk",
  "confidence": 0.87,
  "probabilities": {
    "Low Risk": 0.05,
    "Medium Risk": 0.08,
    "High Risk": 0.87
  },
  "scores": {...},
  "shap_reliability": {...},
  "fuzzy_membership": 0.92,
  "rag_results": {...},
  "react_decision": {...},
  "final_decision": "High Risk - Immediate Review",
  "reasoning_trace": [...]
}
```

### POST /react
Run only the ReAct reasoning engine for decision routing.

**Request:**
```json
{
  "drug_a": "Aspirin",
  "drug_b": "Warfarin"
}
```

**Response:**
```json
{
  "drug_a": "Aspirin",
  "drug_b": "Warfarin",
  "route_taken": "high_confidence_path",
  "action_confidence": 0.89,
  "decision_explanation": "High confidence prediction with increasing trend",
  "human_review_required": false,
  "reasoning_trace": [...],
  "final_action": "Flag for pharmacist review"
}
```

### GET /drift/status
Get current drift monitoring status.

**Response:**
```json
{
  "drift_triggered": false,
  "centroid_drift": 0.12,
  "svm_gap_degradation": 0.05,
  "retraining_gate_status": "closed",
  "rollback_protection_status": "available"
}
```

## Audit Logging

All API requests are logged to `logs/api_audit.jsonl` with:
- Timestamp
- Drug pair
- Route taken
- Final decision
- Confidence scores
- Drift status

## Architecture

The API is a thin wrapper that:
- Loads ML pipeline artifacts once on startup
- Reuses existing `infer_pair()` function
- Returns structured JSON responses
- Maintains full compatibility with Streamlit dashboard
- Does not modify existing codebase

## Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"drug_a": "Aspirin", "drug_b": "Warfarin"}'
```