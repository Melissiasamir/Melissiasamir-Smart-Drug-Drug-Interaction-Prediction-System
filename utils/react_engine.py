"""Simple ReAct reasoning loop for action dispatch.

The loop is explicit and auditable:

Observe -> Reason -> Act -> Observe

It dynamically chooses from available routes using score, dominant SHAP feature,
RAG vote, ARIMA trend, and SHAP reliability.
"""

from __future__ import annotations

from typing import Any, Callable


RAGRetriever = Callable[[], dict[str, Any]]
TERMINAL_ROUTES = {
    "direct_dispatch",
    "human_review",
    "tie_handling",
    "no_evidence_handling",
    "low_score_rejection",
}


def build_observation(
    final_score: float,
    shap_dominant_feature: str | None,
    rag_vote: dict[str, Any] | None,
    arima_multiplier: float,
    reliability_score: float,
    prediction_label: str,
    arima_signal: float | None = None,
) -> dict[str, Any]:
    return {
        "final_score": float(final_score),
        "shap_dominant_feature": shap_dominant_feature,
        "rag_vote": rag_vote,
        "arima_multiplier": float(arima_multiplier),
        "arima_signal": None if arima_signal is None else float(arima_signal),
        "reliability_score": float(reliability_score),
        "prediction_label": prediction_label,
    }


def reason_about_next_action(observation: dict[str, Any], used_actions: set[str]) -> tuple[str, str]:
    """Choose the next route from the current observation."""
    rag_vote = observation.get("rag_vote") or {}
    rag_status = rag_vote.get("status")
    score = float(observation["final_score"])
    reliability = float(observation["reliability_score"])
    arima_multiplier = float(observation["arima_multiplier"])
    arima_signal = observation.get("arima_signal")
    dominant_feature = observation.get("shap_dominant_feature")

    if rag_status == "tie" and "tie_handling" not in used_actions:
        return "tie_handling", "Retrieved evidence is split across labels."
    if rag_status == "no_evidence" and "no_evidence_handling" not in used_actions:
        return "no_evidence_handling", "The retrieval store has no usable neighbors."
    if rag_status == "low_confidence" and "human_review" not in used_actions:
        return "human_review", "Historical cases vote weakly, so review is safer."
    if score < 30 and "low_score_rejection" not in used_actions:
        return "low_score_rejection", "Composite score is too low for automated escalation."
    if reliability < 0.55 and dominant_feature and "shap_recheck" not in used_actions:
        return "shap_recheck", f"SHAP reliability is low around dominant feature '{dominant_feature}'."
    if 35 <= score <= 75 and "rag_retrieval" not in used_actions and rag_status is None:
        return "rag_retrieval", "Borderline score needs historical case evidence."
    trend_is_elevated = bool(arima_multiplier > 1.05 or (arima_signal is not None and float(arima_signal) > 0.70))
    if trend_is_elevated and reliability < 0.70 and "human_review" not in used_actions:
        return "human_review", "Forecast is amplifying risk while explanation reliability is modest."
    return "direct_dispatch", "Signals are sufficiently consistent for direct action dispatch."


def estimate_action_confidence(action: str, observation: dict[str, Any]) -> float:
    """Estimate confidence in the chosen route from current evidence quality."""
    score = float(observation.get("final_score", 0.0))
    reliability = float(observation.get("reliability_score", 0.0))
    rag_vote = observation.get("rag_vote") or {}
    rag_confidence = float(rag_vote.get("vote_confidence", 0.0) or 0.0)

    if action == "direct_dispatch":
        return float(min(0.95, 0.45 + (score / 200.0) + (reliability * 0.30) + (rag_confidence * 0.15)))
    if action in {"human_review", "tie_handling", "no_evidence_handling"}:
        uncertainty = 1.0 - max(reliability, rag_confidence)
        return float(np_clip(0.55 + uncertainty * 0.35, 0.0, 0.95))
    if action == "low_score_rejection":
        return float(np_clip((30.0 - score) / 30.0, 0.40, 0.95))
    if action == "rag_retrieval":
        return 0.70
    if action == "shap_recheck":
        return float(np_clip(1.0 - reliability, 0.40, 0.95))
    return 0.50


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def act(
    action: str,
    observation: dict[str, Any],
    retrieval_fn: RAGRetriever | None = None,
) -> dict[str, Any]:
    """Execute one route and return the new observation data."""
    if action == "rag_retrieval":
        retrieval = retrieval_fn() if retrieval_fn else {"status": "no_evidence", "vote": {"status": "no_evidence"}}
        vote = retrieval.get("vote", {"status": retrieval.get("status", "no_evidence")})
        return {"rag_vote": vote, "rag_evidence": retrieval.get("cases", []), "rag_status": retrieval.get("status")}

    if action == "shap_recheck":
        return {"shap_recheck_required": True}
    if action == "human_review":
        return {"human_review_required": True}
    if action == "tie_handling":
        return {"tie_detected": True, "human_review_required": True}
    if action == "no_evidence_handling":
        return {"no_evidence": True, "human_review_required": True}
    if action == "low_score_rejection":
        return {"rejected": True, "human_review_required": False}
    return {"dispatch_ready": True}


def summarize_decision(route: str, observation: dict[str, Any], action_confidence: float) -> dict[str, Any]:
    """Build a dashboard/report-friendly final decision object."""
    label = observation.get("prediction_label")
    score = float(observation.get("final_score", 0.0))
    review = bool(observation.get("human_review_required", False))
    rejected = bool(observation.get("rejected", False))
    dispatch_ready = bool(observation.get("dispatch_ready", False))

    if rejected:
        summary = "Rejected for automated dispatch because the agentic score is low."
    elif review:
        summary = "Human review required because the evidence is uncertain or conflicting."
    elif dispatch_ready:
        summary = "Direct dispatch approved because model, explanation, trend, and retrieval signals are aligned."
    else:
        summary = "Additional checks completed; use the selected route outcome."

    return {
        "route": route,
        "prediction_label": label,
        "final_score": score,
        "action_confidence": float(action_confidence),
        "human_review_required": review,
        "dispatch_ready": dispatch_ready,
        "rejected": rejected,
        "decision_explanation": summary,
        "terminal": route in TERMINAL_ROUTES,
    }


def run_react_loop(
    initial_observation: dict[str, Any],
    retrieval_fn: RAGRetriever | None = None,
    max_steps: int = 4,
) -> dict[str, Any]:
    """Run Observe -> Reason -> Act -> Observe until a terminal route is reached."""
    observation = dict(initial_observation)
    trace = []
    used_actions: set[str] = set()
    last_confidence = 0.0

    for step in range(1, max_steps + 1):
        before = dict(observation)
        action, reason = reason_about_next_action(observation, used_actions)
        used_actions.add(action)
        last_confidence = estimate_action_confidence(action, observation)
        result = act(action, observation, retrieval_fn)
        observation.update(result)
        trace.append(
            {
                "step": step,
                "observe_before": before,
                "reason": reason,
                "act": action,
                "action_confidence": last_confidence,
                "observe_after": dict(observation),
                "result": result,
                "terminal": action in TERMINAL_ROUTES,
            }
        )
        if action in TERMINAL_ROUTES:
            break

    final_route = trace[-1]["act"] if trace else "direct_dispatch"
    final_summary = summarize_decision(final_route, observation, last_confidence)
    return {
        "route": final_route,
        "trace": trace,
        "final_observation": observation,
        "action_confidence": final_summary["action_confidence"],
        "decision_explanation": final_summary["decision_explanation"],
        "final_decision": final_summary,
        "human_review_required": bool(observation.get("human_review_required", False)),
        "dispatch_ready": bool(observation.get("dispatch_ready", False)),
        "rejected": bool(observation.get("rejected", False)),
        "terminal": final_summary["terminal"],
    }
