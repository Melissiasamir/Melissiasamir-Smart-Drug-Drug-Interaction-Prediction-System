from __future__ import annotations

import json
from typing import Any

import ollama


SYSTEM_PROMPT = """
You are an intelligent clinical decision support agent.

You analyze:
- SVM predictions
- SHAP explanations
- Forecasting trends
- RAG evidence
- ReAct decisions
- Drift signals

You must generate:
- interpretation
- key factors
- treatment plan
- recommended action
- uncertainty notes

Return ONLY valid JSON.
"""


class LLMTreatmentAgent:

    def build_context(
        self,
        prediction: dict[str, Any],
        scores: dict[str, Any],
        shap_importance,
        rag_result: dict[str, Any],
        react_decision: dict[str, Any],
        drift_status: dict[str, Any],
    ) -> dict[str, Any]:

        try:
            top_features = (
                shap_importance.head(5)[
                    ["feature", "shap_value", "abs_importance"]
                ]
                .to_dict("records")
            )
        except Exception:
            top_features = []

        return {
            "risk_level": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "svm_gap": prediction.get("svm_gap"),
            "fuzzy_membership": prediction.get("fuzzy_membership"),

            "forecast_score": scores.get("forecast_score"),
            "final_agentic_score": scores.get("final_agentic_score"),

            "top_shap_features": top_features,

            "rag_vote": rag_result.get("majority_vote"),
            "rag_confidence": rag_result.get("confidence"),

            "react_route": react_decision.get("route"),

            "drift_triggered": drift_status.get("triggered"),
        }

    def build_prompt(self, context: dict[str, Any]) -> str:

        return f"""
Analyze the following AI outputs and generate a structured treatment/action plan.

Context:
{json.dumps(context, indent=2)}

Return ONLY valid JSON using this format:

{{
    "risk_level": "...",
    "interpretation": "...",

    "key_factors": [
        "...",
        "..."
    ],

    "treatment_plan": [
        "...",
        "...",
        "..."
    ],

    "recommended_action": "...",

    "uncertainty_notes": "...",

    "monitoring_strategy": "..."
}}
"""

    def generate_plan(
        self,
        prediction: dict[str, Any],
        scores: dict[str, Any],
        shap_importance,
        rag_result: dict[str, Any],
        react_decision: dict[str, Any],
        drift_status: dict[str, Any],
    ) -> dict[str, Any]:

        context = self.build_context(
            prediction=prediction,
            scores=scores,
            shap_importance=shap_importance,
            rag_result=rag_result,
            react_decision=react_decision,
            drift_status=drift_status,
        )

        prompt = self.build_prompt(context)

        try:

            response = ollama.chat(
                model="mistral",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = response["message"]["content"]

            return json.loads(content)

        except Exception as exc:

            return {
                "risk_level": prediction.get("label"),

                "interpretation": "LLM generation failed",

                "key_factors": [],

                "treatment_plan": [
                    "Fallback to rule-based action system"
                ],

                "recommended_action": "HUMAN_REVIEW",

                "uncertainty_notes": str(exc),

                "monitoring_strategy": "Manual monitoring required",
            }