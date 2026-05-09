from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import ollama


SYSTEM_PROMPT = """
You are a clinical decision support assistant.
Only use the facts provided and return exactly valid JSON.
"""

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "300"))
OLLAMA_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
OLLAMA_CACHE_SIZE = int(os.environ.get("OLLAMA_CACHE_SIZE", "128"))


@lru_cache(maxsize=OLLAMA_CACHE_SIZE)
def _cached_llm_plan(context_key: str, prompt: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=OLLAMA_TEMPERATURE,
        max_tokens=OLLAMA_MAX_TOKENS,
    )
    return response["message"]["content"]


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

        top_features = []
        try:
            for row in shap_importance.head(3)[["feature", "abs_importance"]].to_dict("records"):
                feature = row.get("feature")
                importance = row.get("abs_importance")
                if feature:
                    top_features.append(
                        f"{feature} ({importance:.3f})" if importance is not None else feature
                    )
        except Exception:
            top_features = []

        return {
            "risk_level": prediction.get("label"),
            "confidence": round(float(prediction.get("confidence", 0.0)), 2),
            "top_shap_features": top_features,
            "rag_vote": rag_result.get("majority_vote"),
            "rag_confidence": round(float(rag_result.get("confidence", 0.0)), 2)
            if rag_result.get("confidence") is not None
            else None,
            "react_route": react_decision.get("route"),
            "drift_triggered": bool(drift_status.get("triggered")),
        }

    def build_prompt(self, context: dict[str, Any]) -> str:
        facts = [
            f"Risk level: {context.get('risk_level')}",
            f"Confidence: {context.get('confidence')}",
            f"Top drivers: {', '.join(context.get('top_shap_features', []) or ['none'])}",
            f"RAG vote: {context.get('rag_vote')} ({context.get('rag_confidence')})",
            f"ReAct route: {context.get('react_route')}",
            f"Drift triggered: {'yes' if context.get('drift_triggered') else 'no'}",
        ]

        return (
            "Use only the following facts to generate the requested output. "
            "Do not add any extra fields. Return valid JSON only.\n\n"
            "Facts:\n"
            + "\n".join(facts)
            + "\n\nOutput schema:\n"
            '{"risk_level":"...","interpretation":"...","key_factors":["..."],'
            '"treatment_plan":["..."],"recommended_action":"...","uncertainty_notes":"...",'
            '"monitoring_strategy":"..."}'
        )

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
        cache_key = json.dumps(context, sort_keys=True)

        try:
            content = _cached_llm_plan(cache_key, prompt)
            return json.loads(content)

        except Exception as exc:
            return {
                "risk_level": prediction.get("label"),
                "interpretation": "LLM generation failed",
                "key_factors": [
                    "Use deterministic clinical templates when LLM is unavailable."
                ],
                "treatment_plan": [
                    "Fallback to rule-based action system"
                ],
                "recommended_action": "HUMAN_REVIEW",
                "uncertainty_notes": str(exc),
                "monitoring_strategy": "Manual monitoring required",
            }