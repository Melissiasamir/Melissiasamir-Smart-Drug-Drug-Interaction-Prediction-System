"""Aggregate model outputs into a single structured clinical report."""

from __future__ import annotations

from typing import Any


def _tier_from_svm_label(label: str) -> str:
    if label in {"High Risk", "Dangerous"}:
        return "HIGH"
    if label == "Medium Risk":
        return "MEDIUM"
    return "LOW"


def _alternatives_from_advanced(advanced: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("similar_drugs_1", "similar_drugs_2"):
        for item in advanced.get(key) or []:
            if isinstance(item, dict):
                nm = item.get("drug") or item.get("name")
                if nm:
                    names.append(str(nm))
            elif isinstance(item, str):
                names.append(item)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:8]


def _build_user_friendly_email_and_ui(
    drug_1: str,
    drug_2: str,
    risk_level: str,
    alternatives: list[str],
) -> dict[str, Any]:
    """Plain-language strings for dashboard and email (no ML jargon)."""
    pair = f"{drug_1} and {drug_2}"
    alts = alternatives[:6] if alternatives else []
    alt_text = "\n".join(f"  • {a}" for a in alts) if alts else "  • Ask your doctor or pharmacist for safer options that fit your health profile."

    if risk_level == "HIGH":
        heading = "RISK: HIGH 🔴"
        why = [
            f"These medicines ({pair}) may interact in a way that raises the chance of harm.",
            "The automated check flagged this combination at the highest concern level.",
        ]
        actions = [
            "Do not combine these drugs unless a licensed prescriber tells you it is safe.",
            "Contact your doctor or pharmacist as soon as practical.",
            "Bring your full medication list to the visit.",
        ]
        subject = "⚠️ High Risk Drug Interaction Detected"
        body = (
            f"{heading}\n\n"
            f"Medicines checked: {drug_1} + {drug_2}\n\n"
            "Why this matters:\n"
            + "\n".join(f"  • {line}" for line in why)
            + "\n\nWhat you should do:\n"
            + "\n".join(f"  • {line}" for line in actions)
            + "\n\nSafer alternatives to discuss with a clinician:\n"
            f"{alt_text}\n\n"
            "This message is from an educational screening tool—not a diagnosis."
        )
    elif risk_level == "MEDIUM":
        heading = "RISK: MODERATE 🟡"
        why = [
            f"The pair {pair} may need extra caution.",
            "Some people might need dose changes or monitoring if both are used.",
        ]
        actions = [
            "Review this pair with your doctor or pharmacist before relying on it long term.",
            "Mention other prescriptions, supplements, and herbal products you use.",
        ]
        subject = "Drug interaction check — moderate attention"
        body = (
            f"{heading}\n\nMedicines: {drug_1} + {drug_2}\n\n"
            "Why:\n" + "\n".join(f"  • {w}" for w in why) + "\n\nSteps:\n" + "\n".join(f"  • {a}" for a in actions)
        )
    else:
        heading = "RISK: LOW / SAFE 🟢"
        why = [
            f"The automatic check did not flag {pair} at a high concern level.",
            "This does not guarantee safety for every person or dose.",
        ]
        actions = [
            "Still follow your prescriber’s instructions.",
            "Tell your care team if anything changes with your health or other medicines.",
        ]
        subject = "Drug interaction check — low concern"
        body = (
            f"{heading}\n\nMedicines: {drug_1} + {drug_2}\n\n"
            "Why:\n" + "\n".join(f"  • {w}" for w in why) + "\n\nTips:\n" + "\n".join(f"  • {a}" for a in actions)
        )

    return {
        "user_risk_heading": heading,
        "user_why_bullets": why,
        "user_action_bullets": actions,
        "user_alternatives_bullets": alts if alts else ["Ask your doctor or pharmacist for substitutes appropriate for you."],
        "user_email_subject": subject,
        "user_email_body": body,
    }


def _key_factors_from_shap(shap_rows: list[dict[str, Any]], max_n: int = 6) -> list[str]:
    factors: list[str] = []
    for row in shap_rows[:max_n]:
        feat = row.get("feature")
        if feat is None:
            continue
        imp = row.get("abs_importance", row.get("importance"))
        if isinstance(imp, (int, float)):
            factors.append(f"{feat} (importance {float(imp):.4f})")
        else:
            factors.append(str(feat))
    return factors


def generate_report(context: dict[str, Any]) -> dict[str, Any]:
    """Combine SVM, SHAP, forecast, clustering, and similarity into one report dict."""
    drug_1 = str(context.get("drug_1", "")).strip()
    drug_2 = str(context.get("drug_2", "")).strip()
    description = str(context.get("description", "")).strip()
    svm = context.get("svm") or {}
    svm_label = str(svm.get("label", "Low Risk"))
    confidence = float(svm.get("confidence", 0.0))
    shap_rows = list(context.get("shap") or [])
    scores = context.get("scores") or {}
    total_score = float(scores.get("total_score", 0.0))
    forecast_increasing = bool(context.get("forecast_increasing", False))
    prod_cluster = context.get("production_clustering") or {}
    advanced = context.get("advanced_ai") or {}

    risk_level = _tier_from_svm_label(svm_label)
    if context.get("force_high_risk"):
        risk_level = "HIGH"

    trend = "Forecast indicates increasing dangerous-case activity." if forecast_increasing else (
        "Forecast does not show an increasing dangerous-case trend."
    )

    alternatives = _alternatives_from_advanced(advanced)
    alt_primary = alternatives[0] if alternatives else "Discuss formulary alternatives with pharmacy."

    if risk_level == "HIGH":
        recommended = (
            "Immediate action: avoid this combination when possible, escalate to a clinical reviewer, "
            f"and consider a safer alternative such as {alt_primary}."
        )
    elif risk_level == "MEDIUM":
        recommended = (
            "Review before use: confirm indications, monitor closely, and validate the interaction "
            "against the patient's full medication list."
        )
    else:
        recommended = "Continue routine monitoring; confirm no patient-specific contraindications."

    interpretation = (
        f"SVM risk tier: {svm_label} (mapped to {risk_level}). "
        f"Model confidence {confidence:.1%}. Aggregate score {total_score:.1f}/100. "
        f"{trend} "
        f"Embedding clusters: drug 1 cluster {advanced.get('drug1_cluster', '—')}, "
        f"drug 2 cluster {advanced.get('drug2_cluster', '—')}. "
    )
    if description:
        interpretation += f"Clinician note excerpt: {description[:280]}{'…' if len(description) > 280 else ''}"

    clustering_note = str(prod_cluster.get("dynamic_reclustering", prod_cluster.get("message", "")))
    treatment_plan = [
        recommended,
        "Reconcile the medication list and document the decision in the chart.",
        "If proceeding, define monitoring labs or vitals appropriate to the interaction class.",
    ]
    if clustering_note:
        treatment_plan.append(f"Pipeline clustering status: {clustering_note}")

    note = (
        "This report aggregates automated signals (SVM, SHAP, SARIMA trend, production clustering, "
        "and embedding similarity). It supports—not replaces—clinical judgment."
    )

    report: dict[str, Any] = {
        "risk_level": risk_level,
        "confidence": confidence,
        "interpretation": interpretation,
        "key_factors": _key_factors_from_shap(shap_rows),
        "trend": trend,
        "treatment_plan": treatment_plan,
        "alternatives": alternatives,
        "recommended_action": recommended,
        "note": note,
        "drug_1": drug_1,
        "drug_2": drug_2,
        "svm_label": svm_label,
        "total_score": total_score,
    }
    report.update(_build_user_friendly_email_and_ui(drug_1, drug_2, risk_level, alternatives))
    print("Report generated")
    return report
