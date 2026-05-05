"""SMTP alerts using data/smtp_config.json (no SMTP_* / ALERT_TO environment variables)."""

from __future__ import annotations

from advanced_ai_pipeline.reporting.smtp_client import send_smtp_email


def send_email_alert(
    drug_a: str,
    drug_b: str,
    risk_level: str,
    confidence: float,
    total_score: float,
    explanation: str,
    action: str,
    to_email: str | None = None,
) -> dict[str, str]:
    """
    Legacy helper: sends a plain alert if recipient email is provided.
    Prefer advanced_ai_pipeline.reporting.email_service.send_email for HIGH-risk user reports.
    """
    recipient = (to_email or "").strip()
    if not recipient:
        return {
            "status": "not_configured",
            "message": "No recipient email. Enter and save your email in the User dashboard sidebar.",
        }
    subject = f"Smart Drug Risk Alert [{risk_level}]: {drug_a} + {drug_b}"
    body = (
        "Drug interaction screening alert\n\n"
        f"Medicines: {drug_a} + {drug_b}\n"
        f"Risk level: {risk_level}\n"
        f"Confidence: {confidence:.0%}\n"
        f"Score: {total_score:.1f} / 100\n\n"
        f"What to do:\n{action}\n\n"
        f"Notes:\n{explanation}\n"
    )
    return send_smtp_email(to_addr=recipient, subject=subject, body=body)
