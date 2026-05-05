"""Send HIGH-risk emails to the dashboard user using data/smtp_config.json."""

from __future__ import annotations

from typing import Any

from .smtp_client import send_smtp_email


def send_email(report: dict[str, Any], user_email: str) -> dict[str, str]:
    """Send only for HIGH risk; uses plain-language body from the report."""
    if report.get("risk_level") != "HIGH":
        return {"status": "skipped", "message": "Not a high-risk report; email not sent."}

    subject = "⚠️ High Risk Drug Interaction Detected"
    body = str(
        report.get("user_email_body")
        or f"High risk flagged for {report.get('drug_1', '')} + {report.get('drug_2', '')}. Please contact your clinician."
    )

    result = send_smtp_email(to_addr=user_email, subject=subject, body=body)
    if result.get("status") == "sent":
        print("High risk - email sent")
    else:
        print("Email failed")
    return result
