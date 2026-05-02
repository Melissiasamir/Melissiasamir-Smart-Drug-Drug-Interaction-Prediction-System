"""# 11. Action System (REAL)

This module sends real email via SMTP. It does not print fake alerts. Configure:
SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_FROM, ALERT_TO.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from os import environ, getenv


def send_email_alert(
    drug_a: str,
    drug_b: str,
    risk_level: str,
    confidence: float,
    total_score: float,
    explanation: str,
    action: str,
) -> dict[str, str]:
    """Send a real SMTP alert when an actionable drug interaction risk is predicted."""
    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_FROM", "ALERT_TO"]
    missing = [key for key in required if not getenv(key)]
    if missing:
        return {
            "status": "not_configured",
            "message": "Alert condition met, but SMTP environment variables are missing: " + ", ".join(missing),
        }

    msg = EmailMessage()
    msg["Subject"] = f"Smart Drug Risk Alert [{risk_level}]: {drug_a} + {drug_b}"
    msg["From"] = environ["ALERT_FROM"]
    msg["To"] = environ["ALERT_TO"]
    msg.set_content(
        "Actionable drug interaction risk detected.\n\n"
        f"Drug A: {drug_a}\n"
        f"Drug B: {drug_b}\n"
        f"Risk Level: {risk_level}\n"
        f"Confidence: {confidence:.2%}\n"
        f"Total Score: {total_score:.2f}/100\n\n"
        f"Recommended Action:\n{action}\n\n"
        f"Explanation:\n{explanation}"
    )

    try:
        port = int(environ["SMTP_PORT"])
        with smtplib.SMTP(environ["SMTP_HOST"], port, timeout=20) as server:
            server.starttls()
            server.login(environ["SMTP_USER"], environ["SMTP_PASSWORD"])
            server.send_message(msg)
    except Exception as exc:
        return {"status": "failed", "message": f"SMTP alert failed: {exc}"}

    return {"status": "sent", "message": f"Real SMTP alert sent successfully for {risk_level}."}
