"""Load SMTP settings from data/smtp_config.json (no SMTP_* / ALERT_TO env required)."""

from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMTP_CONFIG_PATH = _PROJECT_ROOT / "data" / "smtp_config.json"


def smtp_config_path() -> Path:
    return SMTP_CONFIG_PATH


def load_smtp_config() -> dict[str, Any] | None:
    if not SMTP_CONFIG_PATH.exists():
        return None
    try:
        raw = json.loads(SMTP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def validate_smtp_config(cfg: dict[str, Any]) -> tuple[bool, str]:
    host = str(cfg.get("host", "")).strip()
    user = str(cfg.get("user", "")).strip()
    password = str(cfg.get("password", "")).strip()
    from_email = str(cfg.get("from_email", "")).strip() or user
    try:
        port = int(cfg.get("port", 587))
    except (TypeError, ValueError):
        return False, "Invalid port in smtp_config.json"
    if not host:
        return False, "Missing host in data/smtp_config.json"
    if not user:
        return False, "Missing user in data/smtp_config.json"
    if not password:
        return False, "Missing password in data/smtp_config.json (use an app password for Gmail)"
    if not from_email:
        return False, "Missing from_email (or user) in data/smtp_config.json"
    return True, ""


def send_smtp_email(*, to_addr: str, subject: str, body: str, cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """Send one email using file config. Returns status + human-readable message."""
    config = cfg if cfg is not None else load_smtp_config()
    if not config:
        return {
            "status": "not_configured",
            "message": "Create data/smtp_config.json (copy data/smtp_config.example.json) with host, port, user, password, and from_email.",
        }
    ok, err = validate_smtp_config(config)
    if not ok:
        return {"status": "not_configured", "message": err}

    host = str(config["host"]).strip()
    port = int(config["port"])
    user = str(config["user"]).strip()
    password = str(config["password"]).strip()
    from_email = str(config.get("from_email", "")).strip() or user

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_addr.strip()
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=25) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
    except Exception as exc:
        return {"status": "failed", "message": f"Could not send email: {exc}"}

    return {"status": "sent", "message": f"Message sent to {to_addr.strip()}."}
