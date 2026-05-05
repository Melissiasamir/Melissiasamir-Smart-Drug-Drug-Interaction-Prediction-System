"""Persist user email locally for alert delivery."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_PATH = _PROJECT_ROOT / "data" / "user_profile.json"

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def _profile_path() -> Path:
    return _PROFILE_PATH


def get_user_email() -> str | None:
    path = _profile_path()
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    email = data.get("email")
    if not isinstance(email, str):
        return None
    email = email.strip()
    return email or None


def save_user_email(email: str) -> tuple[bool, str]:
    raw = (email or "").strip()
    if not raw:
        return False, "Email is empty."
    if not _EMAIL_RE.match(raw):
        return False, "Invalid email format."
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"email": raw}, indent=2), encoding="utf-8")
    return True, "Email saved."
