"""User-facing clinical reporting, profile storage, and HIGH-risk email alerts."""

from __future__ import annotations

from .email_service import send_email
from .integration import build_pair_report, load_persisted_report, process_doctor_submission
from .report_generator import generate_report
from .user_display import display_pair_friendly_report, display_user_dashboard, render_user_email_controls
from .user_profile import get_user_email, save_user_email

__all__ = [
    "build_pair_report",
    "display_pair_friendly_report",
    "display_user_dashboard",
    "generate_report",
    "get_user_email",
    "load_persisted_report",
    "process_doctor_submission",
    "render_user_email_controls",
    "save_user_email",
    "send_email",
]
