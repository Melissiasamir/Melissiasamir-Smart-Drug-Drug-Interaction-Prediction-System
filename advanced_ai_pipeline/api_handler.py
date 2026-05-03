"""API adapter for doctor input in the advanced AI pipeline."""

from __future__ import annotations

from .pipeline import process_doctor_input


def handle_doctor_input(drug1: str, drug2: str, description: str) -> dict[str, object]:
    return process_doctor_input(drug1, drug2, description)
