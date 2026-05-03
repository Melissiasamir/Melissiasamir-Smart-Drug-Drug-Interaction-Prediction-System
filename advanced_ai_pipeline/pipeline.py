"""Core Advanced AI pipeline entrypoint."""

from __future__ import annotations

from .doctor_pipeline.doctor_handler import DoctorPipeline


def process_doctor_input(drug1: str, drug2: str, description: str) -> dict[str, object]:
    """Process doctor-submitted drug pairs through the advanced AI extension."""
    pipeline = DoctorPipeline()
    return pipeline.handle_doctor_input(drug1, drug2, description)
