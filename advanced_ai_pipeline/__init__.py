"""Advanced AI pipeline extension for Smart Drug-Drug Interaction Prediction System."""

from .api_handler import handle_doctor_input
from .pipeline import process_doctor_input

__all__ = ["process_doctor_input", "handle_doctor_input"]
