"""Streamlit UI: user email capture, plain-language reports, and clinical summary panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from .smtp_client import smtp_config_path
from .user_profile import get_user_email, save_user_email


def render_user_email_controls() -> None:
    """Visible email field + Save for the user dashboard."""
    if "user_email_field" not in st.session_state:
        st.session_state["user_email_field"] = get_user_email() or ""

    st.markdown('<div class="section-title">✉️ Alert contact</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.text_input("Enter your email", key="user_email_field")
    with col_b:
        st.write("")
        st.write("")
        if st.button("Save", key="user_email_save_btn", use_container_width=True):
            ok, msg = save_user_email(st.session_state.get("user_email_field", ""))
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    example = Path(__file__).resolve().parents[2] / "data" / "smtp_config.example.json"
    with st.expander("Email delivery setup", expanded=False):
        st.markdown(
            f"High-risk alerts are sent using **`{smtp_config_path().name}`** in the `data/` folder. "
            f"Copy **`{example.name}`** to **`smtp_config.json`** and add your SMTP app password (e.g. Gmail app password). "
            "That file is gitignored so credentials stay on your machine."
        )


def display_pair_friendly_report(report: dict[str, Any]) -> None:
    """Plain-language summary after Analyze Risk (no ML jargon)."""
    print("Displaying report")
    heading = report.get("user_risk_heading") or f"RISK: {report.get('risk_level', '—')}"
    st.markdown('<div class="section-title">📘 Your interaction summary</div>', unsafe_allow_html=True)
    st.markdown(f"### {heading}")
    st.markdown("**Why?**")
    for line in report.get("user_why_bullets") or []:
        st.markdown(f"- {line}")
    st.markdown("**What should you do?**")
    for line in report.get("user_action_bullets") or []:
        st.markdown(f"- {line}")
    st.markdown("**Safer alternatives to discuss with a clinician**")
    for line in report.get("user_alternatives_bullets") or []:
        st.markdown(f"- {line}")
    st.caption("Educational screening only—not medical advice.")


def display_user_dashboard(report: dict[str, Any] | None) -> None:
    """Doctor-triggered report: plain language first, optional technical expander."""
    if not report:
        return

    print("Displaying report")
    if report.get("user_risk_heading"):
        st.markdown('<div class="section-title">📋 Clinical summary report</div>', unsafe_allow_html=True)
        st.markdown(f"### {report.get('user_risk_heading')}")
        st.markdown("**Why?**")
        for line in report.get("user_why_bullets") or []:
            st.markdown(f"- {line}")
        st.markdown("**What should you do?**")
        for line in report.get("user_action_bullets") or []:
            st.markdown(f"- {line}")
        st.markdown("**Safer alternatives**")
        for line in report.get("user_alternatives_bullets") or []:
            st.markdown(f"- {line}")
        with st.expander("Optional technical details", expanded=False):
            st.write(report.get("interpretation", ""))
            st.markdown("**Plan**")
            for step in report.get("treatment_plan") or []:
                st.write(f"- {step}")
            st.caption(report.get("note", ""))
        return

    risk = str(report.get("risk_level", "—"))
    risk_class = "danger" if risk == "HIGH" else "risky" if risk == "MEDIUM" else "safe"

    st.markdown('<div class="section-title">📋 Clinical summary report</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-card">
            <div class="mini-label">Risk (highlighted)</div>
            <div class="mini-value">{risk}</div>
            <span class="risk-chip {risk_class}">{risk}</span>
            <div class="mini-label" style="margin-top:1rem;">Confidence</div>
            <div class="mini-value">{float(report.get("confidence", 0.0)):.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Explanation**", unsafe_allow_html=True)
    st.write(report.get("interpretation", ""))

    st.markdown("**Plan**", unsafe_allow_html=True)
    for step in report.get("treatment_plan") or []:
        st.write(f"- {step}")

    st.markdown("**Alternatives**", unsafe_allow_html=True)
    alts = report.get("alternatives") or []
    if alts:
        for a in alts:
            st.write(f"- {a}")
    else:
        st.write("_No similarity-ranked alternatives returned for this pair._")

    st.caption(report.get("note", ""))
