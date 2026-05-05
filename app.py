"""Smart Drug Risk Agent

# 1. Project Overview
An intelligent AI system that takes Drug A and Drug B, predicts interaction risk,
explains the decision, forecasts future dangerous cases, and triggers real SMTP
alerts when danger is predicted and future risk is increasing.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from advanced_ai_pipeline.reporting.email_service import send_email
from advanced_ai_pipeline.reporting.integration import (
    build_pair_report,
    load_persisted_report,
    process_doctor_submission,
)
from advanced_ai_pipeline.reporting.user_display import (
    display_pair_friendly_report,
    display_user_dashboard,
    render_user_email_controls,
)
from advanced_ai_pipeline.reporting.user_profile import get_user_email
from utils.data_loader import dataset_description, load_datasets
from utils.explainability import compute_shap_importance, shap_plot
from utils.forecasting import forecast_plot
from utils.pipeline import infer_pair, load_or_train_pipeline
from utils.preprocessing import clean_ddi_data
from utils.scoring import compute_scores
from utils.self_learning import SelfLearningEngine, SelfLearningConfig


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / ".streamlit_runtime"
SELF_LEARNING_SAMPLES_PATH = PROJECT_ROOT / "data" / "self_learning_samples.csv"
DOCTOR_INTERACTIONS_SOURCE_PATH = PROJECT_ROOT / "data" / "doctor_added_interactions.xlsx"


def resolve_writable_path(preferred_path: Path) -> Path:
    """Use the preferred path when writable, otherwise store generated files locally."""
    try:
        preferred_path.parent.mkdir(parents=True, exist_ok=True)
        probe = preferred_path.parent / ".write_probe.tmp"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred_path
    except OSError:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        return RUNTIME_DIR / preferred_path.name


BACKGROUND_IMAGE_CANDIDATES = [
    PROJECT_ROOT / "Pin by Ashwin Raj on Set Wallpapers _ Science drawing, Science poster, Chemistry posters.jpg",
    PROJECT_ROOT.parent / "Pin by Ashwin Raj on Set Wallpapers _ Science drawing, Science poster, Chemistry posters.jpg",
]
BACKGROUND_IMAGE_PATH = next((path for path in BACKGROUND_IMAGE_CANDIDATES if path.exists()), None)
BACKGROUND_IMAGE_CSS = (
    f'url("data:image/jpeg;base64,{base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("utf-8")}")'
    if BACKGROUND_IMAGE_PATH
    else """
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.14), transparent 32rem),
        radial-gradient(circle at top right, rgba(168, 85, 247, 0.12), transparent 34rem),
        linear-gradient(135deg, #030712 0%, #07111f 52%, #090b16 100%)
    """
)


st.set_page_config(page_title="Smart Drug Risk Agent", page_icon="💊", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #050816;
        --panel: rgba(255, 255, 255, 0.05);
        --line: rgba(120, 220, 255, 0.18);
        --cyan: #22d3ee;
        --purple: #a855f7;
        --green: #34d399;
        --yellow: #facc15;
        --red: #fb7185;
        --text: #e5edf7;
        --muted: #9aa8bd;
    }

    .stApp {
        background-image: __BACKGROUND_IMAGE_CSS__;
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: var(--text);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background: rgba(0, 0, 0, 0.6);
    }

    .stApp > * {
        position: relative;
        z-index: 1;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2.1rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(10, 16, 32, 0.84), rgba(3, 7, 18, 0.88)),
            radial-gradient(circle at top, rgba(34, 211, 238, 0.16), transparent 22rem);
        border-right: 1px solid rgba(34, 211, 238, 0.16);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        height: 100vh;
        overflow-y: auto;
    }

    section[data-testid="stSidebar"] > div {
        height: 100vh;
        overflow-y: auto;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    h1 {
        font-size: 2.45rem !important;
        font-weight: 850 !important;
        color: #f8fbff;
    }

    h2, h3 {
        color: #f4f7fb;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.55rem 1.6rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(34, 211, 238, 0.18);
        border-radius: 18px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.62), rgba(8, 13, 28, 0.54)),
            linear-gradient(90deg, rgba(34, 211, 238, 0.10), rgba(168, 85, 247, 0.10));
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .hero-kicker {
        color: var(--cyan);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3.45rem);
        line-height: 1.02;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }

    .hero-copy {
        max-width: 880px;
        color: #b9c5d8;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    .section-card {
        position: relative;
        padding: 1.15rem 1.2rem 1.25rem;
        margin: 0.95rem 0 1.15rem;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: linear-gradient(180deg, var(--panel), rgba(8, 13, 28, 0.34));
        box-shadow: 0 18px 54px rgba(0, 0, 0, 0.30), 0 0 22px rgba(34, 211, 238, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }

    .section-card:hover {
        transform: translateY(-1px);
        border-color: rgba(34, 211, 238, 0.34);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.34), 0 0 26px rgba(168, 85, 247, 0.08);
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.1rem 0 0.8rem;
        color: #f8fbff;
        font-size: 1.32rem;
        font-weight: 850;
    }

    .section-card .section-title {
        margin-top: 0;
    }

    .section-subtitle {
        margin-top: -0.35rem;
        margin-bottom: 1rem;
        color: var(--muted);
        font-size: 0.94rem;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.85rem;
    }

    .mini-card, .status-box, .overview-item {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .role-grid, .doctor-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1rem 0 1.25rem;
    }

    .role-option, .doctor-command, .doctor-preview {
        border: 1px solid rgba(34, 211, 238, 0.18);
        border-radius: 16px;
        padding: 1.1rem;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.70), rgba(8, 13, 28, 0.52)),
            radial-gradient(circle at top right, rgba(34, 211, 238, 0.11), transparent 16rem);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .role-option-title, .doctor-command-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 900;
        margin-bottom: 0.35rem;
    }

    .role-option-copy, .doctor-command-copy {
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.94rem;
    }

    .doctor-stat {
        border: 1px solid rgba(148, 163, 184, 0.17);
        border-radius: 14px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.055);
    }

    .doctor-stat-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .doctor-stat-value {
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 900;
        overflow-wrap: anywhere;
    }

    .doctor-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }

    .doctor-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.36rem 0.72rem;
        color: #dff8ff;
        border: 1px solid rgba(34, 211, 238, 0.28);
        background: rgba(34, 211, 238, 0.08);
        font-weight: 800;
        font-size: 0.82rem;
    }

    .mini-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .mini-value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .risk-chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.34rem 0.72rem;
        font-weight: 850;
        font-size: 0.86rem;
        border: 1px solid currentColor;
        margin-top: 0.55rem;
    }

    .stProgress {
        margin-top: 0.85rem;
    }

    .safe { color: var(--green); background: rgba(52, 211, 153, 0.10); }
    .risky { color: var(--yellow); background: rgba(250, 204, 21, 0.11); }
    .danger { color: var(--red); background: rgba(251, 113, 133, 0.12); }

    .status-box.success {
        border-color: rgba(52, 211, 153, 0.42);
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.16), rgba(15, 23, 42, 0.66));
        color: #d1fae5;
    }

    .status-box.warning {
        border-color: rgba(250, 204, 21, 0.46);
        background: linear-gradient(135deg, rgba(250, 204, 21, 0.15), rgba(15, 23, 42, 0.66));
        color: #fef3c7;
    }

    .status-box.danger-box {
        border-color: rgba(251, 113, 133, 0.48);
        background: linear-gradient(135deg, rgba(251, 113, 133, 0.16), rgba(15, 23, 42, 0.66));
        color: #ffe4e6;
    }

    .status-title {
        font-weight: 850;
        margin-bottom: 0.35rem;
    }

    .muted {
        color: var(--muted);
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(34, 211, 238, 0.17);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.07), rgba(10, 15, 31, 0.48));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 14px 36px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 850;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--green), var(--cyan), var(--purple));
    }

    .stButton > button {
        border: 0;
        border-radius: 12px;
        padding: 0.78rem 1rem;
        font-weight: 850;
        color: #03111c;
        background: linear-gradient(135deg, var(--cyan), #7dd3fc 48%, var(--purple));
        box-shadow: 0 12px 34px rgba(34, 211, 238, 0.22);
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        box-shadow: 0 16px 40px rgba(168, 85, 247, 0.26);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(34, 211, 238, 0.16);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .sidebar-brand {
        padding: 0.9rem 0 1rem;
    }

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 900;
        color: #ffffff;
    }

    .sidebar-caption {
        color: var(--muted);
        line-height: 1.55;
        margin-top: 0.35rem;
    }

    @media (max-width: 760px) {
        .summary-grid {
            grid-template-columns: 1fr;
        }

        .role-grid, .doctor-stat-grid {
            grid-template-columns: 1fr;
        }

        .hero {
            padding: 1.2rem;
        }
    }
    </style>
    """.replace("__BACKGROUND_IMAGE_CSS__", BACKGROUND_IMAGE_CSS),
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading saved AI pipeline, or training once if no saved model exists...")
def get_artifacts():
    return load_or_train_pipeline()


@st.cache_data(show_spinner=False)
def get_drug_options() -> list[str]:
    ddi, _ = load_datasets()
    ddi = clean_ddi_data(ddi)
    return sorted(set(ddi["Drug 1"]).union(set(ddi["Drug 2"])))


@st.cache_resource(show_spinner=False)
def get_self_learning_engine() -> SelfLearningEngine:
    """Initialize self-learning engine with config and load persisted samples if available."""
    samples_path = resolve_writable_path(SELF_LEARNING_SAMPLES_PATH)
    config = SelfLearningConfig(
        sample_collection_threshold=20,
        uncertainty_confidence_threshold=0.6,
        augmentation_noise_std=0.05,
        persist_to_csv=True,
        csv_path=samples_path,
        log_actions=True,
    )
    engine = SelfLearningEngine(config)
    # Try to load persisted samples from previous sessions
    if samples_path.exists():
        engine.load_samples(samples_path)
    elif samples_path != SELF_LEARNING_SAMPLES_PATH and SELF_LEARNING_SAMPLES_PATH.exists():
        engine.load_samples(SELF_LEARNING_SAMPLES_PATH)
    return engine


def risk_class(label: str) -> str:
    if label == "Low Risk":
        return "safe"
    if label == "Medium Risk":
        return "risky"
    return "danger"


def recommended_action(risk_label: str, forecast_increasing: bool) -> str:
    if risk_label in {"High Risk", "Dangerous"}:
        return "Immediate action: avoid this drug pair, escalate to the clinical reviewer, and use a safer alternative."
    if forecast_increasing:
        return "Action required: review this drug pair before approval because forecasted dangerous cases are increasing."
    return "Action required: monitor closely and require clinical confirmation before use."


def summary_card(label: str, value: str) -> str:
    return f"""
    <div class="mini-card">
        <div class="mini-label">{label}</div>
        <div class="mini-value">{value}</div>
    </div>
    """


def render_html(markup: str) -> None:
    st.html(markup)


def go_to_dashboard_picker() -> None:
    st.session_state.dashboard = None
    st.rerun()


def render_dashboard_back_button(location: str) -> None:
    if st.button("Back to Dashboard Selection", key=f"back_to_dashboard_picker_{location}", width="stretch"):
        go_to_dashboard_picker()


def status_box(title: str, message: str, tone: str) -> None:
    render_html(
        f"""
        <div class="status-box {tone}">
            <div class="status-title">{title}</div>
            <div>{message}</div>
        </div>
        """
    )


def polish_plotly(fig, height: int = 360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.35)",
        font=dict(color="#dbeafe", family="Inter, Segoe UI, sans-serif"),
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#22d3ee", font_size=13),
        margin=dict(l=24, r=24, t=28, b=24),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig


def score_tone(total_score: float) -> str:
    if total_score < 35:
        return "safe"
    if total_score < 70:
        return "risky"
    return "danger"


def doctor_interactions_path() -> Path:
    return resolve_writable_path(DOCTOR_INTERACTIONS_SOURCE_PATH)


def load_doctor_interactions() -> pd.DataFrame:
    path = doctor_interactions_path()
    columns = ["Submitted At", "Drug 1", "Drug 2", "Interaction Description"]
    if not path.exists() and path != DOCTOR_INTERACTIONS_SOURCE_PATH and DOCTOR_INTERACTIONS_SOURCE_PATH.exists():
        path = DOCTOR_INTERACTIONS_SOURCE_PATH
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_excel(path)


def save_doctor_interaction(drug_1: str, drug_2: str, description: str) -> Path:
    path = doctor_interactions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame(
        [
            {
                "Submitted At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Drug 1": drug_1,
                "Drug 2": drug_2,
                "Interaction Description": description,
            }
        ]
    )
    existing = load_doctor_interactions()
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_excel(path, index=False, sheet_name="Doctor Interactions")
    return path


def choose_dashboard() -> str:
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = None

    if st.session_state.dashboard:
        return st.session_state.dashboard

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Smart Drug Risk Agent</div>
            <div class="hero-title">Choose Dashboard</div>
            <div class="hero-copy">Select the workspace you want to open.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_html(
        """
        <div class="role-grid">
            <div class="role-option">
                <div class="role-option-title">User Dashboard</div>
                <div class="role-option-copy">Analyze two selected drugs, review risk predictions, explanations, forecasts, and alert status.</div>
            </div>
            <div class="role-option">
                <div class="role-option-title">Dr Dashboard</div>
                <div class="role-option-copy">Record clinically reviewed drug interaction notes into a separate doctor-curated Excel dataset.</div>
            </div>
        </div>
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("User Dashboard", type="primary", width="stretch"):
            st.session_state.dashboard = "user"
            st.rerun()
    with col2:
        if st.button("Dr Dashboard", width="stretch"):
            st.session_state.dashboard = "doctor"
            st.rerun()
    st.stop()


def render_doctor_dashboard() -> None:
    nav_col, _ = st.columns([0.25, 0.75])
    with nav_col:
        render_dashboard_back_button("doctor_top")

    doctor_df = load_doctor_interactions()
    total_entries = len(doctor_df)
    unique_drugs = (
        pd.concat([doctor_df["Drug 1"], doctor_df["Drug 2"]]).dropna().astype(str).str.strip().nunique()
        if total_entries
        else 0
    )
    last_update = str(doctor_df["Submitted At"].iloc[-1]) if total_entries else "No records yet"

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Doctor Dashboard</div>
            <div class="hero-title">Clinical Interaction Intake</div>
            <div class="hero-copy">Create a doctor-curated interaction record with two drugs and a clear clinical description.</div>
            <div class="doctor-pill-row">
                <span class="doctor-pill">Excel dataset</span>
                <span class="doctor-pill">Doctor notes</span>
                <span class="doctor-pill">Separate from training CSV</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "doctor_save_message" in st.session_state:
        st.success(st.session_state.pop("doctor_save_message"))

    render_html(
        f"""
        <div class="doctor-stat-grid">
            <div class="doctor-stat">
                <div class="doctor-stat-label">Doctor Entries</div>
                <div class="doctor-stat-value">{total_entries}</div>
            </div>
            <div class="doctor-stat">
                <div class="doctor-stat-label">Unique Drugs</div>
                <div class="doctor-stat-value">{unique_drugs}</div>
            </div>
            <div class="doctor-stat">
                <div class="doctor-stat-label">Dataset</div>
                <div class="doctor-stat-value">doctor_added_interactions.xlsx</div>
            </div>
            <div class="doctor-stat">
                <div class="doctor-stat-label">Last Update</div>
                <div class="doctor-stat-value">{last_update}</div>
            </div>
        </div>
        """
    )

    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        render_html(
            """
            <div class="doctor-command">
                <div class="doctor-command-title">Add Reviewed Interaction</div>
                <div class="doctor-command-copy">Enter the exact pair and the interaction description that should be saved for doctor review.</div>
            </div>
            """
        )
        with st.form("add_interaction_form", clear_on_submit=True):
            drug_1 = st.text_input("Drug 1", placeholder="Example: Warfarin")
            drug_2 = st.text_input("Drug 2", placeholder="Example: Aspirin")
            description = st.text_area(
                "Interaction Description",
                height=180,
                placeholder="Describe the clinical interaction, expected risk, and any monitoring or avoidance advice.",
            )
            submitted = st.form_submit_button("Add Interaction", type="primary", width="stretch")

    with right:
        render_html(
            """
            <div class="doctor-preview">
                <div class="doctor-command-title">Dataset Preview</div>
                <div class="doctor-command-copy">The newest doctor submissions appear here and are stored in the Excel file inside the data folder.</div>
            </div>
            """
        )
        if total_entries:
            st.dataframe(doctor_df.tail(6).iloc[::-1], width="stretch", hide_index=True)
        else:
            status_box(
                "No doctor records yet",
                "Submit the first reviewed interaction to create the Excel dataset.",
                "warning",
            )

    if submitted:
        drug_1 = drug_1.strip()
        drug_2 = drug_2.strip()
        description = description.strip()
        if not drug_1 or not drug_2 or not description:
            st.error("Please fill Drug 1, Drug 2, and Interaction Description.")
            return

        saved_path = save_doctor_interaction(drug_1, drug_2, description)
        try:
            report = process_doctor_submission(drug_1, drug_2, description)
            st.session_state["last_clinical_report"] = report
            st.session_state.pop("doctor_report_error", None)
        except Exception as exc:
            st.session_state["doctor_report_error"] = str(exc)
        st.session_state.doctor_save_message = f"Interaction saved successfully to {saved_path.name}."
        st.rerun()


dashboard = choose_dashboard()

if dashboard == "doctor":
    render_doctor_dashboard()
    st.stop()


all_drugs = get_drug_options()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">💊 Smart Drug Risk Agent</div>
            <div class="sidebar-caption">AI-powered DDI risk prediction, SHAP explanations, SARIMA forecasting, and real alerts.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_dashboard_back_button("user_sidebar")
    st.divider()
    render_user_email_controls()
    st.divider()
    drug_a = st.selectbox("Drug A", all_drugs, index=0)
    drug_b = st.selectbox("Drug B", all_drugs, index=min(1, len(all_drugs) - 1))
    with st.expander("Use custom drug names", expanded=False):
        custom_a = st.text_input("Custom Drug A", "")
        custom_b = st.text_input("Custom Drug B", "")
    run = st.button("Analyze Risk", type="primary", width="stretch")

    st.divider()
    with st.expander("Dataset Description", expanded=False):
        st.write(dataset_description())

drug_a = custom_a.strip() or drug_a
drug_b = custom_b.strip() or drug_b

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Clinical AI Decision Intelligence</div>
        <div class="hero-title">Smart Drug Risk Agent</div>
        <div class="hero-copy">
            A premium dashboard for drug-pair risk classification, model explainability, forecasting,
            scoring, alerting, and adaptive behavior monitoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

err = st.session_state.get("doctor_report_error")
if err:
    st.warning(f"Clinical report pipeline could not complete after the last doctor save: {err}")

clinical_report = st.session_state.get("last_clinical_report") or load_persisted_report()
if clinical_report:
    display_user_dashboard(clinical_report)
else:
    status_box(
        "No clinician report yet",
        "When a doctor saves a reviewed interaction, a structured report (risk, explanation, plan, alternatives) appears here for the patient-facing view.",
        "warning",
    )

if run:
    artifacts = get_artifacts()
    row, row_scaled, prediction = infer_pair(artifacts, drug_a, drug_b)
    
    # 🧠 Self-Learning: Collect prediction for model improvement
    self_learning_engine = get_self_learning_engine()
    self_learning_engine.collect_prediction(
        drug_a=drug_a,
        drug_b=drug_b,
        features=row,
        prediction_label=prediction["label"],
        confidence=prediction["confidence"],
        prediction_probabilities=prediction["probabilities"],
    )
    
    shap_importance = compute_shap_importance(
        artifacts.classifier.model,
        artifacts.scaled_features,
        row_scaled,
        artifacts.feature_columns,
    )
    scores = compute_scores(
        prediction["confidence"],
        shap_importance,
        artifacts.forecast_increasing,
        prediction["label"],
    )

    pair_report = build_pair_report(drug_a, drug_b, artifacts, prediction, shap_importance, scores)
    st.session_state["pair_report_last"] = pair_report

    action = recommended_action(prediction["label"], artifacts.forecast_increasing)
    email_notify_result: dict[str, str] | None = None
    if pair_report.get("risk_level") == "HIGH":
        user_mail = get_user_email()
        if user_mail:
            email_notify_result = send_email(pair_report, user_mail)
        else:
            email_notify_result = {
                "status": "no_recipient",
                "message": "High risk detected. Add your email in the sidebar and save it to receive alerts.",
            }
    else:
        print("Low risk - no email sent")

    label_css = risk_class(prediction["label"])
    total_score = scores["total_score"]
    total_score_css = score_tone(total_score)

    render_html('<div class="section-title">🔍 Input Summary</div>')
    render_html(
        f"""
        <div class="section-card">
            <div class="summary-grid">
                {summary_card("Drug A", drug_a)}
                {summary_card("Drug B", drug_b)}
            </div>
        </div>
        """
    )

    display_pair_friendly_report(pair_report)

    render_html('<div class="section-title">⚠️ Risk Classification</div>')
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Risk Level", prediction["label"])
        render_html(f"<span class='risk-chip {label_css}'>{prediction['label']}</span>")
    with m2:
        st.metric("Confidence", f"{prediction['confidence']:.2%}")
        st.progress(float(prediction["confidence"]))
    with m3:
        st.metric("Total Score", f"{total_score:.2f}/100")
        st.progress(min(max(float(total_score) / 100, 0), 1))
        render_html(f"<span class='risk-chip {total_score_css}'>Score Band</span>")

    render_html('<div class="section-title">Risk Probabilities</div>')
    prob_df = pd.DataFrame({"Risk": list(prediction["probabilities"].keys()), "Probability": list(prediction["probabilities"].values())})
    prob_fig = px.bar(
        prob_df,
        x="Risk",
        y="Probability",
        color="Risk",
        color_discrete_map={"Low Risk": "#34d399", "Medium Risk": "#facc15", "High Risk": "#fb7185"},
    )
    prob_fig.update_traces(marker_line_width=0, opacity=0.92, hovertemplate="%{x}<br>Probability: %{y:.2%}<extra></extra>")
    polish_plotly(prob_fig, height=330)
    st.plotly_chart(prob_fig, width="stretch")

    left, right = st.columns(2)
    with left:
        render_html('<div class="section-title">📊 SHAP Explanation</div>')
        shap_fig = polish_plotly(shap_plot(shap_importance), height=390)
        st.plotly_chart(shap_fig, width="stretch")
        with st.expander("What this means", expanded=False):
            st.write("SHAP shows which features pushed the model decision most strongly for this drug pair.")
    with right:
        render_html('<div class="section-title">📈 Forecast Analysis</div>')
        forecast_fig = polish_plotly(forecast_plot(artifacts.risk_series, artifacts.forecast), height=390)
        forecast_fig.update_traces(line=dict(width=3), mode="lines+markers", hovertemplate="%{x|%b %d, %Y}<br>Risk count: %{y:.3f}<extra></extra>")
        st.plotly_chart(forecast_fig, width="stretch")
        with st.expander("Forecast signal", expanded=False):
            trend = "increasing" if artifacts.forecast_increasing else "not increasing"
            st.write(f"SARIMA forecasts dangerous risk counts over time. Current forecast trend is {trend}.")

    render_html('<div class="section-title">🧮 Scoring System</div>')
    st.progress(min(max(float(total_score) / 100, 0), 1))
    st.dataframe(
        pd.DataFrame([scores]).T.rename(columns={0: "Value"}).style.format("{:.4f}"),
        width="stretch",
    )

    render_html('<div class="section-title">🚨 Email alert status</div>')
    render_html(f"<div class='section-subtitle'>{action}</div>")
    if email_notify_result is None:
        status_box("No high-risk email", "Email alerts are sent only when the summary level is HIGH.", "success")
        st.info("Low or moderate summary level: no alert email is sent.")
    elif email_notify_result.get("status") == "sent":
        status_box("Alert sent", email_notify_result["message"], "danger-box")
        st.success(email_notify_result["message"])
    elif email_notify_result.get("status") == "no_recipient":
        status_box("Email not sent", email_notify_result["message"], "warning")
        st.warning(email_notify_result["message"])
    else:
        status_box("Email not sent", email_notify_result.get("message", "Unknown error"), "warning")
        st.error(email_notify_result.get("message", "Email could not be sent. Check data/smtp_config.json."))

    render_html('<div class="section-title">🔁 Adaptive Behavior</div>')
    with st.expander("Dynamic reclustering status", expanded=True):
        st.write(artifacts.clustering_metadata["message"])
        st.write(artifacts.clustering_metadata["dynamic_reclustering"])
        st.write(
            f"Forecast error = {artifacts.forecast_error:.3f}; threshold = {artifacts.clustering_metadata['forecast_error_threshold']:.3f}."
        )

    # 🧠 Self-Learning Dashboard Section
    render_html('<div class="section-title">🧠 Self-Learning Status</div>')
    learning_status = self_learning_engine.get_summary()
    
    # Display learning metrics
    m_learn1, m_learn2, m_learn3 = st.columns(3)
    with m_learn1:
        st.metric("Total Samples Collected", learning_status["total_samples_collected"])
    with m_learn2:
        st.metric("Uncertain Samples", learning_status["uncertain_samples_count"])
    with m_learn3:
        st.metric("Learning Events Triggered", learning_status["learning_events_triggered"])

    # Display learning readiness
    can_trigger = learning_status["can_trigger_now"]
    trigger_status_color = "success" if can_trigger else "warning"
    samples_collected = learning_status["total_samples_collected"]
    threshold = learning_status["collection_threshold"]
    progress_pct = min(samples_collected / threshold, 1.0)
    
    st.progress(progress_pct)
    
    if can_trigger:
        status_box(
            "✅ Ready to Learn",
            f"Collected {samples_collected}/{threshold} samples. Learning can be triggered.",
            trigger_status_color,
        )
    else:
        status_box(
            "⏳ Learning in Progress",
            f"Collected {samples_collected}/{threshold} samples. Keep analyzing drug pairs to reach threshold.",
            trigger_status_color,
        )

    with st.expander("Self-Learning Details", expanded=False):
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            st.write("**Collection Settings:**")
            st.write(f"- Confidence threshold: {learning_status['confidence_threshold']:.1%}")
            st.write(f"- Collection threshold: {learning_status['collection_threshold']} samples")
            st.write(f"- Session active since: {learning_status['session_active_since']}")
        with col_detail2:
            st.write("**Learning History:**")
            st.write(f"- Total learning events: {learning_status['learning_history_count']}")
            st.write("- Uncertain samples are automatically augmented 3x")
            st.write("- Original data + augmentations enable safe reclustering")

    if can_trigger:
        learning_event = self_learning_engine.trigger_learning_event(
            scaler=artifacts.scaler,
            feature_columns=artifacts.feature_columns,
            run_reclustering_preview=True,
        )
        if learning_event["triggered"] and learning_event["reclustering_preview_ran"]:
            with st.expander("Latest self-learning reclustering preview", expanded=False):
                st.write("DBSCAN + FCM was re-run on collected/augmented samples only.")
                st.write("The production SVM and saved pipeline artifacts were not overwritten.")
                st.write(learning_event["pseudo_label_counts"])
    
    # Auto-persist collected samples (optional, can be disabled in config)
    self_learning_engine.persist_samples()

else:
    status_box("Ready for analysis", "Choose two drugs in the sidebar and click Analyze Risk.", "warning")
    render_html('<div class="section-title">Sectioned System Overview</div>')
    render_html(
        """
        <div class="section-card">
            <div class="summary-grid">
                <div class="overview-item"><b>📊 Dataset</b><br><span class="muted">DDI pairs provide interaction evidence; drug classification data provides clinical context.</span></div>
                <div class="overview-item"><b>🧹 Preprocessing</b><br><span class="muted">Missing values, categorical encoding, cleaning, and StandardScaler are applied.</span></div>
                <div class="overview-item"><b>🧬 Feature Engineering</b><br><span class="muted">Numeric features include interaction frequency, co-occurrence, and similarity score.</span></div>
                <div class="overview-item"><b>🔁 Clustering</b><br><span class="muted">DBSCAN + FCM generate pseudo-labels where direct labels are unavailable.</span></div>
                <div class="overview-item"><b>⚠️ Classification</b><br><span class="muted">SVM with RBF kernel is optimized using GridSearchCV and cross-validation.</span></div>
                <div class="overview-item"><b>📈 Forecasting</b><br><span class="muted">SARIMA forecasts dangerous risk counts and supports dynamic reclustering.</span></div>
                <div class="overview-item"><b>🧮 Scoring</b><br><span class="muted">Score = Confidence x Feature Importance; Total Score = SVM + SHAP + Forecast.</span></div>
                <div class="overview-item"><b>🚨 Alerts</b><br><span class="muted">When the easy-read summary is HIGH, an email goes to your saved address using data/smtp_config.json.</span></div>
            </div>
        </div>
        """
    )
