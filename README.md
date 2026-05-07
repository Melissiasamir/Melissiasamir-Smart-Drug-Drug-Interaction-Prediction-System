# Smart Drug–Drug Interaction Prediction System

## 1. Project title

**Smart Drug–Drug Interaction Prediction System** — clinical decision-support stack for pair-level interaction risk, explainability, time-series signals, and structured user reporting.

---

## 2. Overview

This repository implements an end-to-end **drug–drug interaction (DDI) risk analysis** application. It combines tabular machine learning on curated interaction data, **unsupervised pseudo-labeling** where explicit risk labels are absent, **post-hoc explainability**, **time-series forecasting** over synthesized dangerous-case activity, and an **advanced AI extension** built on **interaction graphs**, **drug embeddings** (optionally graph neural networks), **similarity search**, and a **clinical reporting layer** with optional email alerts.

The system is designed to demonstrate how modern ML pipelines are assembled for healthcare-adjacent use cases: reproducible training, cached artifacts, interactive dashboards, and clear separation between **research-style experimentation** (self-learning preview) and **production inference** paths.

**Why it matters:** preventable adverse drug events remain a major patient-safety problem. Tools that surface risk estimates, drivers of those estimates, temporal context, and human-readable summaries can support—but never replace—qualified clinical judgment and formal knowledge bases.

---

## 3. Key features

- **Drug interaction prediction (SVM)** — RBF Support Vector Machine with `GridSearchCV`, trained on pseudo-labels derived from clustering.
- **Clustering (DBSCAN + Fuzzy C-Means)** — Density-based structure discovery plus soft clustering to map points into **Low / Medium / High Risk** pseudo-labels for supervised training (`utils/clustering.py`).
- **SHAP explainability** — Kernel SHAP (or a deterministic sensitivity fallback) to attribute predictions to input features (`utils/explainability.py`).
- **SARIMA forecasting** — `statsmodels` SARIMAX on a daily dangerous-case series; trend used for scoring, alerts, and dynamic reclustering decisions (`utils/forecasting.py`).
- **Agentic composite scoring** — Five independent signals are combined into one normalized score: class severity, SVM probability gap, FCM membership, SHAP-memory reliability, and ARIMA/SARIMA trend (`utils/scoring.py`).
- **Persistent SHAP memory** — Offline training stores cluster mean SHAP vectors in `models/shap_cluster_memory.joblib`; inference compares new SHAP vectors with cluster memories by cosine similarity (`utils/shap_memory.py`).
- **Borderline RAG retrieval** — Lightweight `NearestNeighbors` retrieval over feature + SHAP vectors retrieves similar historical cases only when the ReAct loop needs evidence (`utils/rag_retrieval.py`).
- **ReAct decision engine** — A transparent Observe → Reason → Act → Observe loop routes each prediction to direct dispatch, SHAP re-check, RAG retrieval, human review, tie handling, no-evidence handling, or low-score rejection (`utils/react_engine.py`).
- **Graph-based drug context** — Interaction pairs are modeled as an undirected graph (`advanced_ai_pipeline/gnn/graph_builder.py`); optional **GCN-style** embeddings when **PyTorch** and **PyTorch Geometric** are installed, otherwise normalized feature-vector embeddings (`advanced_ai_pipeline/gnn/embedder.py`, `model.py`).
- **Embedding clustering & similarity (advanced path)** — `KMeans` over drug embeddings where enough points exist; cosine similarity ranking for alternative drugs (`advanced_ai_pipeline/clustering/embedding_cluster.py`, `similarity/similarity_engine.py`).
- **Dynamic / self-learning behavior** — Collects uncertain inferences plus agentic metadata, augments samples, can run a **non-destructive DBSCAN + FCM preview**, and evaluates drift using both SVM-gap degradation and centroid movement (`utils/self_learning.py`, `utils/drift_detection.py`, surfaced in `app.py`).
- **AI reporting system** — After a doctor saves a reviewed interaction, the pipeline aggregates SVM output, SHAP, five-signal score, SHAP reliability, ReAct decision, forecast direction, production clustering metadata, and advanced similarity into a **structured JSON report** persisted under `data/last_clinical_report.json` (`advanced_ai_pipeline/reporting/`).
- **Email alert systems (two channels)**  
  - **User dashboard HIGH-risk path:** SMTP email to the address stored in `data/user_profile.json` when the aggregated report tier is `HIGH` (`advanced_ai_pipeline/reporting/email_service.py`).  
  - **User “Analyze Risk” path:** SMTP alert to `ALERT_TO` for any prediction that is **not** `Low Risk`, including SHAP/forecast context (`utils/alerts.py`, `app.py`).

---
## 3a. Documentation Structure

| Section | Focus |
|---------|-------|
| **1–2** | Project title & overview |
| **3–3a** | Key features & documentation map |
| **4–12** | Architecture, data pipeline, models, usage, installation |
| **13** | **Self-Learning & Adaptive Learning System** (collection, augmentation, triggers) |
| **14** | **Advanced Features & Integrations** (PyTorch GCN, RDKit, SHAP fallback) |
| **15** | **Dual Dashboard Architecture** (User & Doctor tabs, features) |
| **16** | **Email & Alert Systems** (two channels, SMTP config) |
| **17** | **Structured Clinical Report Generation** (schema, workflow, JSON format) |
| **18** | **Feature Engineering & Data Pipeline** (feature construction, sources) |
| **19** | **Deployment & Configuration** (env vars, first-run, performance) |
| **20** | **Testing & Validation** (self-learning tests, integration) |
| **21** | **Project Structure** (complete directory layout) |
| **22** | Disclaimer |

Agentic AI additions are documented across Sections **4-6**, **11**, **13**, **15**, **19**, and **21** because they are integrated into the existing pipeline rather than implemented as a separate redesign.

---
## 4. System architecture

The system is organized into four conceptual layers:

| Layer | Responsibility |
|--------|----------------|
| **ML layer** | Data loading, cleaning, feature construction, scaling, pseudo-label generation, SVM training/inference, SARIMA training, artifact persistence (`utils/`, `models/pipeline_artifacts.joblib`). |
| **AI / embedding layer** | Graph construction from DDI (+ doctor Excel), name-based feature extraction, GNN or fallback embeddings, embedding clustering, cosine similarity (`advanced_ai_pipeline/gnn/`, `clustering/`, `similarity/`, `doctor_pipeline/`). |
| **Decision / reporting layer** | Agentic composite scoring, SHAP memory reliability, borderline RAG retrieval, ReAct action dispatch, report `generate_report(context)`, persistence, conditional user email (`utils/scoring.py`, `utils/shap_memory.py`, `utils/rag_retrieval.py`, `utils/react_engine.py`, `advanced_ai_pipeline/reporting/`). |
| **User interface** | Streamlit dual-dashboard: **User** (pair analysis, charts, self-learning, clinical report panel, email capture) and **Doctor** (Excel-backed interaction intake) (`app.py`). |

### End-to-end flow (conceptual)

```
Input (drug pair + datasets)
    → Preprocessing & feature engineering
    → Pseudo-labels (DBSCAN + FCM) → SVM training / load cached artifacts
    → Daily risk series → SARIMA → forecast error & trend
    → [Optional dynamic reclustering if forecast error high]
    → Inference: scaled features → SVM probabilities & label
    → SHAP (or fallback) + SHAP memory reliability
    → FCM membership + five-signal agentic score
    → Borderline RAG retrieval + ReAct action dispatch
    → UI: metrics, plots, alerts, self-learning status, drift monitoring

Doctor path (additional):
    → Save row to doctor_added_interactions.xlsx
    → Reporting integration: infer_pair + agentic decision + advanced snapshot
    → generate_report → persist JSON → optional HIGH-risk email to user

User “Analyze Risk” path (additional):
    → Same infer_pair agentic decision loop in-session
    → Optional SMTP alert to ALERT_TO when risk ≠ Low Risk
```

---

## 5. AI components explained

### SVM (classification)

The classifier is a **probability-calibrated RBF SVM** (`sklearn.svm.SVC` inside `GridSearchCV`, `utils/classifier.py`). It predicts **Low Risk**, **Medium Risk**, or **High Risk** for a feature row built for a drug pair. Training labels come from **pseudo-labels**, not raw DDI text labels.

### SHAP (interpretability)

`compute_shap_importance` uses **KernelExplainer** on a background sample of scaled training rows when SHAP is available; on failure it falls back to a **local sensitivity** probe of `predict_proba` (`utils/explainability.py`). Outputs drive the bar chart in the UI and the **key factors** list in the clinical report.

### SARIMA (time-series)

A daily series of **rolling dangerous-case counts** is built from pseudo-labeled high-risk rows with synthetic dates spanning 180 days (`create_daily_risk_series`). **SARIMAX** with order `(1,1,1)` and weekly seasonality `(1,1,1,7)` fits the series; a rolling mean fallback exists if fitting fails (`utils/forecasting.py`). The forecast is compared to recent actuals to set an **increasing / not increasing** flag.

### Agentic scoring, memory, retrieval, and action dispatch

The current inference path centralizes the agentic decision inside `infer_pair` (`utils/pipeline.py`) while preserving the original return shape: `(row, row_scaled, prediction)`. New outputs are attached to the `prediction` dictionary under `agentic_decision`, plus convenience keys such as `scores`, `shap_reliability`, `rag_result`, `react_decision`, and `final_decision`.

**Integration order:**

```
prediction
  -> SHAP explanation
  -> SHAP reliability lookup
  -> fuzzy membership lookup
  -> intelligent scoring
  -> RAG retrieval for borderline/uncertain cases
  -> ReAct reasoning loop
  -> final decision object
```

**Five-signal score (`utils/scoring.py`):**

```
final_agentic_score = 100 * (
    0.30 * class_severity
  + 0.20 * svm_probability_gap
  + 0.20 * fuzzy_membership_of_predicted_class
  + 0.15 * shap_memory_reliability
  + 0.15 * arima_trend_signal
)
```

Backward-compatible score keys are still returned: `total_score`, `svm_score`, `shap_score`, and `forecast_score`.

**Persistent SHAP memory (`utils/shap_memory.py`):**

- During training, representative SHAP vectors are grouped by pseudo-label/cluster.
- Cluster mean SHAP vectors are persisted to `models/shap_cluster_memory.joblib`.
- During inference, the current SHAP vector is compared with cluster means using cosine similarity.
- Reliability is exposed as both a numeric signal and a dashboard-friendly level: `high`, `medium`, or `low`.

**Borderline RAG retrieval (`utils/rag_retrieval.py`):**

- Uses `sklearn.neighbors.NearestNeighbors`.
- Stores retrieval vectors as `[scaled_feature_vector, shap_vector]`.
- Persists the local index to `models/rag_case_index.joblib`.
- Filters weak neighbors using a configurable minimum similarity threshold before majority voting.
- Detects `tie`, `no_evidence`, and `low_confidence` outcomes.

**ReAct engine (`utils/react_engine.py`):**

- Uses an auditable Observe -> Reason -> Act -> Observe trace.
- Routes dynamically using final score, dominant SHAP feature, RAG vote, ARIMA signal, and reliability score.
- Supported routes: `direct_dispatch`, `shap_recheck`, `rag_retrieval`, `human_review`, `tie_handling`, `no_evidence_handling`, and `low_score_rejection`.
- Returns `action_confidence`, `decision_explanation`, `trace`, and `final_decision`.

### Clustering (pattern discovery)

1. **Production pipeline:** tuned **DBSCAN** on scaled pair features, then **Fuzzy C-Means** membership to derive soft risk pseudo-labels (`utils/clustering.py`).  
2. **Advanced pipeline:** **KMeans** (when enough embeddings exist) assigns embedding-space clusters for doctor-processed drugs (`advanced_ai_pipeline/clustering/embedding_cluster.py`).

### GNN (optional)

If `torch` and `torch_geometric` import successfully, `DrugGNN` uses two **GCNConv** layers over `edge_index` from the interaction graph. Otherwise embeddings are **normalized 16-D feature vectors** derived from per-drug name statistics (`advanced_ai_pipeline/gnn/embedder.py`, `model.py`). Embeddings are cached in `advanced_ai_pipeline/data/gnn_embedding_cache.pkl`.

### LLM

**Not used** in this codebase. There is no large-language-model API integration. The ReAct engine is a deterministic, auditable reasoning controller over local model signals; it does not call an external LLM.

---

## 6. Data pipeline

### Inputs

| Source | Path | Role |
|--------|------|------|
| Drug–drug interactions | `data/drug_drug_interactions.csv` | Columns include **Drug 1**, **Drug 2**, **Interaction Description** — builds pair features and graph edges. |
| Drug classification | `data/drug_classification.csv` | Patient/drug context (e.g. age, BP, cholesterol, Na/K, drug category) used for **clinical priors** in feature engineering. |
| Doctor-reviewed interactions | `data/doctor_added_interactions.xlsx` | Optional extra pairs merged into graph building for the advanced pipeline (`InteractionProcessor`). |
| User email (optional) | `data/user_profile.json` | Stores the end-user email for HIGH-tier report alerts. |
| Self-learning buffer (optional) | `data/self_learning_samples.csv` | Persisted uncertain/low-confidence samples for the self-learning engine. |
| SHAP memory artifact | `models/shap_cluster_memory.joblib` | Persisted cluster mean SHAP vectors and reliability thresholds. |
| RAG case index | `models/rag_case_index.joblib` | Persisted nearest-neighbor index over feature + SHAP vectors. |

### Feature processing

`build_features` enriches each DDI row with frequencies, co-occurrence, name similarity, description-derived severity, and classification priors (`utils/feature_engineering.py`). For ad-hoc pairs at inference time, `build_single_pair_features` matches known pairs or blends medians / priors from partial drug presence.

### Model flow

1. Clean DDI + classification tables (`utils/preprocessing.py`).  
2. Build and scale feature matrix; optional row cap via `MAX_TRAIN_ROWS` (default `4000`, `utils/pipeline.py`).  
3. `generate_pseudo_labels` → train SVM → build risk series → SARIMA → evaluate forecast error → optional second clustering + SVM pass.  
4. Build persistent SHAP cluster memory and the lightweight RAG case index for agentic inference.  
5. Serialize `PipelineArtifacts` to `models/pipeline_artifacts.joblib` (versioned payload with cache metadata).

---

## 7. User flow

1. **Launch** the Streamlit app and choose **User Dashboard** or **Dr Dashboard**.  
2. **Doctor dashboard:** enter Drug 1, Drug 2, and an interaction description → submit → row appended to `doctor_added_interactions.xlsx`. The app then runs **`process_doctor_submission`** (reporting integration): `infer_pair` agentic decision on that pair, advanced graph/embedding snapshot, structured report saved to `data/last_clinical_report.json`.  
3. **User dashboard:** optional **Enter your email** + **Save** (validated, persisted). The **Clinical summary report** section loads the latest report from session state or disk after a doctor submission.  
4. **Email (reporting path):** if the report’s `risk_level` is **`HIGH`** and a user email exists, **`send_email`** sends an SMTP message (requires `SMTP_*` and `ALERT_FROM`). For integration testing only, `DDI_SIMULATE_HIGH_RISK=1` forces the report tier to HIGH after the real model run.  
5. **User “Analyze Risk”:** select Drug A / Drug B (or custom names) → **Analyze Risk** runs the main agentic inference visualization (probabilities, SHAP chart, forecast chart, five-signal score, SHAP reliability, ReAct route, RAG outcome, drift alerts, self-learning). Non–Low Risk predictions can trigger **`send_email_alert`** to **`ALERT_TO`** when SMTP variables are set.

---

## 8. Installation

### Prerequisites

- Python **3.10+** recommended (as used with type hints and modern `sklearn` / `pandas`).  
- Optional: **PyTorch** + **PyTorch Geometric** for full GCN embeddings (not listed in `requirements.txt`; install manually if desired).

### Steps

```bash
git clone https://github.com/Melissiasamir/Smart-Drug-Drug-Interaction-Prediction-System.git
cd Smart-Drug-Drug-Interaction-Prediction-System
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Copy `.env` if you use one (the app calls `load_dotenv()` from `python-dotenv`).

---

## 9. Usage

### Run the application

```bash
streamlit run app.py
```

On first run (or if `models/pipeline_artifacts.joblib` is missing or incompatible), the app **trains** the full pipeline and writes artifacts. Later runs **load** cached artifacts for faster startup.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `MAX_TRAIN_ROWS` | Cap rows used for DBSCAN/FCM/SVM training (default `4000`). Must match the value stored in cached artifacts. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_FROM` | Required for **user HIGH-risk** emails and for **`send_email_alert`**. |
| `ALERT_TO` | Recipient for **non–Low Risk** alerts from the **Analyze Risk** flow. |
| `DDI_SIMULATE_HIGH_RISK` | Set to `1` only for **testing** the HIGH-risk user email path (`advanced_ai_pipeline/reporting/integration.py`). |

### Typical tasks

- **Explore risk for a pair:** User dashboard → choose drugs → **Analyze Risk**.  
- **Record a clinician note:** Doctor dashboard → submit form → switch to User dashboard to read the **Clinical summary report**.  
- **Enable alerts:** configure SMTP + `ALERT_FROM` / `ALERT_TO` (analyze path) and/or user email + same SMTP (HIGH-tier user path).

---

## 10. Project structure

```
├── app.py                          # Streamlit entrypoint (dual dashboards)
├── requirements.txt
├── models/
│   └── pipeline_artifacts.joblib   # Cached training payload (created at runtime)
├── data/
│   ├── drug_drug_interactions.csv
│   ├── drug_classification.csv
│   ├── doctor_added_interactions.xlsx   # Doctor submissions (created when used)
│   ├── user_profile.json                # User email (created when saved)
│   ├── last_clinical_report.json        # Latest structured report (created when used)
│   └── self_learning_samples.csv        # Optional persisted learning samples
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py             # DBSCAN + FCM pseudo-labels
│   ├── classifier.py             # SVM train + predict_risk
│   ├── pipeline.py               # train/load/infer_pair
│   ├── explainability.py         # SHAP + plots
│   ├── forecasting.py            # SARIMA + plots
│   ├── scoring.py                # Five-signal agentic composite score
│   ├── shap_memory.py            # Persistent cluster SHAP memory
│   ├── rag_retrieval.py          # Lightweight NearestNeighbors RAG retrieval
│   ├── react_engine.py           # Observe-Reason-Act decision loop
│   ├── drift_detection.py        # Gap + centroid drift gates and rollback helpers
│   ├── alerts.py                 # SMTP alerts (ALERT_TO)
│   └── self_learning.py          # Collection + preview reclustering
├── advanced_ai_pipeline/
│   ├── pipeline.py               # process_doctor_input (full doctor pipeline incl. save)
│   ├── api_handler.py            # Thin wrapper over pipeline.process_doctor_input
│   ├── run_pipeline.py           # CLI-style runner (if used standalone)
│   ├── doctor_pipeline/          # Excel IO, DrugProcessor, DoctorPipeline
│   ├── gnn/                      # Graph, features, embedder, optional GCN model
│   ├── clustering/               # Embedding KMeans clusterer + assigner
│   ├── similarity/               # Cosine similarity engine
│   ├── data/                     # gnn_embedding_cache.pkl, doctor_drug_store.json
│   └── reporting/                # report_generator, integration, user_profile, email_service, user_display
├── SELF_LEARNING_README.md       # Extended documentation for self-learning
└── test_self_learning.py         # Self-learning tests
```

---

## 11. Example output

### Sample SVM prediction (conceptual)

After `infer_pair`, the UI and internal dicts expose values shaped like:

```json
{
  "label": "Medium Risk",
  "confidence": 0.72,
  "probabilities": {
    "Low Risk": 0.08,
    "Medium Risk": 0.72,
    "High Risk": 0.20
  },
  "svm_gap": 0.52,
  "fuzzy_membership": 0.68,
  "scores": {
    "final_agentic_score": 59.4,
    "total_score": 59.4,
    "forecast_score": 83.3
  },
  "final_decision": {
    "route": "rag_retrieval",
    "action_confidence": 0.70,
    "human_review_required": false,
    "decision_explanation": "Additional checks completed; use the selected route outcome."
  }
}
```

### Sample clinical report (`generate_report` / persisted JSON)

Fields are produced by `advanced_ai_pipeline/reporting/report_generator.py`:

```json
{
  "risk_level": "MEDIUM",
  "confidence": 0.72,
  "interpretation": "SVM risk tier: Medium Risk (mapped to MEDIUM). ...",
  "key_factors": [
    "interaction_frequency (importance 0.0412)",
    "clinical_prior_mean (importance 0.0298)"
  ],
  "trend": "Forecast does not show an increasing dangerous-case trend.",
  "treatment_plan": [
    "Review before use: confirm indications, monitor closely, ...",
    "Reconcile the medication list and document the decision in the chart.",
    "..."
  ],
  "alternatives": ["Warfarin", "Clopidogrel"],
  "recommended_action": "Review before use: ...",
  "note": "This report aggregates automated signals ...",
  "drug_1": "Aspirin",
  "drug_2": "Ibuprofen",
  "svm_label": "Medium Risk",
  "total_score": 52.3
}
```

Exact strings and scores depend on data, model seeding, and the submitted pair.

---

## 12. Future improvements

- **Deep learning classifiers** — e.g. graph-aware or transformer encoders over interaction text, with calibrated uncertainty.  
- **REST / FastAPI service** — expose `infer_pair` and reporting as authenticated endpoints for EHR or mobile clients.  
- **Larger, longitudinal real-world datasets** — replace or augment synthetic date logic with true event timestamps and external validation cohorts.  
- **Federated or privacy-preserving learning** — extend the self-learning collector toward multi-site aggregation without centralizing raw PHI.  
- **Formal knowledge integration** — link predictions to curated databases (e.g. DrugBank, ONC high-value data) for traceable citations.

---

## 13. Self-Learning & Adaptive Learning System

The system includes a comprehensive **self-learning module** (`utils/self_learning.py`) that continuously improves through data collection and augmentation:

### Components

| Component | Purpose |
|-----------|---------|
| **SelfLearningCollector** | Automatically captures predictions with drug names, scaled features, labels, and confidence scores after each inference. |
| **Uncertainty Detection** | Flags predictions with confidence < 60% (configurable) for deeper analysis. |
| **DataAugmentor** | Generates synthetic sample variations using Gaussian noise (default ±5% std) to augment training datasets. |
| **Persistent Storage** | Automatically saves collected samples to `data/self_learning_samples.csv` across app sessions. |
| **Learning Event Trigger** | When threshold of 20 samples collected (configurable), triggers safe non-destructive reclustering preview. |
| **Non-Destructive Reclustering** | Runs alternative DBSCAN + FCM clustering on augmented dataset **without replacing** production SVM or artifacts. |
| **Drift Detection Gate** | Opens retraining only when rolling SVM-gap degradation and FCM centroid movement both trigger. |
| **Rollback Protection** | Provides backup, validation, and restore helpers before any future promoted model replaces production artifacts. |

### Integration with UI

- **Self-Learning Status Dashboard** (User tab): Shows total samples collected, uncertain count, learning events triggered, and progress bar toward next trigger.
- **Agentic Monitoring Zones**: Shows Cluster Health, SHAP Reliability, Action Dispatch, Outcome Tracking, and Drift Alerts using live inference values.
- **Automatic Collection**: After every prediction, the engine collects inference outputs transparently.
- **Persistence**: Samples automatically saved after each prediction; loaded on app restart.
- **Testing Support**: `test_self_learning.py` validates collection, augmentation, learning triggers, and CSV persistence.

### Key Methods

```python
# Initialize in app
engine = SelfLearningEngine(config)

# Collect prediction (automatic after infer_pair)
engine.collect_prediction(
    drug_a="Aspirin",
    drug_b="Warfarin",
    features=row_scaled,
    prediction_label="High Risk",
    confidence=0.75,
    prediction_probabilities={"Low Risk": 0.1, "Medium Risk": 0.15, "High Risk": 0.75},
    svm_gap=0.60,
    fuzzy_membership=0.82,
    agentic_score=78.4,
    action_route="direct_dispatch",
)

# Check and trigger learning
if engine.should_trigger_learning():
    df_augmented = engine.prepare_learning_dataset()
    event = engine.trigger_learning_event()

# Persist across sessions
engine.persist_samples()
engine.load_samples()
```

---

## 14. Advanced Features & Optional Integrations

### PyTorch / Torch Geometric GCN Embeddings

- **Runtime Detection**: App auto-detects PyTorch and torch_geometric on startup.
- **GCN Architecture**: Two GCNConv layers (hidden=64, output=32) over interaction graph if libraries available.
- **Graceful Fallback**: Uses normalized 16-D feature vectors (name statistics: length, unique chars, ratios, vowel count, etc.) if PyTorch unavailable.
- **Embedding Cache**: Embeddings persisted to `advanced_ai_pipeline/data/gnn_embedding_cache.pkl` for performance.

### RDKit Cheminformatics

- **SMILES Parsing**: Detects and parses SMILES strings in drug names to extract molecular descriptors.
- **Molecular Features**: Computes molecular weight, logP, atom count when available.
- **Graceful Fallback**: Uses text-only features if RDKit not installed.

### SHAP vs. Sensitivity Fallback

- **Primary**: KernelExplainer with 40-sample background for Shapley values.
- **Fallback**: Deterministic local sensitivity analysis when SHAP unavailable.
- **Output**: Feature importance bar charts and structured report explanations.

---

## 15. Dual Dashboard Architecture

### User Dashboard Features

| Feature | Description |
|---------|-------------|
| **Drug Pair Input** | Autocomplete-enabled text inputs for Drug A and Drug B. |
| **Check Interaction Risk** | Main button triggering full pipeline inference. |
| **Risk Visualization** | Color-coded badge (Low=green, Medium=yellow, High=red) with confidence %. |
| **SHAP Feature Chart** | Top contributing features ranked by importance. |
| **SARIMA Forecast Plot** | Actual vs. predicted dangerous cases with trend indicator. |
| **Score Breakdown** | Transparent five-signal display: severity, SVM gap, FCM membership, SHAP reliability, ARIMA trend, and total score (0-100). |
| **Cluster Health** | DBSCAN outlier count, predicted FCM membership, and baseline SVM gap. |
| **SHAP Reliability** | Cosine similarity against cluster SHAP memory, reliability level, and matched memory label. |
| **Action Dispatch** | ReAct route, RAG vote status, action confidence, and human-review flag. |
| **Outcome Tracking** | Retrieved evidence count, RAG majority label, and vote confidence. |
| **Drift Alerts** | Retrain gate, SVM-gap drop, centroid drift, and rollback-protection notes. |
| **Clinical Summary Report** | Plain-language explanation with "Why?", "What to do?", "Alternatives" sections. |
| **Email Alerts** | Sidebar to capture user email + HIGH-risk alert status. |
| **Analyze Risk Button** | Triggers analysis with optional SMTP alert to ALERT_TO recipient. |
| **Self-Learning Status** | Progress metrics, uncertain samples, learning event count. |

### Doctor Dashboard Features

| Feature | Description |
|---------|-------------|
| **Excel Intake Form** | Input Drug 1, Drug 2, interaction description → save to `doctor_added_interactions.xlsx`. |
| **Advanced AI Processing** | Full pipeline: drug embeddings, KMeans clustering, similarity search. |
| **Similar Drug Recommendations** | Top-5 cosine-similar drugs per submitted drug displayed per cluster. |
| **Doctor Drug Store** | JSON persistence (`doctor_drug_store.json`) of processed drugs, embeddings, cluster IDs. |
| **Graph Update** | Rebuilds interaction graph to include doctor-submitted pairs. |
| **Clinical Report Display** | Shows latest aggregated report from `last_clinical_report.json`. |

---

## 16. Email & Alert Systems (Dual Channels)

### Channel 1: User Dashboard HIGH-Risk Email

**Trigger**: Aggregated report `risk_level` is `HIGH`.

- **User Email Storage**: Captured via UI input → persisted in `data/user_profile.json`.
- **Email Content**: Subject + multi-section body (Why? Actions? Alternatives?).
- **SMTP Config**: Loaded from `data/smtp_config.json` (gitignored). Fields: `host`, `port`, `user`, `password`, `from_email`.
- **Gmail Support**: Explicit documentation for app-specific passwords.
- **Validation**: Checks all required fields; returns clear error messages if misconfigured.

**Example env variables:**
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_FROM=your_email@gmail.com
```

### Channel 2: Analyze Risk Path Email

**Trigger**: User runs "Analyze Risk" and prediction is NOT "Low Risk".

- **Alert Recipient**: Configured via `ALERT_TO` env variable.
- **Email Content**: Drug pair, risk level, confidence, total score, action items, explanation.
- **Implementation**: `utils/alerts.py::send_email_alert`.

**Example:**
```
ALERT_TO=alerts@medical-team.com
DDI_SIMULATE_HIGH_RISK=1  # (testing only) force HIGH tier after real model
```

---

## 17. Structured Clinical Report Generation

**Location**: `advanced_ai_pipeline/reporting/report_generator.py`

**Report Persistence**: `data/last_clinical_report.json`

### JSON Schema

```json
{
  "drug_1": "Aspirin",
  "drug_2": "Ibuprofen",
  "risk_level": "MEDIUM",
  "svm_label": "Medium Risk",
  "confidence": 0.72,
  "total_score": 52.3,
  "probabilities": {
    "Low Risk": 0.08,
    "Medium Risk": 0.72,
    "High Risk": 0.20
  },
  "interpretation": "SVM risk tier: Medium Risk (mapped to MEDIUM). Forecast shows no increasing trend...",
  "key_factors": [
    "interaction_frequency (importance 0.0412)",
    "clinical_prior_mean (importance 0.0298)"
  ],
  "trend": "Forecast does not show an increasing dangerous-case trend.",
  "treatment_plan": [
    "Review before use: confirm indications, monitor closely...",
    "Reconcile the medication list...",
    "Document decision in chart"
  ],
  "alternatives": ["Warfarin", "Clopidogrel"],
  "recommended_action": "Review before use: ...",
  "note": "This report aggregates automated signals..."
}
```

### Report Workflow

1. **Doctor submits pair** or **User runs "Analyze Risk"**.
2. **Full pipeline runs**: SVM prediction → SHAP → SHAP memory reliability → FCM membership → five-signal scoring → borderline RAG → ReAct final decision.
3. **`generate_report(context)` aggregates** all outputs into structured JSON.
4. **Report persisted** to disk.
5. **If HIGH risk + user email exists** → SMTP alert triggered.
6. **Report displayed** in UI (User or Doctor dashboard).

---

## 18. Feature Engineering & Data Pipeline

### Input Sources

| Source | Path | Role |
|--------|------|------|
| Drug–drug interactions | `data/drug_drug_interactions.csv` | Pair-level interaction descriptions, frequency signals. |
| Drug classification | `data/drug_classification.csv` | Patient/drug context (age, BP, cholesterol, Na/K) for clinical priors. |
| Doctor interactions | `data/doctor_added_interactions.xlsx` | Clinician-submitted pairs integrated into graphs. |
| User profile | `data/user_profile.json` | End-user email for HIGH-risk alerts. |
| Self-learning samples | `data/self_learning_samples.csv` | Persisted uncertain predictions for augmentation. |

### Feature Construction (`utils/feature_engineering.py`)

- **Pair Frequency**: Total interactions per drug.
- **Co-occurrence**: Specific pair count in dataset.
- **Neighborhood Similarity**: Jaccard index of drug interaction partners.
- **Clinical Keywords**: Severity score from description (toxicity=2.3, bleeding=2.4, etc.).
- **Clinical Priors**: Risk weights from classification data (age, BP, cholesterol).
- **Name Similarity**: SequenceMatcher-based text similarity.
- **Fallback Logic**: For unseen pairs, uses medians/priors from partial drug presence.

### Feature Matrix & Scaling

- **All numeric features** normalized via `StandardScaler`.
- **Categorical fields** (Sex, BP, Cholesterol) label-encoded.
- **Scaler persisted** in pipeline artifacts for consistent inference-time scaling.

---

## 19. Deployment & Configuration

### Environment Variables (Optional)

```bash
# SMTP for HIGH-risk user alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_FROM=your_email@gmail.com

# Alert recipient for "Analyze Risk" path
ALERT_TO=alerts@medical-team.com

# Model tuning (must match saved artifacts)
MAX_TRAIN_ROWS=4000

# Testing only: force HIGH-risk tier
DDI_SIMULATE_HIGH_RISK=1
```

### First Run Behavior

1. On startup, `app.py` checks for `models/pipeline_artifacts.joblib`.
2. If **missing or incompatible** (version/MAX_TRAIN_ROWS mismatch) → **full pipeline trains**.
3. Training includes: data cleaning, feature engineering, DBSCAN/FCM pseudo-labels, SVM training, SARIMA fitting, SHAP memory creation, and RAG case-index creation.
4. Artifacts serialized; cached for subsequent runs.
5. **Subsequent runs** load cached pipeline for ~2-5 second startup.

### Runtime artifacts

| Artifact | Purpose |
|----------|---------|
| `models/pipeline_artifacts.joblib` | Versioned production pipeline: scaler, clustering metadata, SVM, forecast, SHAP memory reference, RAG index reference. |
| `models/shap_cluster_memory.joblib` | Persistent cluster mean SHAP vectors, case SHAP vectors, cluster counts, and reliability thresholds. |
| `models/rag_case_index.joblib` | Local `NearestNeighbors` index over feature + SHAP vectors for borderline retrieval. |
| `data/self_learning_samples.csv` | Appended inference samples with prediction metadata, SVM gap, fuzzy membership, agentic score, and action route. |

### Performance Optimization

- **Joblib Caching**: Production model loaded from disk artifact on every app run.
- **Streamlit @st.cache_resource**: SelfLearningEngine, datasets, pipeline cached across widget interactions.
- **Embedding Cache**: GNN embeddings pre-computed and cached to avoid recomputation.
- **Agentic Cache**: SHAP memory and RAG index are built during training and loaded with `PipelineArtifacts` for inference.

---

## 20. Testing & Validation

### Self-Learning Module Tests (`test_self_learning.py`)

- **Smoke tests** for collection, uncertainty detection, augmentation, learning triggers.
- **Synthetic dataset generation** for isolated testing.
- **Persistence validation**: CSV save/load verified.
- **Augmentation verification**: Confirms larger dataset with original features.
- **Non-destructive verification**: Confirms production artifacts unchanged.

### Integration Testing

- **DDI_SIMULATE_HIGH_RISK=1**: Manually test HIGH-risk email path without real data.
- **Manual doctor submission**: Submit via Doctor tab, verify report persisted and displayed.
- **Manual email validation**: Configure SMTP, trigger Analyze Risk with non-Low Risk prediction.
- **Agentic decision smoke test**: Run `infer_pair(...)` and verify `prediction["agentic_decision"]`, `prediction["scores"]`, `prediction["shap_reliability"]`, `prediction["rag_result"]`, and `prediction["final_decision"]` are present.
- **RAG evidence quality**: Check that weak neighbors below `RetrievalConfig.min_similarity` appear in `rejected_cases` and do not affect majority voting.
- **Drift gate validation**: Confirm retraining is not triggered unless both rolling SVM-gap degradation and centroid movement are above configured thresholds.

---

## 21. Project Structure (Complete)

```
├── app.py                                  # Streamlit entrypoint (dual dashboards)
├── requirements.txt                        # Dependencies (streamlit, pandas, sklearn, etc.)
├── SELF_LEARNING_README.md                 # Extended self-learning documentation
├── test_self_learning.py                   # Self-learning module tests
├── models/
│   ├── pipeline_artifacts.joblib           # Cached training payload
│   ├── shap_cluster_memory.joblib          # Cluster SHAP memory artifact
│   └── rag_case_index.joblib               # Borderline retrieval index
├── data/
│   ├── drug_drug_interactions.csv          # Base DDI dataset
│   ├── drug_classification.csv             # Patient/drug context
│   ├── doctor_added_interactions.xlsx      # Doctor submissions (created at runtime)
│   ├── user_profile.json                   # User email (created when saved)
│   ├── last_clinical_report.json           # Latest report (created when generated)
│   ├── self_learning_samples.csv           # Collected uncertain samples
│   └── smtp_config.example.json            # SMTP config template
├── utils/
│   ├── data_loader.py                      # Load DDI + classification CSV
│   ├── preprocessing.py                    # Clean, encode, normalize data
│   ├── feature_engineering.py              # Build pair-level features
│   ├── clustering.py                       # DBSCAN + FCM pseudo-labels
│   ├── classifier.py                       # SVM train + predict_risk
│   ├── pipeline.py                         # Main orchestration (train/load/infer)
│   ├── explainability.py                   # SHAP + fallback sensitivity
│   ├── forecasting.py                      # SARIMA daily risk series
│   ├── scoring.py                          # Five-signal agentic scoring system
│   ├── shap_memory.py                      # Persistent cluster SHAP memory
│   ├── rag_retrieval.py                    # Lightweight RAG retrieval
│   ├── react_engine.py                     # ReAct action dispatch
│   ├── drift_detection.py                  # Drift gates + rollback helpers
│   ├── alerts.py                           # SMTP alerts (Analyze Risk path)
│   └── self_learning.py                    # Collection + augmentation + learning
├── advanced_ai_pipeline/
│   ├── pipeline.py                         # Doctor pipeline orchestration
│   ├── api_handler.py                      # Thin API wrapper
│   ├── run_pipeline.py                     # CLI runner (optional standalone)
│   ├── doctor_pipeline/
│   │   ├── __init__.py
│   │   ├── doctor_handler.py               # Doctor submission processing
│   │   ├── drug_processor.py               # Individual drug processing
│   │   └── interaction_processor.py        # Interaction integration
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── graph_builder.py                # Build interaction graph
│   │   ├── features.py                     # Node feature generation
│   │   ├── embedder.py                     # GCN or fallback embeddings
│   │   └── model.py                        # GCN model definition
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── embedding_cluster.py            # KMeans clustering + assignment
│   │   └── cluster_assigner.py             # Map drugs to clusters
│   ├── similarity/
│   │   ├── __init__.py
│   │   └── similarity_engine.py            # Cosine similarity search
│   ├── data/
│   │   ├── doctor_drug_store.json          # Processed drug persistence
│   │   └── gnn_embedding_cache.pkl         # Cached embeddings
│   └── reporting/
│       ├── __init__.py
│       ├── report_generator.py             # Aggregate + JSON report
│       ├── integration.py                  # Doctor submission → report
│       ├── user_profile.py                 # User email management
│       ├── email_service.py                # HIGH-risk email (user channel)
│       ├── smtp_client.py                  # SMTP send implementation
│       └── user_display.py                 # UI rendering functions
└── README.md                               # This file
```

---

## 22. Disclaimer

**This system is for research and educational purposes only.** It is not a medical device, not FDA-cleared or CE-marked software, and must not be used as the sole basis for prescribing, deprescribing, or changing patient care. Drug interaction risk is context-dependent (dose, organ function, genetics, comorbidities, and co-medications). Always consult qualified healthcare professionals and authoritative drug information resources before making clinical decisions.
