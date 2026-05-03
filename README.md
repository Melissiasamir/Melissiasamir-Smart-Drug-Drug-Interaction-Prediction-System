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
- **Graph-based drug context** — Interaction pairs are modeled as an undirected graph (`advanced_ai_pipeline/gnn/graph_builder.py`); optional **GCN-style** embeddings when **PyTorch** and **PyTorch Geometric** are installed, otherwise normalized feature-vector embeddings (`advanced_ai_pipeline/gnn/embedder.py`, `model.py`).
- **Embedding clustering & similarity (advanced path)** — `KMeans` over drug embeddings where enough points exist; cosine similarity ranking for alternative drugs (`advanced_ai_pipeline/clustering/embedding_cluster.py`, `similarity/similarity_engine.py`).
- **Dynamic / self-learning behavior** — Collects uncertain inferences, augments samples, and can run a **non-destructive DBSCAN + FCM preview** without overwriting the saved production SVM or `pipeline_artifacts.joblib` (`utils/self_learning.py`, surfaced in `app.py`).
- **AI reporting system** — After a doctor saves a reviewed interaction, the pipeline aggregates SVM output, SHAP, forecast direction, production clustering metadata, and advanced similarity into a **structured JSON report** persisted under `data/last_clinical_report.json` (`advanced_ai_pipeline/reporting/`).
- **Email alert systems (two channels)**  
  - **User dashboard HIGH-risk path:** SMTP email to the address stored in `data/user_profile.json` when the aggregated report tier is `HIGH` (`advanced_ai_pipeline/reporting/email_service.py`).  
  - **User “Analyze Risk” path:** SMTP alert to `ALERT_TO` for any prediction that is **not** `Low Risk`, including SHAP/forecast context (`utils/alerts.py`, `app.py`).

---

## 4. System architecture

The system is organized into four conceptual layers:

| Layer | Responsibility |
|--------|----------------|
| **ML layer** | Data loading, cleaning, feature construction, scaling, pseudo-label generation, SVM training/inference, SARIMA training, artifact persistence (`utils/`, `models/pipeline_artifacts.joblib`). |
| **AI / embedding layer** | Graph construction from DDI (+ doctor Excel), name-based feature extraction, GNN or fallback embeddings, embedding clustering, cosine similarity (`advanced_ai_pipeline/gnn/`, `clustering/`, `similarity/`, `doctor_pipeline/`). |
| **Decision / reporting layer** | Composite scoring, SHAP tables, report `generate_report(context)`, persistence, conditional user email (`utils/scoring.py`, `advanced_ai_pipeline/reporting/`). |
| **User interface** | Streamlit dual-dashboard: **User** (pair analysis, charts, self-learning, clinical report panel, email capture) and **Doctor** (Excel-backed interaction intake) (`app.py`). |

### End-to-end flow (conceptual)

```
Input (drug pair + datasets)
    → Preprocessing & feature engineering
    → Pseudo-labels (DBSCAN + FCM) → SVM training / load cached artifacts
    → Daily risk series → SARIMA → forecast error & trend
    → [Optional dynamic reclustering if forecast error high]
    → Inference: scaled features → SVM probabilities & label
    → SHAP (or fallback) + composite scores
    → UI: metrics, plots, alerts, self-learning status

Doctor path (additional):
    → Save row to doctor_added_interactions.xlsx
    → Reporting integration: infer_pair + SHAP + scores + advanced snapshot
    → generate_report → persist JSON → optional HIGH-risk email to user

User “Analyze Risk” path (additional):
    → Same infer_pair / SHAP / forecast / scoring loop in-session
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

### Clustering (pattern discovery)

1. **Production pipeline:** tuned **DBSCAN** on scaled pair features, then **Fuzzy C-Means** membership to derive soft risk pseudo-labels (`utils/clustering.py`).  
2. **Advanced pipeline:** **KMeans** (when enough embeddings exist) assigns embedding-space clusters for doctor-processed drugs (`advanced_ai_pipeline/clustering/embedding_cluster.py`).

### GNN (optional)

If `torch` and `torch_geometric` import successfully, `DrugGNN` uses two **GCNConv** layers over `edge_index` from the interaction graph. Otherwise embeddings are **normalized 16-D feature vectors** derived from per-drug name statistics (`advanced_ai_pipeline/gnn/embedder.py`, `model.py`). Embeddings are cached in `advanced_ai_pipeline/data/gnn_embedding_cache.pkl`.

### LLM

**Not used** in this codebase. There is no large-language-model API integration.

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

### Feature processing

`build_features` enriches each DDI row with frequencies, co-occurrence, name similarity, description-derived severity, and classification priors (`utils/feature_engineering.py`). For ad-hoc pairs at inference time, `build_single_pair_features` matches known pairs or blends medians / priors from partial drug presence.

### Model flow

1. Clean DDI + classification tables (`utils/preprocessing.py`).  
2. Build and scale feature matrix; optional row cap via `MAX_TRAIN_ROWS` (default `4000`, `utils/pipeline.py`).  
3. `generate_pseudo_labels` → train SVM → build risk series → SARIMA → evaluate forecast error → optional second clustering + SVM pass.  
4. Serialize `PipelineArtifacts` to `models/pipeline_artifacts.joblib` (versioned payload with cache metadata).

---

## 7. User flow

1. **Launch** the Streamlit app and choose **User Dashboard** or **Dr Dashboard**.  
2. **Doctor dashboard:** enter Drug 1, Drug 2, and an interaction description → submit → row appended to `doctor_added_interactions.xlsx`. The app then runs **`process_doctor_submission`** (reporting integration): SVM/SHAP/scores on that pair, advanced graph/embedding snapshot, structured report saved to `data/last_clinical_report.json`.  
3. **User dashboard:** optional **Enter your email** + **Save** (validated, persisted). The **Clinical summary report** section loads the latest report from session state or disk after a doctor submission.  
4. **Email (reporting path):** if the report’s `risk_level` is **`HIGH`** and a user email exists, **`send_email`** sends an SMTP message (requires `SMTP_*` and `ALERT_FROM`). For integration testing only, `DDI_SIMULATE_HIGH_RISK=1` forces the report tier to HIGH after the real model run.  
5. **User “Analyze Risk”:** select Drug A / Drug B (or custom names) → **Analyze Risk** runs the main pipeline visualization (probabilities, SHAP chart, forecast chart, scores, alert status, self-learning). Non–Low Risk predictions can trigger **`send_email_alert`** to **`ALERT_TO`** when SMTP variables are set.

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
│   ├── scoring.py
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

## 13. Disclaimer

**This system is for research and educational purposes only.** It is not a medical device, not FDA-cleared or CE-marked software, and must not be used as the sole basis for prescribing, deprescribing, or changing patient care. Drug interaction risk is context-dependent (dose, organ function, genetics, comorbidities, and co-medications). Always consult qualified healthcare professionals and authoritative drug information resources before making clinical decisions.

---

## 14. Author section

**Smart Drug–Drug Interaction Prediction System** — repository maintained by **[Melissiasamir](https://github.com/Melissiasamir)** and contributors.  
If you extend or cite this work in academic or portfolio contexts, please reference this repository and clearly describe which components (baseline ML vs. advanced AI vs. reporting) were used or modified.
