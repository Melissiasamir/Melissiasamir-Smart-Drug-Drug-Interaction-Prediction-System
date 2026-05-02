# Smart Drug Risk Agent

Production-style AI project for drug-drug interaction risk analysis.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first risk analysis trains the pipeline if `models/pipeline_artifacts.joblib`
does not exist yet. After that, the app loads the saved model from `models/`
instead of retraining on every Streamlit restart.

## SMTP Alert Configuration

Set these environment variables to enable real alerts:

```bash
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_user
SMTP_PASSWORD=your_password
ALERT_FROM=alerts@example.com
ALERT_TO=clinician@example.com
```

The alert is sent immediately for any actionable risk prediction, meaning any prediction other than `Low Risk`.
The email includes the risk level, confidence, total score, model explanation, and recommended action.

### Gmail SMTP Example

Copy `.env.example` to `.env`, then use:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_16_character_app_password
ALERT_FROM=your_email@gmail.com
ALERT_TO=clinician@example.com
```

For Gmail, `SMTP_PASSWORD` must be an App Password, not your normal Gmail password:

1. Enable 2-Step Verification on your Google account.
2. Open Google Account > Security > App passwords.
3. Create an app password for Mail.
4. Paste the generated 16-character password into `.env`.

Never commit `.env` or hardcode SMTP credentials in source code.
