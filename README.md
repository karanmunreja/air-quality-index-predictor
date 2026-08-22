# AQI Predictor for Lahore

An end-to-end, multi-horizon AQI forecasting project built by **Karan, Data Science Intern**. The system predicts Lahore’s US AQI 24, 48, and 72 hours ahead using environmental and pollutant data, Hopsworks, FastAPI, and Streamlit.

## What it does

- Collects weather and air-quality observations from Open-Meteo.
- Builds time, lag, and change-rate features.
- Stores historical and latest features in Hopsworks.
- Compares Random Forest, Ridge, XGBoost, and LSTM models.
- Registers and deploys the best model by mean R².
- Shows a live forecast, health alerts, EDA, and per-horizon SHAP explanations.

The currently logged experiments favour **Ridge** as the best balanced model across all three horizons. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the complete analysis.

## Architecture

```text
Open-Meteo APIs -> feature engineering -> Hopsworks feature store
                                           |-> train / compare / register / deploy
Streamlit <- FastAPI <- latest feature view + Hopsworks model serving
```

## Setup

Create and activate the backend/training environment. If your pip mirror cannot find SHAP, force the PyPI index as shown below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

For a separate lightweight Streamlit environment:

```bash
python3 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --index-url https://pypi.org/simple -r requirements-streamlit.txt
```

Copy the configuration template and add your real Hopsworks key:

```bash
cp .env.example .env
```

`HOPSWORKS_API_KEY` is required. Never commit `.env`.

## Run locally

Start the API in one terminal:

```bash
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in a second terminal:

```bash
streamlit run streamlit_app.py
```

The dashboard loads the forecast first, then shows EDA and SHAP below it. The refresh button updates only the live forecast.

## Pipelines

```bash
# Fetch the latest hourly observation and update feature groups
python3 -m src.data.pipelines.hourly_pipeline

# Train candidates, register the best one, and deploy it
python3 -m src.data.pipelines.daily_training_pipeline

```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check. |
| `POST /predict` | Latest 24/48/72-hour AQI forecast. |
| `GET /history?days=30` | Historical feature rows for EDA. |
| `POST /explain` | SHAP feature contributions for every forecast horizon. |

## Repository guide

- `src/data/`: Open-Meteo clients, merging, feature engineering, and scheduled data pipelines.
- `src/config.py`: environment-based settings.
- `models/`: preprocessing, candidate model training, evaluation, and registry integration.
- `app.py`: FastAPI application.
- `streamlit_app.py`: user dashboard.
- `.github/workflows/`: hourly ingestion and daily retraining automation.

## GitHub Actions secrets

Set `HOPSWORKS_API_KEY` in the repository’s **Settings → Secrets and variables → Actions** before enabling the scheduled feature and training workflows.
