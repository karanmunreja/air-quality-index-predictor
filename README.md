# AQI Predictor for Lahore

An end-to-end, multi-horizon AQI forecasting project built by **Karan, Data Science Intern at 10 Pearls**. The system predicts Lahore's US AQI 24, 48, and 72 hours ahead using environmental and pollutant data, Hopsworks, FastAPI, and Streamlit.

## 🔗 Live Demo

**Dashboard:** https://air-quality-index-predictor-2pm6dxb7hdr7roz5jafdvp.streamlit.app/
**API (Swagger docs):** https://air-quality-index-predictor-three.vercel.app/

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

The dashboard and API are deployed as **two independent services** — Streamlit Cloud (frontend) and Vercel (FastAPI backend) — connected over HTTPS rather than running as a single combined process. See [Deployment](#deployment) below.

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

Both services run as separate processes and talk to each other over `localhost` in this mode.

Start the API in one terminal:

```bash
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in a second terminal:

```bash
streamlit run streamlit_app.py
```

The dashboard loads the forecast first, then shows EDA and SHAP below it. The refresh button updates only the live forecast. If Streamlit reports it can't reach `localhost:8000`, confirm the FastAPI process above is still running.

## Deployment

In production, the dashboard and API run on separate platforms and are connected by environment variables instead of `localhost`.

| Service | Platform | Key files |
|---|---|---|
| Dashboard (`streamlit_app.py`) | Streamlit Cloud | `requirements-streamlit.txt`, `runtime.txt` |
| API (`app.py`) | Vercel | `requirements-api.txt`, `vercel.json` |

### Environment variables

| Variable | Set where | Purpose |
|---|---|---|
| `AQI_API_BASE_URL` | Streamlit Cloud → Settings → Secrets | Base URL of the deployed FastAPI backend (e.g. `https://your-project.vercel.app`). Falls back to `http://localhost:8000` if unset, which only works locally. |
| `HOPSWORKS_API_KEY` | Vercel → Project Settings → Environment Variables (Sensitive) | Authenticates the API against the Hopsworks feature store and model registry. |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | Vercel → Project Settings → Environment Variables | Set to `1`. Required because the API's dependency bundle (mainly the `hopsworks` package) exceeds Vercel's standard function size limit; also requires Fluid Compute enabled under Project Settings → Functions. |

### Deploying the API (Vercel)

1. Import this repository into Vercel with Root Directory left blank (repo root).
2. Override the Install Command to `pip install -r requirements-api.txt`.
3. Set `HOPSWORKS_API_KEY` and `VERCEL_SUPPORT_LARGE_FUNCTIONS` as above, and confirm Fluid Compute is enabled.
4. Deploy, then verify `/health` and `/docs` on the resulting URL before wiring up the dashboard.

### Deploying the dashboard (Streamlit Cloud)

1. Deploy this repository on [share.streamlit.io](https://share.streamlit.io), pointing at `streamlit_app.py`.
2. `runtime.txt` pins the Python version so the deployment doesn't drift onto an incompatible newer default.
3. Set `AQI_API_BASE_URL` in the app's Secrets, pointing at the live Vercel URL from above.

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
- `app.py`: FastAPI application, deployed on Vercel.
- `streamlit_app.py`: user dashboard, deployed on Streamlit Cloud.
- `requirements-api.txt`: slim dependency list used for the Vercel deployment (excludes training-only packages).
- `runtime.txt`: pins the Python version for Streamlit Cloud.
- `vercel.json`: Vercel build/routing configuration.
- `.github/workflows/`: hourly ingestion and daily retraining automation.

## GitHub Actions secrets

Set `HOPSWORKS_API_KEY` in the repository's **Settings → Secrets and variables → Actions** before enabling the scheduled feature and training workflows.
