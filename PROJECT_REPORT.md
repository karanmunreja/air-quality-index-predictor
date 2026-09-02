# AQI Predictor for Lahore — Project Report

**Developer:** Karan, Data Science Intern at 10 Pearls
**Project:** AQI Predictor for Lahore
**Report scope:** End-to-end implementation, as represented by the repository at the time of this report.

## 🔗 Live Demo

**Dashboard:** https://air-quality-index-predictor-2pm6dxb7hdr7roz5jafdvp.streamlit.app/
**API (Swagger docs):** https://air-quality-index-predictor-three.vercel.app/

*Replace both links above with the current live URLs before publishing.*

## 1. Executive summary

This project is an end-to-end air-quality forecasting system for **Lahore, Pakistan**. It ingests weather and air-pollution data from Open-Meteo, creates hourly predictive features, stores them in Hopsworks, trains and compares multi-output machine-learning models, registers and deploys the preferred model, and presents forecast, exploratory data analysis (EDA), and explainability in a Streamlit dashboard.

The forecasting task predicts the **US AQI 24, 48, and 72 hours ahead** in a single model call. The current comparison results show that **Ridge** is the strongest all-round tabular model in the logged experiments, with an average R² of approximately **0.630**. XGBoost has the best 24-hour result, but Ridge is more reliable for the longer 48- and 72-hour horizons.

The delivered application has three main user-facing capabilities:

1. A live AQI forecast with visual AQI bands, health guidance, a persistent severity alert, and a forecast-only refresh control.
2. EDA using the historical Hopsworks feature group: data-quality summary, AQI trend, pollutant overlays, correlation matrix, and feature-versus-AQI exploration.
3. SHAP-based local model explanations for each forecast horizon, using the same latest online feature row used for live prediction.

The system is deployed as **two independently hosted services** — a Streamlit Cloud dashboard and a Vercel-hosted FastAPI backend — communicating over HTTPS. Section 4.1 and Section 12 describe this in detail.

## 2. Experimentation

The repository now contains an explicit, linear experiment script in `main.py` that
is written to reflect the step-by-step learning process used during the
internship. Unlike previous guarded versions, this script intentionally raises
errors if environment pieces are missing so reviewers can see the exact calls
made during exploration. The script performs the following:

- Loads `model_comparison_results.csv` if present; otherwise fetches a small
       sample of hourly AQI/PM data from Open-Meteo and saves it as `sample_aqi.csv`.
- Performs basic EDA: head, dtypes, missing-value counts, numeric summary,
       and correlations vs. AQI.
- Constructs simple time features (`hour`, `weekday`) and lag features
       (`aqi_lag1`, `aqi_lag24`) and fills small gaps deterministically.
- Shows how to call the project's Hopsworks feature-store client explicitly
       via `src.data.features.feature_store.hopswork_client.get_latest_feature_view()`
       (the call is present in the script and will error if Hopsworks is not set up).
- Trains two baseline models (Ridge and RandomForest), prints RMSE/MAE/R²,
       and saves a demo Ridge model and scaler under `saved_models/`.

This change documents the exploratory workflow and provides a reproducible
starter flow for reviewers who want to run the experiments locally.

## 3. Business problem and objective

Lahore can experience substantial variation in pollution levels. A short-horizon AQI forecast can support safer choices about outdoor exercise, commuting, school activities, and health precautions.

The project objective is to build a reproducible system that:

- collects hourly environmental and pollutant observations for Lahore;
- predicts future AQI at 24, 48, and 72 hours;
- identifies the best model using common regression metrics;
- deploys the selected model through Hopsworks Model Serving;
- provides a non-technical dashboard with actionable health context; and
- explains which features caused an individual forecast to increase or decrease.

## 4. Solution architecture

```text
Open-Meteo Weather API ─┐
                        ├─> merge + feature engineering ─> Hopsworks feature groups
Open-Meteo AQI API ─────┘                                      │
                                                               ├─> model training and comparison
                                                               │       └─> Hopsworks Model Registry/Serving
                                                               │
                                                               └─> latest online feature view
                                                                           │
Streamlit dashboard <─ FastAPI (`app.py`) <─ Hopsworks serving endpoint ──┘
       │                    │
       ├─ EDA (`/history`) ─┘ uses historical training feature group
       └─ SHAP (`/explain`) ─ uses registered model + historical background sample
```

Two Hopsworks data paths are intentionally used:

- **Historical training feature group:** used for model training and EDA.
- **Latest serving feature view:** contains the most recent Lahore feature row and is used by `/predict` and `/explain`.

This separation prevents the dashboard from using a one-row serving table as its historical analysis dataset.

### 4.1 Deployment topology

The dashboard and API are deployed as two independently hosted services rather than a single combined process:

```text
┌─────────────────────────┐        HTTPS         ┌──────────────────────────┐
│   Streamlit Cloud        │  ─────────────────>  │   Vercel                 │
│   streamlit_app.py       │   AQI_API_BASE_URL    │   app.py (FastAPI)       │
│   (frontend/dashboard)   │  <─────────────────  │   (backend API)          │
└─────────────────────────┘        JSON            └──────────────┬───────────┘
                                                                    │
                                                                    ▼
                                                          Hopsworks Feature
                                                          Store / Model Serving
```

Key implementation points:

- The Streamlit app reads the backend's base URL from the `AQI_API_BASE_URL` environment variable (`os.getenv("AQI_API_BASE_URL", "http://localhost:8000")` in `streamlit_app.py`), defaulting to `localhost:8000` for local development only. In production this is set as a Streamlit Cloud secret pointing at the Vercel deployment.
- The FastAPI backend enables `CORSMiddleware` (`allow_origins=["*"]`) so that requests from the Streamlit Cloud origin are accepted.
- Vercel's Python runtime is serverless: functions cold-start rather than running as a single persistent process. This means the in-memory `@lru_cache` on `load_explanation_model()` and `latest_feature_view()` in `app.py` does not persist as reliably across invocations as it would on a long-running server, which can make `/explain` noticeably slower on a cold start than on a traditional host.
- Because Vercel enforces a function bundle size limit, the backend uses a slimmer dependency list for deployment (excluding training-only packages such as `tensorflow`, `xgboost`, and `scikit-learn`, which are not imported by `app.py`). Vercel's Large Functions beta (`VERCEL_SUPPORT_LARGE_FUNCTIONS=1`, requiring Fluid Compute) is enabled to accommodate the remaining size of the `hopsworks` dependency tree.

## 5. Data acquisition and preparation

### 5.1 Data sources

The project uses Open-Meteo services:

- **Geocoding API:** resolves Lahore into latitude and longitude.
- **Forecast/weather API:** collects hourly temperature, humidity, surface pressure, wind speed, and wind direction.
- **Archive weather API:** retrieves historical weather for the batch feature pipeline.
- **Air Quality API:** retrieves PM10, PM2.5, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone, and US AQI.

### 5.2 Feature construction

The raw hourly weather and AQI payloads are merged by time position. The resulting record is converted to a project feature record with city and AQI fields. Time components are then derived:

- year, month, day, hour, minute;
- weekday, later encoded from Monday–Sunday to 0–6;
- AQI lags at 1, 2, 3, 5, 6, 8, 10, 12, 24, 30, 36, 42, 48, 54, 60, and 72 hours;
- AQI, PM2.5, and PM10 change-rate features.

The supervised targets are produced by shifting AQI forward by 24, 48, and 72 rows: `target_aqi_24`, `target_aqi_48`, and `target_aqi_72`.

### 5.3 Handling history for online features

The hourly pipeline reads the last **76 hours** of historical data. This provides enough context for the largest 72-hour lag plus the current observation. It combines the new record with history, prevents duplicate timestamps, re-engineers the features, validates that the latest row is complete, and writes that row to both Hopsworks groups.

## 6. Modelling approach

### 6.1 Train/test design

Records are sorted by time and split chronologically: the first 80% are training data and the final 20% are test data. This is more appropriate than random splitting for a forecasting task because it avoids training on future observations.

For tabular models, non-model identifiers and targets are removed. For the LSTM, features are standardized and transformed into rolling 24-hour sequences.

### 6.2 Candidate models

| Model | Implementation | Key settings | Notes |
|---|---|---|---|
| Random Forest | `RandomForestRegressor` | 200 trees, fixed random seed, parallel jobs | Native multi-output regression. |
| Ridge | `RidgeCV` | alpha grid 0.001–100 | Linear baseline with automatic regularization selection. |
| XGBoost | `XGBRegressor` inside `MultiOutputRegressor` | 400 estimators, depth 7, learning rate 0.05 | One boosted-tree estimator per forecast horizon. |
| LSTM | Keras Sequential model | 64 LSTM units, Dense(32), Dense(3), early stopping | Sequence model over 24-hour windows. |

The evaluation routine reports R², RMSE, and MAE independently for each horizon, then uses the mean of the three R² values as `average_r2` for model selection.

## 7. Model-comparison results

`model_comparison_results.csv` contains four logged runs for Random Forest, Ridge, and XGBoost, plus two LSTM entries. One LSTM entry has a blank `average_r2`; it is excluded from average-R² aggregation below.

### 7.1 Mean results across logged experiments

| Model | 24h R² | 24h RMSE | 24h MAE | 48h R² | 48h RMSE | 48h MAE | 72h R² | 72h RMSE | 72h MAE | Mean average R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.694 | 29.427 | 21.320 | 0.490 | 37.999 | 27.774 | 0.427 | 40.283 | 29.929 | 0.537 |
| Ridge | **0.758** | 26.165 | 19.146 | **0.598** | **33.723** | **25.392** | **0.534** | **36.354** | **27.335** | **0.630** |
| XGBoost | **0.773** | **25.370** | **18.223** | 0.508 | 37.315 | 27.645 | 0.455 | 39.311 | 29.413 | 0.579 |
| LSTM* | 0.666 | 30.821 | 21.767 | 0.534 | 36.419 | 26.587 | 0.494 | 37.953 | 28.171 | 0.565 |

*LSTM average metrics use the one complete average-R² record where applicable; per-horizon figures reflect the available logged LSTM values.

### 7.2 Interpretation

- **Best overall: Ridge.** It has the highest mean average R² (about 0.630) and the best 48-hour and 72-hour R², RMSE, and MAE results. This makes it the best current choice when all three horizons matter.
- **Best next-day model: XGBoost.** Its 24-hour R² is about 0.773 with the lowest 24-hour RMSE and MAE, so it is attractive when the product focuses primarily on the next-day forecast.
- **Random Forest:** provides a reasonable nonlinear baseline but trails the other tabular candidates at every horizon in the logged results.
- **LSTM:** improves over Random Forest at longer horizons but does not beat Ridge overall. The current deployment format is better aligned with the tabular sklearn candidates than with a sequence model.
- **Forecast uncertainty increases with horizon.** Every model’s R² falls and error grows from 24 to 72 hours, which is expected because future atmospheric and emission conditions become harder to infer.

The best individual complete logged result is **Ridge**, with `average_r2 = 0.6302634341`.

## 8. Training, registration, and deployment

After comparison, the selected model is serialized to `saved_models/aqi_forecast_multi.pkl`. Its metrics and algorithm name are registered in Hopsworks under the `aqi_forecast_multi` model name.

The deployment script asks Hopsworks for the model with the greatest `Average_R2`, uploads `predictor.py` as the serving script, replaces any existing `aqiforecastmulti` deployment, and starts the replacement. The predictor loads the serialized artifact from `MODEL_FILES_PATH` and returns a three-value forecast.

The daily training pipeline performs the full orchestration: load feature-group data, build targets, train all models, select the best, save/register it, and deploy it.

## 9. API layer

`app.py` is the FastAPI boundary between the dashboard, the feature store, and Hopsworks Model Serving. It is deployed independently on **Vercel** (see Section 4.1) and is called by the Streamlit dashboard over HTTPS rather than being co-located in the same process.

| Endpoint | Purpose |
|---|---|
| `GET /` | Basic service identity response. |
| `GET /health` | Health probe response. |
| `POST /predict` | Retrieves the latest online feature vector, removes non-model columns, and forwards it to Hopsworks model serving. |
| `GET /history?days=30` | Returns recent Lahore rows from the historical feature group for EDA. |
| `POST /explain` | Produces local SHAP explanations for 24h, 48h, and 72h forecasts. |

For explainability, the API downloads the best registered artifact, confirms its feature schema, obtains a 100-row historical background sample, selects TreeExplainer for tree models and LinearExplainer for Ridge, and normalizes multi-output SHAP values into one feature contribution vector per horizon. Model and background data are cached in memory to reduce repeat latency — though see Section 4.1 for a caveat on how well this caching holds up across Vercel's serverless cold starts.

Interactive Swagger documentation for all endpoints is available at `/docs` on the deployed API (see Live Demo section above).

## 10. Streamlit dashboard

The main dashboard is intentionally full-width and does not expose a sidebar. It uses a dark visual system with AQI-severity colours.

### Forecast area

- staged AQI loading screen;
- hero card showing the 24-hour AQI, severity, timestamp, and health advice;
- refresh control that refreshes only the forecast;
- three gauge cards for 24h, 48h, and 72h values;
- forecast trend chart and health-advisory card;
- dynamic AQI alert for every condition: Good, Moderate, Unhealthy, Very Unhealthy, or Hazardous.

### EDA area

Shown directly below the forecast after the historical request completes:

- historical row count, feature completeness, and latest observation time;
- AQI time-series chart, with PM2.5 and PM10 available from the legend;
- feature correlation heatmap;
- interactive selected-feature versus observed-AQI scatter plot.

### SHAP area

Shown below EDA after explanation calculation completes:

- horizon selector for 24h, 48h, or 72h;
- top 15 positive/negative feature contributions;
- hover values for the actual feature value and SHAP impact;
- prediction and base-value summary.

## 11. File-by-file implementation inventory

| File | Responsibility |
|---|---|
| `README.md` | Project introduction, local setup, and deployment instructions. |
| `main.py` | Development scratchpad for testing feature-store and feature-view operations. |
| `requirements.txt` | Pinned backend, data-pipeline, model-training, and SHAP packages, used for training and by the Streamlit deployment. |
| `requirements-api.txt` | Slimmer dependency list scoped to the FastAPI backend's Vercel deployment; excludes training-only packages (`tensorflow`, `xgboost`, `scikit-learn`) to stay within Vercel's function bundle size limit. |
| `requirements-streamlit.txt` | Lightweight Streamlit/Plotly/Pillow dashboard environment. |
| `runtime.txt` | Pins the Python version used by Streamlit Cloud, to avoid drift to a newer default runtime that may be incompatible with pinned packages such as `tensorflow`. |
| `vercel.json` | Vercel build/routing configuration for the FastAPI backend, including the Fluid Compute flag used to support the deployment's dependency size. |
| `icon.png` | Browser/page icon used by Streamlit. |
| `app.py` | FastAPI service for live prediction, history retrieval, and SHAP explanation. Deployed on Vercel; see Section 4.1. |
| `streamlit_app.py` | Main dashboard: forecast visualisation, alerts, EDA, and SHAP presentation. Deployed on Streamlit Cloud; calls the FastAPI backend via `AQI_API_BASE_URL`. |
| `deploy_model.py` | Selects the highest-Average-R² registry model and deploys/replaces Hopsworks serving. |
| `predictor.py` | Hopsworks serving predictor that loads the serialized model and returns three forecasts. |
| `model_comparison_results.csv` | Experiment log used for the model-comparison section of this report. |
| `models/preprocessing.py` | Target generation, lag/change-rate features, categorical weekday encoding, model matrix construction, chronological split, and online preparation helper. |
| `models/train.py` | Defines Random Forest, Ridge, and XGBoost training routines. |
| `models/lstm.py` | Defines 24-step sequence creation, scaling, and Keras LSTM architecture/training. |
| `models/evaluate.py` | Computes per-horizon R², RMSE, and MAE. |
| `models/training_pipeline.py` | Trains all candidates, records results, and returns the best candidate. |
| `models/model_registry.py` | Creates a Hopsworks registry model and saves metrics/artifact. |
| `src/data/weather_client.py` | Open-Meteo geocoding, historical-weather, and latest-weather requests. |
| `src/data/air_quality_client.py` | Open-Meteo air-quality history and latest AQI requests. |
| `src/data/data_merger.py` | Combines same-index hourly weather and air-quality arrays into records. |
| `src/data/features/feature_engineering.py` | Time-feature extraction. |
| `src/data/features/historical_pipeline.py` | Converts merged records to Lahore feature records and renames `us_aqi` to `aqi`. |
| `src/data/features/feature_view.py` | Creates/gets the historical `aqi_prediction_fv` feature view. |
| `src/data/features/prediction_features.py` | Reads a historical feature vector by city/time and removes non-model columns. |
| `src/data/features/feature_store/hopswork_client.py` | Hopsworks login, historical/latest feature-group creation, retrying inserts, and latest feature-view creation. |
| `src/data/pipelines/feature_pipeline.py` | Batch historical ingest, merge, engineering, null removal, and feature-group write. |
| `src/data/pipelines/hourly_pipeline.py` | Hourly online ingestion and latest-feature-group update. |
| `src/data/pipelines/daily_training_pipeline.py` | Scheduled-style end-to-end retraining, registry, and deployment pipeline. |

## 12. How to operate the system

### 12.1 Prerequisites

- Python virtual environment;
- Hopsworks project access and `HOPSWORKS_API_KEY`;
- access to the Hopsworks model-serving endpoint;
- installed dependencies. If a custom pip index cannot find SHAP, install it explicitly from PyPI:

```bash
python3 -m pip install --index-url https://pypi.org/simple shap
```

### 12.2 Local development

For local development, both services run as separate processes on the same machine, and the Streamlit app talks to the FastAPI backend over `localhost`.

Set `HOPSWORKS_API_KEY` in a local `.env` file (never committed to version control).

Start FastAPI in one terminal:

```bash
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Start Streamlit in a second terminal:

```bash
streamlit run streamlit_app.py
```

If Streamlit reports that it cannot connect to `localhost:8000`, FastAPI is not running, has stopped with an error, or is using another port.

### 12.3 Production deployment

In production, the two services run independently, on different platforms, and are connected by environment variables rather than `localhost`:

| Variable | Set where | Purpose |
|---|---|---|
| `AQI_API_BASE_URL` | Streamlit Cloud → App Settings → Secrets | Full base URL of the deployed FastAPI backend (e.g. `https://your-project.vercel.app`), so the dashboard stops defaulting to `localhost:8000`. |
| `HOPSWORKS_API_KEY` | Vercel → Project Settings → Environment Variables (marked Sensitive) | Authenticates the FastAPI backend against the Hopsworks feature store and model registry. |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | Vercel → Project Settings → Environment Variables | Set to `1` to opt into Vercel's Large Functions beta, required because the backend's dependency bundle (chiefly the `hopsworks` package) exceeds the standard 500 MB function size limit. Requires Fluid Compute to be enabled on the project. |

Deployment steps, at a high level:

1. Deploy the FastAPI backend to Vercel from this repository, with `requirements-api.txt` as the install target and `vercel.json` providing build/routing configuration.
2. Set `HOPSWORKS_API_KEY` and `VERCEL_SUPPORT_LARGE_FUNCTIONS` in the Vercel project's environment variables, and confirm Fluid Compute is enabled under Project Settings → Functions.
3. Once the backend is live, verify it independently at `/health` and `/docs` before connecting the dashboard.
4. Deploy `streamlit_app.py` to Streamlit Cloud from the same repository, with `runtime.txt` pinning the Python version and `requirements-streamlit.txt` (or `requirements.txt`, depending on final repo layout) as the dependency source.
5. Set `AQI_API_BASE_URL` as a Streamlit Cloud secret, pointing at the live Vercel URL from step 1.
6. Confirm CORS is enabled in `app.py` (`CORSMiddleware`) so that requests from the Streamlit Cloud origin succeed rather than being blocked by the browser.

## 13. Current limitations and recommendations

1. **Feature-pipeline dates are hard-coded.** `feature_pipeline.py` currently uses a fixed August 2026 range. Convert dates to function parameters or scheduled rolling windows.
2. **Validation can be stronger.** The current 80/20 chronological split is sensible, but walk-forward or expanding-window cross-validation would give more robust performance estimates.
3. **Data quality checks should precede merging.** Verify timestamps, gaps, API response length, and time-zone alignment rather than relying only on matching array indexes.
4. **Persist preprocessing artefacts.** The LSTM scaler is not saved alongside the model. Persist and version the scaler and feature schema for reproducible inference.
5. **LSTM serving/explanation needs a sequence path.** The deployed predictor and SHAP endpoint are designed for tabular sklearn artifacts. If LSTM is selected for deployment, save/load it with Keras and construct a 24-step input sequence plus a sequence-aware explainer.
6. **Results log hygiene.** The CSV appends experiments indefinitely and includes one incomplete LSTM average. Add run ID, training date, data range, model hyperparameters, and validation status; do not select incomplete results.
7. **Secrets and endpoints.** API keys are kept in environment variables on each hosting platform (Vercel, Streamlit Cloud) rather than in source control; continue to avoid committing real `.env` files, and rotate `HOPSWORKS_API_KEY` if it is ever accidentally exposed.
8. **Cold-start latency on the backend.** Because the FastAPI backend runs as a serverless function on Vercel, `/explain` in particular may re-load the model artifact on a cold invocation rather than reusing an in-memory cache, which can make the first SHAP request after idle time noticeably slower than subsequent ones. Monitor whether this is acceptable for the intended audience, or consider a persistent-host alternative if not.
9. **Monitoring.** Add prediction latency, feature freshness, API error, missing-data, and post-deployment forecast-error monitoring.
10. **Documentation.** Keep the README and this report current as the deployed endpoint, model, and workflow mature.

## 14. Conclusion

Karan’s AQI Predictor for Lahore demonstrates a complete data-science lifecycle: acquisition, feature engineering, multi-horizon forecasting, model comparison, experiment logging, feature-store integration, model registry/deployment, interactive EDA, explainability, and a practical dashboard, deployed as two independently hosted, production services. The logged results support Ridge as the current best balanced model, while XGBoost remains the strongest next-day specialist. The main next steps are operational hardening: scheduled ingestion and retraining, stronger time-series validation, clean experiment metadata, monitoring of the serverless backend's cold-start behaviour, and a dedicated sequence-serving path if the LSTM is to be deployed.
