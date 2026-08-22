# AQI Predictor for Lahore — Project Report

**Prepared for:** Karan, Data Science Intern
**Project:** AQI Predictor for Lahore
**Report scope:** End-to-end implementation, as represented by the repository at the time of this report.

## 1. Executive summary

This project is an end-to-end air-quality forecasting system for **Lahore, Pakistan**. It ingests weather and air-pollution data from Open-Meteo, creates hourly predictive features, stores them in Hopsworks, trains and compares multi-output machine-learning models, registers and deploys the preferred model, and presents forecast, exploratory data analysis (EDA), and explainability in a Streamlit dashboard.

The forecasting task predicts the **US AQI 24, 48, and 72 hours ahead** in a single model call. The current comparison results show that **Ridge** is the strongest all-round tabular model in the logged experiments, with an average R² of approximately **0.630**. XGBoost has the best 24-hour result, but Ridge is more reliable for the longer 48- and 72-hour horizons.

The delivered application has three main user-facing capabilities:

1. A live AQI forecast with visual AQI bands, health guidance, a persistent severity alert, and a forecast-only refresh control.
2. EDA using the historical Hopsworks feature group: data-quality summary, AQI trend, pollutant overlays, correlation matrix, and feature-versus-AQI exploration.
3. SHAP-based local model explanations for each forecast horizon, using the same latest online feature row used for live prediction.

## 2. Business problem and objective

Lahore can experience substantial variation in pollution levels. A short-horizon AQI forecast can support safer choices about outdoor exercise, commuting, school activities, and health precautions.

The project objective is to build a reproducible system that:

- collects hourly environmental and pollutant observations for Lahore;
- predicts future AQI at 24, 48, and 72 hours;
- identifies the best model using common regression metrics;
- deploys the selected model through Hopsworks Model Serving;
- provides a non-technical dashboard with actionable health context; and
- explains which features caused an individual forecast to increase or decrease.

## 3. Solution architecture

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

## 4. Data acquisition and preparation

### 4.1 Data sources

The project uses Open-Meteo services:

- **Geocoding API:** resolves Lahore into latitude and longitude.
- **Forecast/weather API:** collects hourly temperature, humidity, surface pressure, wind speed, and wind direction.
- **Archive weather API:** retrieves historical weather for the batch feature pipeline.
- **Air Quality API:** retrieves PM10, PM2.5, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone, and US AQI.

### 4.2 Feature construction

The raw hourly weather and AQI payloads are merged by time position. The resulting record is converted to a project feature record with city and AQI fields. Time components are then derived:

- year, month, day, hour, minute;
- weekday, later encoded from Monday–Sunday to 0–6;
- AQI lags at 1, 2, 3, 5, 6, 8, 10, 12, 24, 30, 36, 42, 48, 54, 60, and 72 hours;
- AQI, PM2.5, and PM10 change-rate features.

The supervised targets are produced by shifting AQI forward by 24, 48, and 72 rows: `target_aqi_24`, `target_aqi_48`, and `target_aqi_72`.

### 4.3 Handling history for online features

The hourly pipeline reads the last **76 hours** of historical data. This provides enough context for the largest 72-hour lag plus the current observation. It combines the new record with history, prevents duplicate timestamps, re-engineers the features, validates that the latest row is complete, and writes that row to both Hopsworks groups.

## 5. Modelling approach

### 5.1 Train/test design

Records are sorted by time and split chronologically: the first 80% are training data and the final 20% are test data. This is more appropriate than random splitting for a forecasting task because it avoids training on future observations.

For tabular models, non-model identifiers and targets are removed. For the LSTM, features are standardized and transformed into rolling 24-hour sequences.

### 5.2 Candidate models

| Model | Implementation | Key settings | Notes |
|---|---|---|---|
| Random Forest | `RandomForestRegressor` | 200 trees, fixed random seed, parallel jobs | Native multi-output regression. |
| Ridge | `RidgeCV` | alpha grid 0.001–100 | Linear baseline with automatic regularization selection. |
| XGBoost | `XGBRegressor` inside `MultiOutputRegressor` | 400 estimators, depth 7, learning rate 0.05 | One boosted-tree estimator per forecast horizon. |
| LSTM | Keras Sequential model | 64 LSTM units, Dense(32), Dense(3), early stopping | Sequence model over 24-hour windows. |

The evaluation routine reports R², RMSE, and MAE independently for each horizon, then uses the mean of the three R² values as `average_r2` for model selection.

## 6. Model-comparison results

`model_comparison_results.csv` contains four logged runs for Random Forest, Ridge, and XGBoost, plus two LSTM entries. One LSTM entry has a blank `average_r2`; it is excluded from average-R² aggregation below.

### 6.1 Mean results across logged experiments

| Model | 24h R² | 24h RMSE | 24h MAE | 48h R² | 48h RMSE | 48h MAE | 72h R² | 72h RMSE | 72h MAE | Mean average R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.694 | 29.427 | 21.320 | 0.490 | 37.999 | 27.774 | 0.427 | 40.283 | 29.929 | 0.537 |
| Ridge | **0.758** | 26.165 | 19.146 | **0.598** | **33.723** | **25.392** | **0.534** | **36.354** | **27.335** | **0.630** |
| XGBoost | **0.773** | **25.370** | **18.223** | 0.508 | 37.315 | 27.645 | 0.455 | 39.311 | 29.413 | 0.579 |
| LSTM* | 0.666 | 30.821 | 21.767 | 0.534 | 36.419 | 26.587 | 0.494 | 37.953 | 28.171 | 0.565 |

*LSTM average metrics use the one complete average-R² record where applicable; per-horizon figures reflect the available logged LSTM values.

### 6.2 Interpretation

- **Best overall: Ridge.** It has the highest mean average R² (about 0.630) and the best 48-hour and 72-hour R², RMSE, and MAE results. This makes it the best current choice when all three horizons matter.
- **Best next-day model: XGBoost.** Its 24-hour R² is about 0.773 with the lowest 24-hour RMSE and MAE, so it is attractive when the product focuses primarily on the next-day forecast.
- **Random Forest:** provides a reasonable nonlinear baseline but trails the other tabular candidates at every horizon in the logged results.
- **LSTM:** improves over Random Forest at longer horizons but does not beat Ridge overall. The current deployment format is better aligned with the tabular sklearn candidates than with a sequence model.
- **Forecast uncertainty increases with horizon.** Every model’s R² falls and error grows from 24 to 72 hours, which is expected because future atmospheric and emission conditions become harder to infer.

The best individual complete logged result is **Ridge**, with `average_r2 = 0.6302634341`.

## 7. Training, registration, and deployment

After comparison, the selected model is serialized to `saved_models/aqi_forecast_multi.pkl`. Its metrics and algorithm name are registered in Hopsworks under the `aqi_forecast_multi` model name.

The deployment script asks Hopsworks for the model with the greatest `Average_R2`, uploads `predictor.py` as the serving script, replaces any existing `aqiforecastmulti` deployment, and starts the replacement. The predictor loads the serialized artifact from `MODEL_FILES_PATH` and returns a three-value forecast.

The daily training pipeline performs the full orchestration: load feature-group data, build targets, train all models, select the best, save/register it, and deploy it.

## 8. API layer

`app.py` is the FastAPI boundary between the dashboard, the feature store, and Hopsworks Model Serving.

| Endpoint | Purpose |
|---|---|
| `GET /` | Basic service identity response. |
| `GET /health` | Health probe response. |
| `POST /predict` | Retrieves the latest online feature vector, removes non-model columns, and forwards it to Hopsworks model serving. |
| `GET /history?days=30` | Returns recent Lahore rows from the historical feature group for EDA. |
| `POST /explain` | Produces local SHAP explanations for 24h, 48h, and 72h forecasts. |

For explainability, the API downloads the best registered artifact, confirms its feature schema, obtains a 100-row historical background sample, selects TreeExplainer for tree models and LinearExplainer for Ridge, and normalizes multi-output SHAP values into one feature contribution vector per horizon. Model and background data are cached in memory to reduce repeat latency.

## 9. Streamlit dashboard

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

## 10. File-by-file implementation inventory

| File | Responsibility |
|---|---|
| `README.md` | Minimal project introduction. It should be expanded with setup and architecture instructions. |
| `main.py` | Development scratchpad for testing feature-store and feature-view operations. |
| `requirements.txt` | Pinned runtime packages: data science, TensorFlow, Hopsworks, FastAPI/Uvicorn, Streamlit, Plotly, SHAP, and Pillow. |
| `icon.png` | Browser/page icon used by Streamlit. |
| `app.py` | FastAPI service for live prediction, history retrieval, and SHAP explanation. |
| `streamlit_app.py` | Main dashboard: forecast visualisation, alerts, EDA, and SHAP presentation. |
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

## 11. How to operate the system

### Prerequisites

- Python virtual environment;
- Hopsworks project access and `HOPSWORKS_API_KEY` in `.env`;
- access to the Hopsworks model-serving endpoint;
- installed dependencies. If a custom pip index cannot find SHAP, install it explicitly from PyPI:

```bash
python3 -m pip install --index-url https://pypi.org/simple shap
```

### Local application startup

Start FastAPI in one terminal:

```bash
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Start Streamlit in a second terminal:

```bash
streamlit run streamlit_app.py
```

If Streamlit reports that it cannot connect to `localhost:8000`, FastAPI is not running, has stopped with an error, or is using another port.

## 12. Current limitations and recommendations

1. **Feature-pipeline dates are hard-coded.** `feature_pipeline.py` currently uses a fixed August 2026 range. Convert dates to function parameters or scheduled rolling windows.
2. **Validation can be stronger.** The current 80/20 chronological split is sensible, but walk-forward or expanding-window cross-validation would give more robust performance estimates.
3. **Data quality checks should precede merging.** Verify timestamps, gaps, API response length, and time-zone alignment rather than relying only on matching array indexes.
4. **Persist preprocessing artefacts.** The LSTM scaler is not saved alongside the model. Persist and version the scaler and feature schema for reproducible inference.
5. **LSTM serving/explanation needs a sequence path.** The deployed predictor and SHAP endpoint are designed for tabular sklearn artifacts. If LSTM is selected for deployment, save/load it with Keras and construct a 24-step input sequence plus a sequence-aware explainer.
6. **Results log hygiene.** The CSV appends experiments indefinitely and includes one incomplete LSTM average. Add run ID, training date, data range, model hyperparameters, and validation status; do not select incomplete results.
7. **Secrets and endpoints.** Keep API keys exclusively in environment variables and move the model-serving URL into configuration rather than source code.
8. **Monitoring.** Add prediction latency, feature freshness, API error, missing-data, and post-deployment forecast-error monitoring.
9. **Documentation.** Keep the README current as the deployed endpoint, model, and workflow mature.

## 13. Conclusion

Karan’s AQI Predictor for Lahore demonstrates a complete data-science lifecycle: acquisition, feature engineering, multi-horizon forecasting, model comparison, experiment logging, feature-store integration, model registry/deployment, interactive EDA, explainability, and a practical dashboard. The logged results support Ridge as the current best balanced model, while XGBoost remains the strongest next-day specialist. The main next step is operational hardening: scheduled ingestion and retraining, stronger time-series validation, clean experiment metadata, and a dedicated sequence-serving path if the LSTM is to be deployed.
