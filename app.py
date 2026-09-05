"""FastAPI backend for forecasts, EDA data, and local SHAP explanations."""

from functools import lru_cache
from pathlib import Path
import os
import re
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, Query
import joblib
import numpy as np
import pandas as pd
import requests
try:
    import shap
    SHAP_IMPORT_ERROR = None
except ImportError as exc:
    shap = None
    SHAP_IMPORT_ERROR = str(exc)
from dotenv import load_dotenv

from src.data.features.feature_store.hopswork_client import (
    connect,
    get_feature_group,
    get_latest_feature_view,
)
from src.config import CITY, HOPSWORKS_ENDPOINT, MODEL_NAME


NON_MODEL_COLUMNS = [
    "time", "city", "minute", "target_aqi_24", "target_aqi_48", "target_aqi_72",
]
HORIZONS = ("24h", "48h", "72h")

# Path to the bundled model artifact, used as a fallback when Hopsworks
# Model Serving is unreachable (e.g. quota-frozen, network error).
LOCAL_MODEL_PATH = Path(__file__).parent / "saved_models" / "aqi_forecast_multi.pkl"

load_dotenv()
HOPSWORKS_API_KEY = (os.getenv("HOPSWORKS_API_KEY") or "").strip()

app = FastAPI(title="AQI Forecast API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def latest_feature_view():
    return get_latest_feature_view()


def latest_features() -> pd.DataFrame:
    features = latest_feature_view().get_feature_vector(
        entry={"city": CITY}, return_type="pandas"
    )
    if features is None or features.empty:
        raise HTTPException(status_code=404, detail="Latest Lahore feature vector not found")
    return features


def model_inputs(features: pd.DataFrame, feature_names=None) -> pd.DataFrame:
    """Apply the exact serving drop-list and preserve the training feature order."""
    X = features.drop(columns=NON_MODEL_COLUMNS, errors="ignore").copy()
    if feature_names is not None:
        missing = set(feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Latest feature row is missing model columns: {sorted(missing)}")
        X = X.loc[:, list(feature_names)]
    return X


@lru_cache(maxsize=1)
def load_explanation_model():
    """Download the registered artifact only when an explanation is requested."""
    registry = connect().get_model_registry()
    # Keep explanations aligned with deploy_model(), which serves the best
    # registered version rather than necessarily the most recently trained one.
    model_meta = registry.get_best_model(MODEL_NAME, "Average_R2", "max")
    model_dir = Path(model_meta.download())
    artifacts = list(model_dir.rglob("*.pkl")) + list(model_dir.rglob("*.joblib"))
    if not artifacts:
        raise RuntimeError(f"No joblib model artifact found in {model_dir}")
    return joblib.load(artifacts[0]), model_meta


@lru_cache(maxsize=1)
def load_local_fallback_model():
    """Loads the bundled model artifact so /predict can still work if
    Hopsworks Model Serving is unreachable (e.g. quota-frozen, network error)."""
    if not LOCAL_MODEL_PATH.exists():
        raise RuntimeError(f"Local fallback model not found at {LOCAL_MODEL_PATH}")
    return joblib.load(LOCAL_MODEL_PATH)


def predict_locally(X: pd.DataFrame) -> dict:
    """Run inference with the local model instead of the Hopsworks serving endpoint."""
    model = load_local_fallback_model()
    if hasattr(model, "feature_names_in_"):
        X = X.loc[:, list(model.feature_names_in_)]
    preds = np.asarray(model.predict(X)).reshape(-1)
    if len(preds) < 3:
        raise RuntimeError(f"Local fallback model returned {len(preds)} values, expected 3")
    return {"predictions": [preds[:3].tolist()], "source": "local_fallback"}


@lru_cache(maxsize=4)
def background_data(feature_names: tuple) -> pd.DataFrame:
    """Use historical training data, not the one-row online feature view, as SHAP baseline."""
    df = get_feature_group().read()
    df = df[df["city"] == CITY].sort_values("time")
    X = model_inputs(df, feature_names).dropna()
    if X.empty:
        raise ValueError("No complete historical rows are available for the SHAP baseline")
    return X.sample(min(100, len(X)), random_state=42)


def vector_for_horizon(values, horizon_index: int) -> np.ndarray:
    """Normalize SHAP's list/2-D/3-D multi-output formats to one feature vector."""
    if isinstance(values, list):
        return np.asarray(values[horizon_index])[0]
    values = np.asarray(values)
    if values.ndim == 3:
        return values[0, :, horizon_index]
    if values.ndim == 2:
        return values[0]
    return values.reshape(-1)


def base_for_horizon(expected_value, horizon_index: int) -> float:
    values = np.asarray(expected_value).reshape(-1)
    return float(values[min(horizon_index, len(values) - 1)])


def explain_tabular_model(model, X: pd.DataFrame, background: pd.DataFrame) -> dict:
    """Return one SHAP contribution vector per forecast horizon for sklearn models."""
    # MultiOutputRegressor (the XGBoost option) has a separately explainable
    # estimator for each horizon.
    if hasattr(model, "estimators_") and type(model).__name__ == "MultiOutputRegressor":
        result = {}
        for index, horizon in enumerate(HORIZONS):
            explainer = shap.TreeExplainer(model.estimators_[index])
            result[horizon] = {
                "base_value": base_for_horizon(explainer.expected_value, 0),
                "shap_values": vector_for_horizon(explainer.shap_values(X), 0).tolist(),
            }
        return result

    model_name = type(model).__name__.lower()
    explainer = (
        shap.LinearExplainer(model, background)
        if "ridge" in model_name or "linear" in model_name
        else shap.TreeExplainer(model)
    )
    shap_values = explainer.shap_values(X)
    return {
        horizon: {
            "base_value": base_for_horizon(explainer.expected_value, index),
            "shap_values": vector_for_horizon(shap_values, index).tolist(),
        }
        for index, horizon in enumerate(HORIZONS)
    }


@app.get("/")
def home():
    return {"message": "AQI Forecast API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict():
    try:
        X = model_inputs(latest_features())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        response = requests.post(
            HOPSWORKS_ENDPOINT,
            headers={
                "authorization": f"ApiKey {HOPSWORKS_API_KEY}",
                "content-type": "application/json",
            },
            json={"inputs": X.values.tolist()},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        result["source"] = "hopsworks"
        return result
    except requests.RequestException as exc:
        # Hopsworks Model Serving is unreachable (e.g. quota freeze) — fall
        # back to the bundled local model instead of failing outright.
        try:
            return predict_locally(X)
        except Exception as fallback_exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Hopsworks model request failed: {exc}. "
                    f"Local fallback also failed: {fallback_exc}"
                ),
            ) from exc


@app.get("/model")
def model_info():
   
    try:
        registry = connect().get_model_registry()
        model_meta = registry.get_best_model(MODEL_NAME, "Average_R2", "max")
        desc = getattr(model_meta, "description", None)
        if model_meta.version == 1:
            desc = "AQI Forecasting Model — algorithm: Ridge"
        return {
            "model_name": MODEL_NAME,
            "registered_name": getattr(model_meta, "name", None),
            "version": getattr(model_meta, "version", None),
            "metrics": getattr(model_meta, "training_metrics", {}),
            "description": desc
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not retrieve model info: {exc}") from exc


@app.get("/history")
def history(days: int = Query(default=30, ge=7, le=365)):
    """Historical, engineered data used by EDA; target columns are present when available."""
    try:
        df = get_feature_group().read()
        df = df[df["city"] == CITY].copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").tail(days * 24)
        df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return df.replace({np.nan: None}).to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain")
def explain():
    try:
        if shap is None:
            raise HTTPException(
                status_code=503,
                detail="SHAP is not installed. Install it from PyPI before using explanations.",
            )
        model, metadata = load_explanation_model()
        if not hasattr(model, "feature_names_in_"):
            raise HTTPException(
                status_code=422,
                detail="The deployed model has no feature schema. Retrain and register a tabular sklearn model.",
            )
        feature_names = list(model.feature_names_in_)
        X = model_inputs(latest_features(), feature_names)
        explanations = explain_tabular_model(model, X, background_data(tuple(feature_names)))
        predictions = np.asarray(model.predict(X)).reshape(-1)

        for index, horizon in enumerate(HORIZONS):
            explanations[horizon]["prediction"] = float(predictions[index])

        return {
            "model_version": getattr(metadata, "version", None),
            "features": feature_names,
            "feature_values": {name: float(X.iloc[0][name]) for name in feature_names},
            "horizons": explanations,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not compute SHAP explanation: {exc}") from exc