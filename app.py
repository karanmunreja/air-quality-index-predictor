"""FastAPI backend for forecasts, EDA data, and local SHAP explanations."""

from functools import lru_cache
from pathlib import Path
import os

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

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

app = FastAPI(title="AQI Forecast API", version="1.1.0")


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
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Hopsworks model request failed: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
