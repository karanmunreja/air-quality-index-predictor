from src.data.features.feature_store.hopswork_client import get_feature_group, connect
from src.data.weather_client import get_weather
from src.data.air_quality_client import get_current_aqi
from src.data.data_merger import merge_data
from src.data.features.pipeline import build_features
from models.preprocessing import prepare_prediction_data

import pandas as pd
MAX_HISTORY=72
MODEL_NAMES = {
    "prediction_24": "aqi_forecast_24",
    "prediction_48": "aqi_forecast_48",
    "prediction_72": "aqi_forecast_72",
}
PROJECT = None
REGISTRY = None

MODELS = {}
def initialize_prediction_service():
    global PROJECT
    global REGISTRY
    global MODELS

    PROJECT = connect()
    REGISTRY = PROJECT.get_model_registry()
    MODELS = {}
    for model_name in MODEL_NAMES.values():
        model = REGISTRY.get_model(
            name=model_name
        )
        model_dir = model.download()
        MODELS[model_name] = joblib.load(
            f"{model_dir}/{model_name}.pkl"
        )
    print("Prediction service initialized successfully.")

def get_recent_history():
    fg=get_feature_group()
    df=fg.read()
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(MAX_HISTORY)

def fetch_latest_data():
    weather = get_weather("Lahore")
    aqi = get_current_aqi("Lahore")
    merged = merge_data(weather, aqi)
    features = build_features("Lahore", merged)
    latest = pd.DataFrame(features)
    return latest


import joblib

def predict_model(model,X):
    prediction=model.predict(X)
    return float(prediction[0])
def predict_all():

    history = get_recent_history()

    latest = fetch_latest_data()

    X = prepare_prediction_data(
        history,
        latest
    )

    predictions = {}

    for key, model_name in MODEL_NAMES.items():

        predictions[key] = predict_model(
            MODELS[model_name],
            X
        )

    return predictions