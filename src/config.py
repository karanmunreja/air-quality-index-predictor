"""Central, environment-based settings for the AQI Predictor."""

import os


CITY = os.getenv("AQI_CITY", "Lahore")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "jshsmekedaxakb")
TRAINING_FEATURE_GROUP = os.getenv("TRAINING_FEATURE_GROUP", "aqi_training_features")
TRAINING_FEATURE_GROUP_VERSION = int(os.getenv("TRAINING_FEATURE_GROUP_VERSION", "2"))
LATEST_FEATURE_GROUP = os.getenv("LATEST_FEATURE_GROUP", "latest_aqi_features")
LATEST_FEATURE_GROUP_VERSION = int(os.getenv("LATEST_FEATURE_GROUP_VERSION", "2"))
MODEL_NAME = os.getenv("MODEL_NAME", "aqi_forecast_multi")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "aqiforecastmulti")
HOPSWORKS_ENDPOINT = os.getenv(
    "HOPSWORKS_ENDPOINT",
    "http://57.130.17.185/v1/jshsmekedaxakb/"
    "aqiforecastmulti/v1/models/aqiforecastmulti:predict",
)
API_BASE_URL = os.getenv("AQI_API_BASE_URL", "http://localhost:8000")
