from src.data.weather_client import extract_weather_data
from src.data.features.feature_engineering import add_time_features

def build_features(raw_data):
    extracted_features=extract_weather_data(raw_data)
    added_features_data=add_time_features(extracted_features)
    return added_features_data
