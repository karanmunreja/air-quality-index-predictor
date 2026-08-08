from src.data.air_quality_client import get_aqi_data
from src.data.data_merger import merge_data
from src.data.features.feature_store.hopswork_client import insert_features
from src.data.features.historical_pipeline import build_historical_feat
from src.data.weather_client import get_historical_weather
import pandas as pd

from models.preprocessing import engineer_features

def run_feature_pipeline():
    raw_w_data=get_historical_weather('Lahore','2023-07-01','2026-08-07')
    raw_aqi_data=get_aqi_data('Lahore','2023-07-01','2026-08-07')
    merged_data=merge_data(raw_w_data,raw_aqi_data)
    features=build_historical_feat("Lahore",merged_data)
    df=pd.DataFrame(features)
    df=engineer_features(df)
    df = df.dropna().reset_index(drop=True)
    insert_features(df)
    return df