import pandas as pd

from models.preprocessing import create_target, engineer_features
from src.data.data_merger import merge_data
from src.data.features.feature_engineering import add_time_features


def test_add_time_features_extracts_expected_fields():
    row = add_time_features({"time": "2026-08-22T14:00"})

    assert row["year"] == 2026
    assert row["month"] == 8
    assert row["day"] == 22
    assert row["hour"] == 14
    assert row["weekday"] == "Saturday"


def test_merge_data_combines_hourly_weather_and_aqi():
    weather = {"hourly": {"time": ["t1"], "temperature_2m": [31.0]}}
    air = {"hourly": {"time": ["t1"], "us_aqi": [90]}}

    assert merge_data(weather, air) == [{"time": "t1", "temperature_2m": 31.0, "us_aqi": 90}]


def test_targets_and_lag_features_are_created_without_future_leakage():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=80, freq="h").astype(str),
            "city": "Lahore",
            "weekday": "Thursday",
            "aqi": range(80),
            "pm2_5": range(80),
            "pm10": range(80),
        }
    )

    result = engineer_features(create_target(df))

    assert result.loc[0, "target_aqi_24"] == 24
    assert result.loc[0, "target_aqi_72"] == 72
    assert result.loc[72, "target_aqi_24"] != result.loc[72, "target_aqi_24"]
    assert result.loc[72, "aqi_72"] == 0
