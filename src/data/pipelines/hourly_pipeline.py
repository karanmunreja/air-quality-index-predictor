import pandas as pd

from src.data.weather_client import get_weather
from src.data.air_quality_client import get_current_aqi
from src.data.data_merger import merge_data
from src.data.features.historical_pipeline import build_historical_feat

from models.preprocessing import engineer_features

from src.data.features.feature_store.hopswork_client import (
    get_feature_group,
    insert_features,
    insert_latest_features
)
from src.config import CITY


HISTORY_HOURS = 76


def get_recent_history():

    fg = get_feature_group()

    # Read historical Feature Group
    df = fg.read()

    # Convert time for sorting
    df["time"] = pd.to_datetime(df["time"])

    # Only Lahore
    df = df[df["city"] == CITY]

    # Sort oldest → newest
    df = df.sort_values("time")

    # Enough context for the largest lag (72 hours)
    return df.tail(HISTORY_HOURS).copy()


def get_current_features():

    # Get current weather
    weather_data = get_weather(CITY)

    # Get current AQI
    aqi_data = get_current_aqi(CITY)

    # Reuse your existing merge function
    merged_data = merge_data(
        weather_data,
        aqi_data
    )

    # Reuse your existing feature builder
    features = build_historical_feat(
        CITY,
        merged_data
    )

    return pd.DataFrame(features)


def run_hourly_pipeline():

    # ==================================================
    # 1. Get historical context from Hopsworks
    # ==================================================

    history = get_recent_history()

    history_latest_time = history["time"].max()

    print("\nHistorical latest:")
    print(history_latest_time)

    # ==================================================
    # 2. Get the NEW current observation
    # ==================================================

    current = get_current_features()

    current["time"] = pd.to_datetime(
        current["time"]
    )

    # Since get_current_* returns one hourly row
    current_time = current["time"].iloc[0]

    print("\nCurrent observation:")
    print(
        current[["city", "time", "aqi"]]
    )

    # ==================================================
    # 3. Prevent duplicate / old data
    # ==================================================

    if current_time <= history_latest_time:

        print(
            f"\nNo new data to insert."
            f"\nHistorical latest: {history_latest_time}"
            f"\nCurrent: {current_time}"
        )

        return

    # ==================================================
    # 4. Use current observation's columns as the
    #    base schema
    #
    #    This avoids manually writing feature names.
    # ==================================================

    base_columns = current.columns.tolist()

    history = history[
        [
            column
            for column in base_columns
            if column in history.columns
        ]
    ].copy()

    current = current[
        [
            column
            for column in base_columns
            if column in current.columns
        ]
    ].copy()

    # ==================================================
    # 5. Combine historical context + NEW observation
    # ==================================================

    combined = pd.concat(
        [
            history,
            current
        ],
        ignore_index=True
    )

    combined = (
        combined
        .sort_values("time")
        .drop_duplicates(
            subset=["city", "time"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # ==================================================
    # 6. Engineer features
    # ==================================================

    engineered = engineer_features(
        combined
    )

    # ==================================================
    # 7. Select ONLY the new hourly row
    # ==================================================

    latest_row = engineered[
        engineered["time"] == current_time
    ].copy()

    if latest_row.empty:
        raise RuntimeError(
            f"Could not create feature row for {current_time}"
        )

    # ==================================================
    # 8. Check for missing values
    # ==================================================

    if latest_row.isna().any().any():

        missing_columns = latest_row.columns[
            latest_row.isna().any()
        ].tolist()

        raise RuntimeError(
            f"Latest row contains missing values: "
            f"{missing_columns}"
        )

    # ==================================================
    # 9. Display new feature row
    # ==================================================

    print("\nNEW ENGINEERED ROW:")
    print(
        latest_row[
            ["city", "time", "aqi"]
        ]
    )

    print(
        "\nFeature shape:",
        latest_row.shape
    )

    # ==================================================
    # 10. Insert ONLY new row into historical FG
    # ==================================================
    latest_row["time"] = latest_row["time"].dt.strftime("%Y-%m-%dT%H:%M")
    latest_row["weekday"] = latest_row["weekday"].astype("int64")
    insert_features(
        latest_row
    )

    print(
        "\nHistorical Feature Group updated."
    )

    # ==================================================
    # 11. Update latest serving FG
    # ==================================================

    insert_latest_features(
        latest_row
    )

    print(
        "Latest Feature Group updated."
    )

    print(
        "\nHourly pipeline completed successfully."
    )


if __name__ == "__main__":
    run_hourly_pipeline()
