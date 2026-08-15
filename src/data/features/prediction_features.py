import pandas as pd

from src.data.features.feature_view import get_feature_view


DROP_FOR_MODEL = [
    "time",
    "city",
    "minute",
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72"
]


def get_feature_vector(city, time):

    fv = get_feature_view()

    feature_vector = fv.get_feature_vector(
        entry={
            "city": city,
            "time": time
        },
        return_type="pandas"
    )

    if feature_vector is None:
        raise ValueError(
            f"No online feature vector found for "
            f"city={city}, time={time}"
        )

    X = feature_vector.drop(
        columns=DROP_FOR_MODEL,
        errors="ignore"
    )

    return X
