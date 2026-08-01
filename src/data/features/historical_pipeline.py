from src.data.features.feature_engineering import add_time_features
def build_historical_feat(city,merged_data):
    features=[]
    for record in merged_data:
        feature=record.copy()
        feature["city"] = city
        feature["aqi"] = feature.pop("us_aqi")
        feature=add_time_features(feature)
        features.append(feature)
    return features