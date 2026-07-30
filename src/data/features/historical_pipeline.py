from src.data.features.feature_engineering import add_time_features
def build_historical_feat(raw_data):
    hourly=raw_data.get("hourly")
    if not hourly:
        raise ValueError("Hourly data not found.")
    features=[]
    num_records=len(hourly['time'])
    for i in range(num_records):
        feature={}
        for key,values in hourly.items():
            feature[key]=values[i]
        feature=add_time_features(feature)
        features.append(feature)
    return features