"""Small local diagnostic for the latest online feature view.

Run scheduled ingestion and training through `src.data.pipelines.hourly_pipeline`
and `src.data.pipelines.daily_training_pipeline` instead.
"""

# start = time.time()
# fg = get_feature_group()
# print("Get FG:", time.time() - start)

# start = time.time()
# query = fg.select_all().limit(5)
# print("Build Query:", time.time() - start)

# start = time.time()
# df = query.read()
# print("Read:", time.time() - start)

# df=run_feature_pipeline()
# run_training_pipeline(df)

# data = get_current_aqi("Lahore")
# print(data)
# from predictions.predict import predict_all
# results=predict_all()
# print(results)

# from src.data.features.feature_view import get_feature_view


# fv = get_feature_view()

# print("Feature View created/found successfully")

# print("Name:", fv.name)
# print("Version:", fv.version)

# from datetime import datetime

# from src.data.features.feature_view import get_feature_view


# fv = get_feature_view()

# print("Feature View:", fv.name)
# print("Version:", fv.version)


# entry = {
#     "city": "Lahore",
#     "time":"2023-07-04T14:00"
    
# }

# features = fv.get_feature_vector(
#     entry=entry
# )

# print("\nFeature vector:")
# print(features)

 
# print("Feature names:")
# for feature in fv.features:
#     print(feature.name)

# from src.data.features.prediction_features import get_feature_vector

# X = get_feature_vector(
#     city="Lahore",
#     time="2023-07-04T14:00"
# )

# print(X.shape)
# print(X.columns.tolist())
# print(X)

from src.data.features.feature_store.hopswork_client import (
    get_latest_feature_view
)

fv = get_latest_feature_view()

print("Feature View created:")
print(fv.name)
print("Version:", fv.version)
