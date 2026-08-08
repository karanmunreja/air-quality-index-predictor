from src.data.features.feature_store.hopswork_client import get_feature_group
from src.data.pipelines.feature_pipeline import run_feature_pipeline
from src.data.pipelines.training_pipeline import run_training_pipeline

import time

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

# from predictions.predict import predict_all
# results=predict_all()
# print(results)
