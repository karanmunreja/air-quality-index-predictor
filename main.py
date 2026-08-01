from src.data.storage import save_weather_data, save_historical_data
from src.data.weather_client import get_weather
from src.data.features.pipeline import build_features 
from src.data.weather_history_client import get_historical_weather
from src.data.features.historical_pipeline import build_historical_feat
from src.data.features.feature_store.hopswork_client import insert_features
from src.data.air_quality_history_client import get_aqi_data
from src.data.data_merger import merge_data
import pandas as pd

# weather=get_weather("Ghotki")
# final_data=build_features(weather)
# save_weather_data(final_data)
# print(final_data)
# print('saved successfully')
raw_w_data=get_historical_weather('Ghotki','2025-01-01','2025-01-02')
raw_aqi_data=get_aqi_data('Ghotki','2025-01-01','2025-01-02')
# merged_data=merge_data(raw_w_data,raw_aqi_data)
# features=build_historical_feat("Ghotki",merged_data)
# save_historical_data(features)
# df=pd.DataFrame(features)
# insert_features(df.head(1))
