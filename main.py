from src.data.storage import save_weather_data, save_historical_data
from src.data.weather_client import get_weather
from src.data.features.pipeline import build_features 
from src.data.historical_client import get_historical_weather
from src.data.features.historical_pipeline import build_historical_feat

weather=get_weather("Ghotki")
final_data=build_features(weather)
save_weather_data(final_data)
print(final_data)
print('saved successfully') 
raw_data=get_historical_weather('Ghotki','2025-01-01','2025-01-02')
features=build_historical_feat(raw_data)
save_historical_data(features)
print('saved h data')