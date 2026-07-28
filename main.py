from src.data.storage import save_weather_data
from src.data.weather_client import get_weather
from src.data.features.pipeline import build_features 

weather=get_weather("Ghotki")
final_data=build_features
save_weather_data(final_data)
print(final_data)
print('saved successfully') 
