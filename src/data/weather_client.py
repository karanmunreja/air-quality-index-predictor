import requests
import os

from dotenv import load_dotenv
load_dotenv()
weather_api=os.getenv("WEATHER_API_KEY")
weather_api_url = "https://api.weatherapi.com/v1/current.json"

def get_weather(city):
    params={
        "key": weather_api,
        "q": city,
        "aqi":"yes"
    }
    response=requests.get(weather_api_url, params=params)
    response.raise_for_status()
    return response.json()
