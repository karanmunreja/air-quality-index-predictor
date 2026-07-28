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

def extract_weather_data(data):
    return {
        "city": data["location"]["name"],
        "country": data["location"]["country"],
        "temperature": data["current"]["temp_c"],
        "humidity": data["current"]["humidity"],
        "pressure": data["current"]["pressure_mb"],
        "wind_speed": data["current"]["wind_kph"],
        "wind_direction": data["current"]["wind_dir"],
        "visibility": data["current"]["vis_km"],
        "uv": data["current"]["uv"],
        "pm25": data["current"]["air_quality"]["pm2_5"],
        "pm10": data["current"]["air_quality"]["pm10"],
        "co": data["current"]["air_quality"]["co"],
        "no2": data["current"]["air_quality"]["no2"],
        "o3": data["current"]["air_quality"]["o3"],
        "so2": data["current"]["air_quality"]["so2"],
        "last_updated": data["current"]["last_updated"],
    }