import requests
GEOCODING_URL="https://geocoding-api.open-meteo.com/v1/search"
HIST_WEATHER_URL="https://archive-api.open-meteo.com/v1/archive"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
   "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m"
]

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

def get_cordinates(city):
    params={
        "name":city,
        "count":1,
        "language":"en",
        "format":"json"
    }
    response=requests.get(GEOCODING_URL,params=params)
    response.raise_for_status()
    data=response.json()
    results=data.get('results')
    if not results:
        raise ValueError(f"City '{city}' not found.")
    result=results[0]
    latitude=result.get('latitude')
    longitude=result.get('longitude')
    return latitude,longitude

def get_historical_weather(city, start_date, end_date):
    lat,long=get_cordinates(city)
    params={
       "latitude":lat,
       "longitude":long,
       "start_date":start_date,
       "end_date":end_date,
       "hourly":",".join(HOURLY_VARIABLES),
       "timezone":"auto"
    }
    response=requests.get(HIST_WEATHER_URL,params=params)
    response.raise_for_status()
    data=response.json()
    return data

def get_weather(city):
    latitude, longitude = get_cordinates(city)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_hours": 1,
        "timezone": "auto"
    }
    response = requests.get(WEATHER_URL, params=params)
    response.raise_for_status()
    return response.json()