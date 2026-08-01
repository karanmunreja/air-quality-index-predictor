import requests
GEOCODING_URL="https://geocoding-api.open-meteo.com/v1/search"
HIST_WEATHER_URL="https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
   "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m"
]

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
