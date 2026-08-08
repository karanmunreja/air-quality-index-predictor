from src.data.weather_client import get_cordinates
import requests
AIR_QUALITY_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi"
]
AQI_URL="https://air-quality-api.open-meteo.com/v1/air-quality"
def get_aqi_data(city,start_date,end_date):
    lat,long=get_cordinates(city)
    params={
            "latitude":lat,
               "longitude":long,
               "start_date":start_date,
               "end_date":end_date,
               "hourly":",".join(AIR_QUALITY_VARIABLES),
               "timezone":"auto"
    }
    response=requests.get(AQI_URL,params=params)
    response.raise_for_status()
    data=response.json()
    return data

def get_current_aqi(city):
    lat,long=get_cordinates(city)
    params={
        
    } 