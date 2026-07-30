import os 
import pandas as pd

DATA_DIR="data"
HISTORICAL_FILE = os.path.join(DATA_DIR, "historical_weather_data.csv")
def save_weather_data(weather):
   
    FILE_NAME="weather_data.csv"
    FILE_PATH=os.path.join(DATA_DIR,FILE_NAME)

    os.makedirs(DATA_DIR, exist_ok=True)
    df=pd.DataFrame([weather])
    if os.path.exists(FILE_PATH):
        df.to_csv(FILE_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(FILE_PATH, index=False)

def save_historical_data(features):
    df=pd.DataFrame(features)
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(HISTORICAL_FILE):
        df.to_csv(HISTORICAL_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(HISTORICAL_FILE, index=False)
    