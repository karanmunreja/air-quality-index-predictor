import os 
import pandas as pd

def save_weather_data(weather):
    DATA_DIR="data"
    FILE_NAME="weather_data.csv"
    FILE_PATH=os.path.join(DATA_DIR,FILE_NAME)

    os.makedirs(DATA_DIR, exist_ok=True)
    df=pd.DataFrame([weather])
    if os.path.exists(FILE_PATH):
        df.to_csv(FILE_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(FILE_PATH, index=False)
