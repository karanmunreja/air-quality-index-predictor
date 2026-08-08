from src.data.features.feature_store.hopswork_client import get_feature_group
import pandas as pd

TARGET_COLS = ['target_aqi_24', 'target_aqi_48', 'target_aqi_72']
DROP_COLS = [ "time", "minute", "city"]

LAG_COLS = ['aqi']
LAG_HOURS = [1,6,24,48,72]
def load_training_data():
    feature_group=get_feature_group()
    df=feature_group.read()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    return df

def prepare_data(df,horizon_col):
    df = df.dropna(subset=[horizon_col]).reset_index(drop=True)
    y = df[horizon_col]
    X = df.drop(columns=TARGET_COLS + ['minute', 'time', 'city'])
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    return X, y

def preprocess(df):
    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    df["weekday"] = df["weekday"].map(weekday_map)
    return df

def split_data(X,y):
    split_index=int(len(X)*0.8)
    X_train=X[:split_index]
    X_test=X[split_index:]
    y_train=y[:split_index]
    y_test=y[split_index:]
    return X_train,X_test,y_train,y_test

def create_lag_feat(df):
    for col in LAG_COLS:
        for lag in LAG_HOURS:
            df[f'{col}_{lag}'] = df[col].shift(lag)
    return df

def create_change_rate(df):
    df['aqi_change_rate'] = df['aqi'].diff()
    df['pm2_5_change_rate'] = df['pm2_5'].diff()
    df['pm10_change_rate'] = df['pm10'].diff()
    return df

def create_target(df):
    df = df.sort_values("time").reset_index(drop=True)
    df['target_aqi_24']=df['aqi'].shift(-24)
    df['target_aqi_48']=df['aqi'].shift(-48)
    df['target_aqi_72']=df['aqi'].shift(-72)
    return df

def engineer_features(df):
    df = preprocess(df)
    df = create_lag_feat(df)
    df = create_change_rate(df)
    return df

def prepare_prediction_data(latest,history):
    df=pd.concat([history,latest],ignore_index=True)
    df=engineer_features(df)
    latest=df.tail(1)
    X=df.drop(columns=DROP_COLS+TARGET_COLS, errors="ignore")
    return X