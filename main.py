
import os
from pathlib import Path
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).parent


def fetch_sample_weather_aqi():
   
    lat, lon = 31.5204, 74.3587
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
        "&hourly=pm10,pm2_5,us_aqi"
    )
    print('Fetching sample data from Open-Meteo:', url)
    r = requests.get(url, timeout=30)
    data = r.json()
    hourly = data['hourly']
    times = pd.to_datetime(hourly['time'])
    df = pd.DataFrame({
        'time': times,
        'pm10': hourly.get('pm10', [np.nan] * len(times)),
        'pm2_5': hourly.get('pm2_5', [np.nan] * len(times)),
        'aqi': hourly.get('us_aqi', [np.nan] * len(times)),
    })
    return df


def load_or_fetch():
    csv_path = ROOT / 'model_comparison_results.csv'
    if csv_path.exists():
        print('Loading CSV from', csv_path)
        df = pd.read_csv(csv_path)
    else:
        print('CSV not found — fetching sample data from Open-Meteo')
        df = fetch_sample_weather_aqi()
        csv_out = ROOT / 'sample_aqi.csv'
        df.to_csv(csv_out, index=False)
        print('Saved sample to', csv_out)
    return df


def eda(df):
    print('\n=== EDA ===')
    print('rows, cols:', df.shape)
    print('\nhead:\n', df.head(5).to_string(index=False))
    print('\ndtypes:\n', df.dtypes)
    print('\nmissing values:\n', df.isna().sum())
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        print('\nnumeric describe:\n', numeric.describe().T[['mean','std','min','50%','max']])
        if 'aqi' in numeric.columns:
            print('\ncorrelation vs aqi:\n', numeric.corr()['aqi'].sort_values(ascending=False).head(10))


def create_features(df):
    print('\n=== Feature engineering ===')
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    else:
        df['time'] = pd.date_range('2023-01-01', periods=len(df), freq='H')
    df = df.sort_values('time').reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    # Create simple lag features for aqi
    if 'aqi' in df.columns:
        df['aqi_lag1'] = df['aqi'].shift(1)
        df['aqi_lag24'] = df['aqi'].shift(24)
    # fill small gaps
    df = df.fillna(method='ffill').fillna(method='bfill')
    feature_cols = [c for c in ['pm2_5', 'pm10', 'hour', 'weekday', 'aqi_lag1', 'aqi_lag24'] if c in df.columns]
    print('feature columns:', feature_cols)
    return df, feature_cols


def feature_group_actions():
    # Explicit import and call — will raise if module unavailable (user requested no try/except)
    from src.data.features.feature_store.hopswork_client import get_latest_feature_view
    print('\n=== Feature-group demo ===')
    fv = get_latest_feature_view()
    print('Feature view name:', fv.name)
    print('Feature view version:', fv.version)


def train_and_evaluate(df, feature_cols):
    print('\n=== Train baselines ===')
    X = df[feature_cols].values
    y = df['aqi'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    ridge.fit(X_train_s, y_train)
    rf.fit(X_train, y_train)  # tree-based model uses raw features

    pred_ridge = ridge.predict(X_test_s)
    pred_rf = rf.predict(X_test)

    def metrics(y_true, preds):
        return {
            'rmse': mean_squared_error(y_true, preds, squared=False),
            'mae': mean_absolute_error(y_true, preds),
            'r2': r2_score(y_true, preds)
        }

    print('\nRidge metrics:', metrics(y_test, pred_ridge))
    print('RandomForest metrics:', metrics(y_test, pred_rf))

    # Save ridge and scaler for quick demo
    out = ROOT / 'saved_models'
    out.mkdir(exist_ok=True)
    import joblib
    joblib.dump(ridge, out / 'ridge_demo.pkl')
    joblib.dump(scaler, out / 'scaler_demo.pkl')
    print('Saved ridge and scaler to', out)


if __name__ == '__main__':
    df = load_or_fetch()
    eda(df)
    df, feature_cols = create_features(df)
    # Feature-group step (explicit, no guards) — will fail if Hopsworks not configured
    # Uncomment the following line if you have Hopsworks configured and want to run it:
    # feature_group_actions()
    train_and_evaluate(df, feature_cols)
