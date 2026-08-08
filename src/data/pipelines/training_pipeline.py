import joblib
import os
from models.lstm import preprocess_lstm_feat
from models.model_registery import register_model
from models.preprocessing import  create_target, prepare_data, split_data
from models.training_pipeline import train_and_get_best_model


def run_training_pipeline(df):
    df=create_target(df)
    df = df.sort_values("time").reset_index(drop=True)
    TARGETS = [
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72"
    ]
    MODEL_NAMES = {
    "target_aqi_24": "aqi_forecast_24",
    "target_aqi_48": "aqi_forecast_48",
    "target_aqi_72": "aqi_forecast_72",
    }
    for target in TARGETS:
        print(f"\nTraining for {target}\n")
        X, y = prepare_data(df, target)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = preprocess_lstm_feat( X_train, X_test,y_train,y_test)
        best_model_name,best_model,best=train_and_get_best_model(X_train,X_test,y_train,y_test,X_train_seq,X_test_seq, y_train_seq, y_test_seq, MODEL_NAMES[target])
        os.makedirs("saved_models", exist_ok=True)
        filename = f"saved_models/{MODEL_NAMES[target]}.pkl"
        joblib.dump(best_model, filename)
        register_model(
        model_name=MODEL_NAMES[target],
        model_path=filename,
        r2_Score=best["r2_Score"],
        rmse=best["rmse"],
        mae=best["mae"])