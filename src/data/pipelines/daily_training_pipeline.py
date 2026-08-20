from models.lstm import preprocess_lstm_feat
from models.model_registery import register_model
from models.preprocessing import create_target, load_training_data, prepare_data
from models.preprocessing import split_data
from models.training_pipeline import train_and_get_best_model
import os
import joblib

def run_daily_training_pipeline():
    df = load_training_data()
    df = create_target(df)
    df = df.sort_values("time").reset_index(drop=True)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = preprocess_lstm_feat(X_train, X_test, y_train, y_test)

    best_model_name, best_model, best = train_and_get_best_model(
        X_train, X_test, y_train, y_test,
        X_train_seq, X_test_seq, y_train_seq, y_test_seq
    )

    print(f"\nBest model: {best_model_name}\n")

    os.makedirs("saved_models", exist_ok=True)
    filename = f"saved_models/aqi_forecast_{best_model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(best_model, filename)

    register_model(
        model_name=f"aqi_forecast_{best_model_name.lower().replace(' ', '_')}",
        model_path=filename,
        r2_24=best["r2_24"], rmse_24=best["rmse_24"], mae_24=best["mae_24"],
        r2_48=best["r2_48"], rmse_48=best["rmse_48"], mae_48=best["mae_48"],
        r2_72=best["r2_72"], rmse_72=best["rmse_72"], mae_72=best["mae_72"],
        average_r2=best["average_r2"]
    )
    if __name__ == "__main__":
        run_daily_training_pipeline()