from src.data.features.feature_store.hopswork_client import connect


def register_model(model_name, model_path, algorithm,
                    r2_24, rmse_24, mae_24,
                    r2_48, rmse_48, mae_48,
                    r2_72, rmse_72, mae_72,
                    average_r2):
    project = connect()
    model_registry = project.get_model_registry()

    model = model_registry.python.create_model(
        name=model_name,
        metrics={
            "R2_24": float(r2_24), "RMSE_24": float(rmse_24), "MAE_24": float(mae_24),
            "R2_48": float(r2_48), "RMSE_48": float(rmse_48), "MAE_48": float(mae_48),
            "R2_72": float(r2_72), "RMSE_72": float(rmse_72), "MAE_72": float(mae_72),
            "Average_R2": float(average_r2)
        },
        description=f"AQI Forecasting Model — algorithm: {algorithm}"
    )
    model.save(model_path)
    print(f"Uploaded model {model_name}, version {model.version}")
    return model