from src.data.features.feature_store.hopswork_client import connect

MODEL_NAME = "aqi_forecast_multi"
DEPLOYMENT_NAME = "aqiforecastmulti"


def deploy_model():
    project = connect()
    mr = project.get_model_registry()

    # Pick the best-performing version, not just the newest
    model = mr.get_best_model(MODEL_NAME, "Average_R2", "max")
    print(f"Best model selected: version {model.version} (Average_R2={model.training_metrics.get('Average_R2')})")

    dataset_api = project.get_dataset_api()
    dataset_api.upload("predictor.py", "Resources", overwrite=True)
    predictor_script_path = f"/Projects/{project.name}/Resources/predictor.py"

    ms = project.get_model_serving()

    try:
        existing = ms.get_deployment(DEPLOYMENT_NAME)
        print("Stopping and removing previous deployment...")
        existing.stop()
        existing.delete()
    except Exception:
        print("No previous deployment found, creating fresh one.")

    predictor = ms.create_predictor(
        model=model,
        name=DEPLOYMENT_NAME,
        script_file=predictor_script_path
    )

    deployment = predictor.deploy()
    deployment.start()

    print(f"\nDeployed {MODEL_NAME} v{model.version} as '{DEPLOYMENT_NAME}'")
    print("Deployment state:")
    deployment.get_state().describe()

    return deployment


if __name__ == "__main__":
    deploy_model()