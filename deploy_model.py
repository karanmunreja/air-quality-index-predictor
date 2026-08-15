from src.data.features.feature_store.hopswork_client import connect
import os

def deploy_model():

    project = connect()
    mr = project.get_model_registry()

    model = mr.get_model(
        name="aqi_forecast_multi",
        version=1
    )
    
    dataset_api = project.get_dataset_api()
    dataset_api.upload(
        "predictor.py",
        "Resources",
        overwrite=True
    )

    # Build the full path explicitly instead of relying on upload()'s return value
    predictor_script_path = f"/Projects/{project.name}/Resources/predictor.py"

    print("Predictor uploaded:")
    print(predictor_script_path)

    # Model Serving
    ms = project.get_model_serving()

    predictor = ms.create_predictor(
        model=model,
        name="aqiforecastmulti",
        script_file=predictor_script_path
    )

    deployment = predictor.deploy()

    print("\nDeployment created successfully.")
    print("\nDeployment state:")
    deployment.get_state().describe()

    return deployment


if __name__ == "__main__":
    deploy_model()