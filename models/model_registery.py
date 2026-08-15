import hopsworks
import os
from dotenv import load_dotenv

def register_model(model_name,model_path, r2_24,rmse_24,
    mae_24,
    r2_48,
    rmse_48,
    mae_48,
    r2_72,
    rmse_72,
    mae_72,
    average_r2):
    load_dotenv()
    API_KEY=os.getenv('HOPSWORKS_API_KEY')
    project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",       # DNS of your Hopsworks instance
    project="jshsmekedaxakb", 
    engine="python" ,              # Name of your Hopsworks project
    api_key_value=API_KEY  # Hopsworks API key value
)   
    model_registery=project.get_model_registry()
    model=model_registery.python.create_model(
        name=model_name,
        metrics={
            "R2_24": float(r2_24),
            "RMSE_24": float(rmse_24),
            "MAE_24": float(mae_24),

            "R2_48": float(r2_48),
            "RMSE_48": float(rmse_48),
            "MAE_48": float(mae_48),

            "R2_72": float(r2_72),
            "RMSE_72": float(rmse_72),
            "MAE_72": float(mae_72),

            "Average_R2": float(average_r2)
        },
         description="AQI Forecasting Model"
    )
    model.save(model_path)
    print('uploaded model')