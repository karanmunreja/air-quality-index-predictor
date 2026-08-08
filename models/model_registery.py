import hopsworks
import os
from dotenv import load_dotenv

def register_model(model_name,model_path,r2_Score,rmse,mae):
    load_dotenv
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
           'R2_Score':float(r2_Score),
           'RMSE': float(rmse),
           'MAE':float(mae)
        }, description="AQI Forecasting Model"
    )
    model.save(model_path)
    print('uploaded model')