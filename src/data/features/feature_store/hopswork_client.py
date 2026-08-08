import os
import hopsworks
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv('HOPSWORKS_API_KEY')

def connect():
    if not API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is not set.")

    project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",       # DNS of your Hopsworks instance
    project="jshsmekedaxakb", 
    engine="python" ,              # Name of your Hopsworks project
    api_key_value=API_KEY  # Hopsworks API key value
)   
    return project

def get_feature_store():
    project=connect()
    fs = project.get_feature_store()  
    return fs

def get_feature_group():
    fs=get_feature_store()
    featureGroup=fs.get_or_create_feature_group(
    name="aqi_training_features",
    version=1,
    primary_key=["city","time"],
    description="Historical weather and air quality features for AQI prediction",
    online_enabled=True,
    time_travel_format="HUDI"
    )
    return featureGroup

def insert_features(df):
    feature_group=get_feature_group()
    feature_group.insert(df,write_options={"wait_for_job":True})
