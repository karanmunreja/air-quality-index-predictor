import os
import joblib


class Predictor:

    def __init__(self):

        model_path = os.path.join(
            os.environ["MODEL_FILES_PATH"],
            "aqi_forecast_multi.pkl"
        )

        self.model = joblib.load(model_path)

        print("AQI multi-output model loaded successfully")

    def predict(self, inputs):

        predictions = self.model.predict(inputs)
        
        return predictions.tolist()