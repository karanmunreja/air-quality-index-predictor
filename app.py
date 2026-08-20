from fastapi import FastAPI, HTTPException
import os
import requests
from dotenv import load_dotenv

from src.data.features.feature_view import get_feature_view


HOPSWORKS_ENDPOINT = (
    "http://57.130.17.185/v1/jshsmekedaxakb/"
    "aqiforecastmulti/v1/models/aqiforecastmulti:predict"
)


load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


app = FastAPI(
    title="AQI Forecast API",
    version="1.0.0"
)


prediction_fv = get_feature_view()


@app.get("/")
def home():

    return {
        "message": "AQI Forecast API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict():

    try:

        # =========================================
        # 1. Fetch LATEST Lahore row from FV
        # =========================================

        features = prediction_fv.get_feature_vector(
            entry={
                "city": "Lahore"
            },
            return_type="pandas"
        )

        if features is None or features.empty:

            raise HTTPException(
                status_code=404,
                detail="Latest feature vector not found"
            )

        # =========================================
        # 2. Print fetched row for evaluation
        # =========================================

        print("\n==============================================")
        print("LATEST ROW FETCHED FROM FEATURE VIEW")
        print("==============================================")

        print(
            features.to_string(index=False)
        )

        print("==============================================\n")

        # =========================================
        # 3. Remove non-model columns
        # =========================================

        X = features.drop(
            columns=[
                "time",
                "city",
                "minute",
                "target_aqi_24",
                "target_aqi_48",
                "target_aqi_72"
            ],
            errors="ignore"
        )

        print("Model input shape:", X.shape)

        # =========================================
        # 4. Send latest row to deployed model
        # =========================================

        response = requests.post(
            HOPSWORKS_ENDPOINT,
            headers={
                "authorization": f"ApiKey {HOPSWORKS_API_KEY}",
                "content-type": "application/json"
            },
            json={
                "inputs": X.values.tolist()
            }
        )

        response.raise_for_status()

        # =========================================
        # 5. Return prediction
        # =========================================

        return response.json()


    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Hopsworks model request failed: {str(e)}"
        )


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )