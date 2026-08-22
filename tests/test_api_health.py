from fastapi.testclient import TestClient

from app import app


def test_home_endpoint_is_available():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AQI Forecast API is running"


def test_health_endpoint_is_available():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
