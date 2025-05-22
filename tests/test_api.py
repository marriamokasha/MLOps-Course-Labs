import sys
import os

# Add the parent directory to sys.path so Python can find main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "model_loaded" in json_data
    assert "message" in json_data

def test_health():
    response = client.get("/health")
    # Depending on model load success, this might fail. Adjust if needed.
    if response.status_code == 503:
        assert response.json()["detail"] == "Models not loaded"
    else:
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["status"] == "OK"
        assert "model_name" in json_data

def test_predict():
    sample_input = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    json_data = response.json()
    assert "churn_prediction" in json_data
    assert "prediction_label" in json_data
    # Confidence is optional, so just check if present it is float
    if "confidence" in json_data:
        assert isinstance(json_data["confidence"], float)
