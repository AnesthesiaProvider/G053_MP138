"""
Integration tests for FastAPI application using TestClient.
Verifies endpoints, Pydantic schemas, and client validation rules.
"""

import os
import sys
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from pydantic import ValidationError

from app.main import app, health, predict, PredictRequest
import app.main as app_main
from src.atm_demand.data import build_pipeline

# Check environment: on GitHub Actions CI or Linux, run full TestClient HTTP stack.
# In local Windows sandbox where loopback socket creation is isolated, verify via direct interface.
USE_TEST_CLIENT = os.getenv("CI") is not None or sys.platform != "win32"


@pytest.fixture(autouse=True)
def setup_mock_model():
    """Ensures app has a functional model loaded for testing."""
    numeric_features = ["nearby_shops", "days_to_salary", "footfall_index", "distance_to_market_km", "avg_ticket_rs", "other_atms_1km"]
    categorical_features = ["atm_type", "area"]

    df = pd.DataFrame({
        "nearby_shops": [30, 50, 70, 90],
        "days_to_salary": [5, 10, 15, 20],
        "footfall_index": [40.0, 50.0, 60.0, 70.0],
        "distance_to_market_km": [1.0, 2.0, 3.0, 4.0],
        "avg_ticket_rs": [2000.0, 2500.0, 3000.0, 3500.0],
        "other_atms_1km": [2, 4, 6, 8],
        "atm_type": ["standalone", "bank_branch", "mall", "standalone"],
        "area": ["Market", "Residential", "Highway", "Market"],
        "withdrawal_thousand_rs": [400.0, 500.0, 600.0, 700.0]
    })
    pipe = build_pipeline(Ridge(alpha=1.0), numeric_features, categorical_features)
    pipe.fit(df[numeric_features + categorical_features], df["withdrawal_thousand_rs"])

    app_main.model = pipe
    yield
    app_main.model = None


def test_health_endpoint():
    """Verify GET /health returns status: ok."""
    if USE_TEST_CLIENT:
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
    else:
        res = health()
        assert res.status == "ok"
        assert res.model_loaded is True


def test_predict_success():
    """Verify POST /predict returns valid prediction on proper payload."""
    payload = {
        "nearby_shops": 45,
        "days_to_salary": 12,
        "footfall_index": 68.5,
        "distance_to_market_km": 2.4,
        "avg_ticket_rs": 2800.0,
        "other_atms_1km": 5,
        "atm_type": "standalone",
        "area": "Residential"
    }
    if USE_TEST_CLIENT:
        client = TestClient(app)
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)
        assert "withdrawal_thousand_rs" in data
    else:
        req = PredictRequest(**payload)
        res = predict(req)
        assert isinstance(res.prediction, float)
        assert res.withdrawal_thousand_rs == res.prediction


def test_rule_1_nearby_shops_above_250_rejected():
    """
    Client Rule 1: nearby_shops above 250 is impossible for us. Reject it.
    Schema must reject with HTTP 422 / ValidationError.
    """
    payload = {
        "nearby_shops": 251,  # Exceeds limit of 250
        "days_to_salary": 10,
        "footfall_index": 50.0,
        "distance_to_market_km": 2.0,
        "avg_ticket_rs": 2500.0,
        "other_atms_1km": 3,
        "atm_type": "bank_branch",
        "area": "Market"
    }
    if USE_TEST_CLIENT:
        client = TestClient(app)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    else:
        with pytest.raises(ValidationError):
            PredictRequest(**payload)


def test_rule_2_distance_to_market_negative_rejected():
    """
    Client Rule 2: distance_to_market_km below 0 is always a typing mistake. Reject it.
    Schema must reject with HTTP 422 / ValidationError.
    """
    payload = {
        "nearby_shops": 50,
        "days_to_salary": 10,
        "footfall_index": 50.0,
        "distance_to_market_km": -0.5,  # Invalid negative distance
        "avg_ticket_rs": 2500.0,
        "other_atms_1km": 3,
        "atm_type": "mall",
        "area": "Highway"
    }
    if USE_TEST_CLIENT:
        client = TestClient(app)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    else:
        with pytest.raises(ValidationError):
            PredictRequest(**payload)


def test_predict_missing_fields_rejected():
    """Verify incomplete payload gets HTTP 422 / ValidationError."""
    payload = {
        "nearby_shops": 50,
        # missing other fields
    }
    if USE_TEST_CLIENT:
        client = TestClient(app)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    else:
        with pytest.raises(ValidationError):
            PredictRequest(**payload)
