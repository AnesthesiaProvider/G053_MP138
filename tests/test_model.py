"""
Unit tests for pipeline construction, training, prediction, and serialization.
Tests use synthetic in-memory data for portability.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.atm_demand.data import build_pipeline


def test_build_pipeline_and_predict():
    numeric_features = ["nearby_shops", "days_to_salary", "footfall_index", "distance_to_market_km", "avg_ticket_rs", "other_atms_1km"]
    categorical_features = ["atm_type", "area"]

    # Synthetic training data
    df = pd.DataFrame({
        "nearby_shops": [30, 50, 70, 90, 110],
        "days_to_salary": [5, 10, 15, 20, 25],
        "footfall_index": [40.0, 50.0, 60.0, 70.0, 80.0],
        "distance_to_market_km": [1.0, 2.0, 3.0, 4.0, 5.0],
        "avg_ticket_rs": [2000.0, 2500.0, 3000.0, 3500.0, 4000.0],
        "other_atms_1km": [2, np.nan, 4, 6, 8],
        "atm_type": ["standalone", "bank_branch", "mall", "standalone", "bank_branch"],
        "area": ["Market", "Residential", "Highway", "Market", "Residential"],
        "withdrawal_thousand_rs": [400.0, 500.0, 600.0, 700.0, 800.0]
    })

    features = numeric_features + categorical_features
    X = df[features]
    y = df["withdrawal_thousand_rs"]

    pipeline = build_pipeline(Ridge(alpha=1.0), numeric_features, categorical_features)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(df)
    assert not np.isnan(preds).any()


def test_model_serialization(tmp_path):
    numeric_features = ["nearby_shops", "days_to_salary"]
    categorical_features = ["area"]

    df = pd.DataFrame({
        "nearby_shops": [10, 20, 30],
        "days_to_salary": [1, 2, 3],
        "area": ["Market", "Highway", "Residential"],
        "target": [100.0, 200.0, 300.0]
    })

    pipeline = build_pipeline(Ridge(alpha=1.0), numeric_features, categorical_features)
    pipeline.fit(df[["nearby_shops", "days_to_salary", "area"]], df["target"])

    save_file = tmp_path / "test_model.joblib"
    joblib.dump(pipeline, save_file)
    assert os.path.exists(save_file)

    loaded = joblib.load(save_file)
    preds = loaded.predict(df[["nearby_shops", "days_to_salary", "area"]])
    assert len(preds) == 3
