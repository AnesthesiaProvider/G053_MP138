"""
Unit tests for data cleaning, preprocessing, and anomaly detection.
Tests run in CI without needing git-ignored CSV files.
"""

import numpy as np
import pandas as pd
import pytest

from src.atm_demand.data import clean_training_data, detect_data_issues
from src.atm_demand.monitor import compute_psi


def create_synthetic_train_df():
    """Generates synthetic dataset replicating train data issues."""
    data = {
        "record_id": [f"R{i:05d}" for i in range(1, 11)],
        "date": ["2026-01-01"] * 5 + ["2026-01-02"] * 5,
        "nearby_shops": [30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        "days_to_salary": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "footfall_index": [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
        "distance_to_market_km": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
        "avg_ticket_rs": [2500.0, 2600.0, 2.8, 3.1, 2700.0, 2800.0, 2.5, 3.0, 2900.0, 3000.0],
        "other_atms_1km": [2, 3, -999, 5, 6, 2, -999, 4, 5, 6],
        "atm_type": ["standalone", "bank_branch", "mall", "standalone", "bank_branch"] * 2,
        "area": ["Market", "Residential", "Highway", "Highway", "Market", "Residential", "Highway", "Highway", "Market", "Residential"],
        "withdrawal_thousand_rs": [500.0, 600.0, 700.0, 5500.0, 650.0, 750.0, 800.0, 8500.0, 900.0, 950.0]
    }
    df = pd.DataFrame(data)
    # Add 2 duplicate rows
    dups = df.iloc[[0, 1]].copy()
    return pd.concat([df, dups], ignore_index=True)


def test_clean_training_data():
    raw_df = create_synthetic_train_df()
    assert len(raw_df) == 12

    cleaned = clean_training_data(raw_df)

    # 1. Deduplication
    assert len(cleaned) == 10

    # 2. Unit mixup correction for Highway
    highway_tickets = cleaned[cleaned["area"] == "Highway"]["avg_ticket_rs"]
    assert (highway_tickets >= 1000.0).all()

    # 3. Target typo correction (> 4000 divided by 10)
    assert (cleaned["withdrawal_thousand_rs"] <= 1500.0).all()
    assert cleaned.loc[3, "withdrawal_thousand_rs"] == 550.0
    assert cleaned.loc[7, "withdrawal_thousand_rs"] == 850.0

    # 4. Sentinel value handling
    assert (cleaned["other_atms_1km"] == -999).sum() == 0
    assert cleaned["other_atms_1km"].isna().sum() == 2


def test_detect_data_issues():
    train_df = create_synthetic_train_df()
    later_df = train_df.drop_duplicates().copy()
    later_df.loc[later_df["area"] == "Highway", "avg_ticket_rs"] *= 1000
    later_df.loc[later_df["withdrawal_thousand_rs"] > 4000, "withdrawal_thousand_rs"] /= 10
    holdout_df = later_df.drop(columns=["withdrawal_thousand_rs"])

    issues = detect_data_issues(train_df, later_df, holdout_df)

    assert issues["leaking_column"]["present"] is False
    assert issues["unit_mixup"]["present"] is True
    assert issues["unit_mixup"]["column"] == "avg_ticket_rs"
    assert issues["unit_mixup"]["group_value"] == "Highway"
    assert issues["unit_mixup"]["factor"] == 0.001
    assert issues["sentinel_value"]["present"] is True
    assert issues["sentinel_value"]["column"] == "other_atms_1km"
    assert issues["sentinel_value"]["value"] == -999
    assert issues["duplicate_rows"]["present"] is True
    assert issues["duplicate_rows"]["count"] == 2
    assert issues["target_typos"]["present"] is True
    assert issues["target_typos"]["count"] == 2


def test_compute_psi():
    np.random.seed(42)
    s1 = pd.Series(np.random.normal(50, 10, 500))
    # Identical distribution -> low PSI
    s2 = pd.Series(np.random.normal(50, 10, 500))
    psi_identical = compute_psi(s1, s2, num_bins=10)
    assert psi_identical < 0.2

    # Drastically shifted distribution -> high PSI
    s3 = pd.Series(np.random.normal(100, 10, 500))
    psi_shifted = compute_psi(s1, s3, num_bins=10)
    assert psi_shifted > 1.0
