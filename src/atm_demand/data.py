"""
Data preprocessing, cleaning, anomaly detection, and pipeline building utilities.
"""

import hashlib
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_params(params_path: str = "params.yaml") -> dict:
    """Load settings from params.yaml using yaml.safe_load."""
    if not os.path.exists(params_path):
        # Fallback if called from subfolder
        alt_path = os.path.join(os.path.dirname(__file__), "..", "..", params_path)
        if os.path.exists(alt_path):
            params_path = alt_path
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_file_sha256(filepath: str) -> str:
    """Compute the SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        alt = os.path.join("data", os.path.basename(filepath))
        if os.path.exists(alt):
            filepath = alt
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def detect_data_issues(train_df: pd.DataFrame, later_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    """
    Examine the seven potential data problems defined in the course handbook:
    1. leaking_column
    2. unit_mixup
    3. sentinel_value
    4. duplicate_rows
    5. stuck_sensor
    6. target_typos
    7. spelling_variants
    """
    issues = {}

    # 1. Leaking column
    target_col = "withdrawal_thousand_rs"
    extra_cols = [c for c in train_df.columns if c not in holdout_df.columns and c != target_col]
    leaking_present = len(extra_cols) > 0
    issues["leaking_column"] = {
        "present": leaking_present,
        "column": extra_cols[0] if leaking_present else None
    }

    # 2. Unit mixup
    # Compare min/mean of avg_ticket_rs by area
    unit_present = False
    unit_col = None
    group_val = None
    factor = None
    if "avg_ticket_rs" in train_df.columns and "area" in train_df.columns:
        means = train_df.groupby("area")["avg_ticket_rs"].mean()
        for area_name, mean_val in means.items():
            if mean_val < 50.0:  # typical values are ~2800 rupees, recorded as ~2.8
                unit_present = True
                unit_col = "avg_ticket_rs"
                group_val = area_name
                # wrong value = right value x factor => 2.85 = 2850 x 0.001
                factor = 0.001
                break
    issues["unit_mixup"] = {
        "present": unit_present,
        "column": unit_col,
        "group_value": group_val,
        "factor": factor
    }

    # 3. Sentinel value
    sentinel_present = False
    sentinel_col = None
    sentinel_val = None
    sentinel_cnt = None
    for col in ["other_atms_1km", "nearby_shops", "footfall_index"]:
        if col in train_df.columns:
            counts = train_df[train_df[col].isin([-999, -1, 9999])]
            if len(counts) > 0:
                sentinel_present = True
                sentinel_col = col
                val = counts[col].iloc[0]
                sentinel_val = int(val)
                sentinel_cnt = int(len(counts))
                break
    issues["sentinel_value"] = {
        "present": sentinel_present,
        "column": sentinel_col,
        "value": sentinel_val,
        "count": sentinel_cnt
    }

    # 4. Duplicate rows
    dup_count = int(train_df.duplicated().sum())
    issues["duplicate_rows"] = {
        "present": dup_count > 0,
        "count": dup_count
    }

    # 5. Stuck sensor
    stuck_present = False
    stuck_col = None
    stuck_start = None
    stuck_end = None
    if "date" in train_df.columns:
        train_sorted = train_df.sort_values("date")
        for col in ["nearby_shops", "footfall_index", "distance_to_market_km"]:
            if col in train_sorted.columns:
                streaks = (train_sorted[col] != train_sorted[col].shift()).cumsum()
                streak_sizes = train_sorted.groupby(streaks).size()
                if (streak_sizes > 20).any():
                    stuck_present = True
                    stuck_col = col
                    break
    issues["stuck_sensor"] = {
        "present": stuck_present,
        "column": stuck_col,
        "start_date": stuck_start,
        "end_date": stuck_end
    }

    # 6. Target typos
    typo_present = False
    typo_count = None
    if target_col in train_df.columns:
        # Values 10 times too big (normal max ~1100, typos > 4000)
        typos = train_df[train_df[target_col] > 4000]
        if len(typos) > 0:
            typo_present = True
            typo_count = int(len(typos))
    issues["target_typos"] = {
        "present": typo_present,
        "count": typo_count
    }

    # 7. Spelling variants
    spelling_present = False
    spelling_col = None
    true_levels = None
    for col in ["atm_type", "area"]:
        if col in train_df.columns and col in later_df.columns:
            train_cats = set(train_df[col].dropna().unique())
            later_cats = set(later_df[col].dropna().unique())
            if len(train_cats - later_cats) > 0:
                spelling_present = True
                spelling_col = col
                true_levels = len(later_cats)
                break
    issues["spelling_variants"] = {
        "present": spelling_present,
        "column": spelling_col,
        "true_levels": true_levels
    }

    return issues


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans training data:
    1. Removes duplicate rows.
    2. Corrects unit mixup in avg_ticket_rs for Highway (multiplies by 1000).
    3. Corrects target typos (> 4000 divided by 10).
    4. Replaces sentinel values (-999 in other_atms_1km) with NaN for imputation.
    """
    cleaned = df.copy()

    # 1. Deduplicate
    cleaned = cleaned.drop_duplicates()

    # 2. Fix unit mixup (Highway recorded in thousands of rupees instead of rupees)
    if "avg_ticket_rs" in cleaned.columns and "area" in cleaned.columns:
        mask = (cleaned["area"] == "Highway") & (cleaned["avg_ticket_rs"] < 100.0)
        cleaned.loc[mask, "avg_ticket_rs"] = cleaned.loc[mask, "avg_ticket_rs"] * 1000.0

    # 3. Fix target typos (typed with an extra zero: 10 times too big)
    if "withdrawal_thousand_rs" in cleaned.columns:
        mask_typo = cleaned["withdrawal_thousand_rs"] > 4000.0
        cleaned.loc[mask_typo, "withdrawal_thousand_rs"] = cleaned.loc[mask_typo, "withdrawal_thousand_rs"] / 10.0

    # 4. Handle sentinel values (-999)
    if "other_atms_1km" in cleaned.columns:
        cleaned.loc[cleaned["other_atms_1km"] == -999, "other_atms_1km"] = np.nan

    return cleaned


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Constructs sklearn ColumnTransformer for numeric and categorical features."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )
    return preprocessor


def build_pipeline(model, numeric_features: list, categorical_features: list) -> Pipeline:
    """Constructs a complete scikit-learn pipeline with preprocessing and model."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
