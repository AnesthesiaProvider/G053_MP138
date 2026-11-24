"""
Data drift monitoring, group performance evaluation, model retraining, and promotion.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import yaml

from .data import clean_training_data, build_pipeline, compute_file_sha256, load_params, detect_data_issues


def compute_psi(expected: pd.Series, actual: pd.Series, num_bins: int = 10, eps: float = 1e-4) -> float:
    """
    Computes the Population Stability Index (PSI) between reference and current data using 10 quantile bins.
    """
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = expected.quantile(quantiles).values
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    exp_counts = pd.cut(expected, bins=bins).value_counts().sort_index()
    act_counts = pd.cut(actual, bins=bins).value_counts().sort_index()

    exp_pct = (exp_counts / len(expected)).replace(0, eps)
    act_pct = (act_counts / len(actual)).replace(0, eps)

    # Normalize proportions
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi_val, 4))


def run_monitoring_and_retraining(params: dict = None) -> dict:
    """
    Executes monitoring and retraining pipeline:
    1. Computes PSI between cleaned train.csv and later.csv.
    2. Measures v1 MAE on later.csv sliced by area.
    3. Retrains v2 on clean train + first 70% later.csv (sorted by date).
    4. Evaluates v1 and v2 on last 30% of later.csv.
    5. Implements promotion decision rule (>= 10% lower MAE).
    6. Registers v2 in MLflow, updates @champion alias if promoted.
    7. Updates findings.yaml and reports/findings.yaml.
    """
    if params is None:
        params = load_params()

    data_cfg = params["data"]
    model_cfg = params["model"]
    monitoring_cfg = params["monitoring"]

    train_path = data_cfg.get("raw_train_path", "data/raw/train.csv")
    if not os.path.exists(train_path):
        train_path = data_cfg.get("train_path", "data/train.csv")

    later_path = data_cfg.get("raw_later_path", "data/raw/later.csv")
    if not os.path.exists(later_path):
        later_path = data_cfg.get("later_path", "data/later.csv")

    holdout_path = data_cfg.get("raw_holdout_path", "data/raw/holdout_features.csv")
    if not os.path.exists(holdout_path):
        holdout_path = data_cfg.get("holdout_path", "data/holdout_features.csv")

    raw_train = pd.read_csv(train_path)
    clean_train = clean_training_data(raw_train)
    later_df = pd.read_csv(later_path)
    holdout_df = pd.read_csv(holdout_path)

    features = data_cfg["features"]
    numeric_features = data_cfg["numeric_features"]
    categorical_features = data_cfg["categorical_features"]
    target_col = data_cfg["target_column"]

    # 1. Compute PSI for all numeric columns
    psi_dict = {}
    for col in numeric_features:
        # Impute temporary missing values for PSI computation if any
        train_s = clean_train[col].fillna(clean_train[col].median())
        later_s = later_df[col].fillna(clean_train[col].median())
        psi_val = compute_psi(train_s, later_s, num_bins=monitoring_cfg.get("n_bins", 10))
        psi_dict[col] = psi_val

    most_drifted = max(psi_dict.items(), key=lambda x: x[1])[0]

    # Load existing v1 champion model
    model_path = model_cfg["model_save_path"]
    v1_model = joblib.load(model_path)

    # 2. Compute v1 MAE on later.csv separately for each area
    later_eval = later_df.copy()
    later_eval["v1_pred"] = v1_model.predict(later_eval[features])
    later_eval["v1_abs_err"] = np.abs(later_eval[target_col] - later_eval["v1_pred"])

    mae_by_group_series = later_eval.groupby("area")["v1_abs_err"].mean()
    mae_by_group = {
        "Market": float(round(mae_by_group_series.get("Market", 0.0), 4)),
        "Residential": float(round(mae_by_group_series.get("Residential", 0.0), 4)),
        "Highway": float(round(mae_by_group_series.get("Highway", 0.0), 4)),
    }
    worst_served = str(mae_by_group_series.idxmax())

    # 3. Sort later.csv by date and split into 70% train and 30% check
    later_sorted = later_df.sort_values(data_cfg.get("date_column", "date")).reset_index(drop=True)
    split_ratio = monitoring_cfg.get("split_ratio", 0.70)
    split_idx = int(split_ratio * len(later_sorted))

    later_train = later_sorted.iloc[:split_idx]
    later_test = later_sorted.iloc[split_idx:]

    # Retrain v2 on cleaned train.csv + first 70% of later.csv
    train_v2 = pd.concat([clean_train, later_train], ignore_index=True)
    v2_pipeline = build_pipeline(
        Ridge(alpha=model_cfg.get("ridge_alpha", 1.0), random_state=model_cfg.get("random_state", 42)),
        numeric_features,
        categorical_features
    )
    v2_pipeline.fit(train_v2[features], train_v2[target_col])

    # Evaluate v1 and v2 on check split (last 30%)
    v1_check_preds = v1_model.predict(later_test[features])
    v2_check_preds = v2_pipeline.predict(later_test[features])

    mae_v1_on_check = float(round(mean_absolute_error(later_test[target_col], v1_check_preds), 4))
    mae_v2_on_check = float(round(mean_absolute_error(later_test[target_col], v2_check_preds), 4))

    # 4. Promotion decision rule (promote if MAE is at least 10% lower)
    promo_thresh = model_cfg.get("promotion_threshold", 0.10)
    relative_reduction = (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check

    decision = "promote" if relative_reduction >= promo_thresh else "keep"

    # Log v2 to MLflow
    tracking_uri = model_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = model_cfg.get("experiment_name", "mp138_cash_demand_experiment")
    mlflow.set_experiment(exp_name)

    data_sha256 = compute_file_sha256(train_path)
    reg_model_name = model_cfg.get("registered_model_name", "mp138_atm_cash_demand")

    with mlflow.start_run(run_name="Retrained_v2_Model") as run:
        v2_run_id = run.info.run_id
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", "Ridge_v2")
        mlflow.log_param("retrained", True)
        mlflow.log_param("split_ratio", split_ratio)

        mlflow.log_metric("mae", mae_v2_on_check)
        mlflow.log_metric("mae_v1_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_check", mae_v2_on_check)
        mlflow.log_metric("relative_reduction", float(round(relative_reduction, 4)))

        mlflow.sklearn.log_model(
            v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )

        model_uri = f"runs:/{v2_run_id}/model"
        v2_version = mlflow.register_model(model_uri, reg_model_name)

    client = MlflowClient()
    if decision == "promote":
        champion_version = int(v2_version.version)
        client.set_registered_model_alias(reg_model_name, "champion", v2_version.version)
        joblib.dump(v2_pipeline, model_path)
        print(f"Promoted v2 to champion (Version {champion_version}) with alias '@champion'")
    else:
        champion_version = 1
        print("Kept v1 as champion.")

    # Detect the seven data issues
    data_issues = detect_data_issues(raw_train, later_df, holdout_df)

    # Get v1 best run id from MLflow
    best_v1_run = client.search_model_versions(f"name='{reg_model_name}'")
    v1_run_id = None
    for mv in best_v1_run:
        if str(mv.version) == "1":
            v1_run_id = mv.run_id
            break

    # Build findings structure
    findings_data = {
        "group_id": "G053",
        "problem_id": "MP138",
        "data_issues": data_issues,
        "tracking": {
            "experiment_name": exp_name,
            "best_run_id": v1_run_id,
            "registered_model_name": reg_model_name,
            "champion_version": champion_version,
        },
        "monitoring": {
            "psi": psi_dict,
            "most_drifted_feature": most_drifted,
            "mae_by_group": mae_by_group,
            "worst_served_group": worst_served,
            "mae_v1_on_check": mae_v1_on_check,
            "mae_v2_on_check": mae_v2_on_check,
            "decision": decision,
        }
    }

    # Save to findings.yaml and reports/findings.yaml
    for target_path in ["findings.yaml", "reports/findings.yaml"]:
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(findings_data, f, sort_keys=False, default_flow_style=False)

    print("Monitoring and retraining finished successfully. Findings updated.")
    return findings_data


if __name__ == "__main__":
    run_monitoring_and_retraining()
