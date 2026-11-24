"""
MLflow experiment tracking, candidate training, evaluation, and model registration.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from .data import clean_training_data, build_pipeline, compute_file_sha256, load_params


def run_experiments(params: dict = None) -> str:
    """
    Executes MLflow experiments:
    1. Loads params from params.yaml.
    2. Sets SQLite tracking URI (sqlite:///mlflow.db).
    3. Computes and logs data_sha256 in every run.
    4. Trains 3 candidate models (Ridge with different regularization parameters).
    5. Logs MAE and metrics.
    6. Registers the best v1 model as Version 1 with champion alias.
    7. Saves champion model to models/model.joblib.
    """
    if params is None:
        params = load_params()

    data_cfg = params["data"]
    model_cfg = params["model"]

    # Resolve train path
    train_path = data_cfg.get("raw_train_path", "data/raw/train.csv")
    if not os.path.exists(train_path):
        train_path = data_cfg.get("train_path", "data/train.csv")

    data_sha256 = compute_file_sha256(train_path)

    # Load and clean training data
    raw_df = pd.read_csv(train_path)
    clean_df = clean_training_data(raw_df)

    features = data_cfg["features"]
    numeric_features = data_cfg["numeric_features"]
    categorical_features = data_cfg["categorical_features"]
    target_col = data_cfg["target_column"]

    X = clean_df[features]
    y = clean_df[target_col]

    # Setup MLflow
    tracking_uri = model_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = model_cfg.get("experiment_name", "mp138_cash_demand_experiment")
    mlflow.set_experiment(exp_name)

    candidates = [
        {"model_type": "Ridge", "alpha": 0.1},
        {"model_type": "Ridge", "alpha": 1.0},
        {"model_type": "Ridge", "alpha": 10.0},
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=model_cfg.get("random_state", 42))

    best_mae = float("inf")
    best_run_id = None
    best_pipeline = None

    for cand in candidates:
        alpha = cand["alpha"]
        run_name = f"Ridge_alpha_{alpha}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log required parameters
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", "Ridge")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("n_splits", 5)

            # 5-fold CV evaluation
            cv_maes = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                pipe_cv = build_pipeline(
                    Ridge(alpha=alpha, random_state=model_cfg.get("random_state", 42)),
                    numeric_features,
                    categorical_features
                )
                pipe_cv.fit(X_tr, y_tr)
                preds = pipe_cv.predict(X_val)
                cv_maes.append(mean_absolute_error(y_val, preds))

            mean_mae = float(np.mean(cv_maes))

            # Train on full clean training set
            pipeline = build_pipeline(
                Ridge(alpha=alpha, random_state=model_cfg.get("random_state", 42)),
                numeric_features,
                categorical_features
            )
            pipeline.fit(X, y)
            train_preds = pipeline.predict(X)
            train_mae = float(mean_absolute_error(y, train_preds))
            train_r2 = float(r2_score(y, train_preds))

            # Log metrics required by handbook
            mlflow.log_metric("mae", mean_mae)
            mlflow.log_metric("cv_mae", mean_mae)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("r2", train_r2)

            # Log model with cloudpickle serialization format
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            if mean_mae < best_mae:
                best_mae = mean_mae
                best_run_id = run_id
                best_pipeline = pipeline

    # Register best v1 model
    reg_model_name = model_cfg.get("registered_model_name", "mp138_atm_cash_demand")
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri, reg_model_name)

    client = MlflowClient()
    client.set_registered_model_alias(reg_model_name, "champion", model_version.version)

    # Save to models/model.joblib
    os.makedirs(os.path.dirname(model_cfg["model_save_path"]), exist_ok=True)
    joblib.dump(best_pipeline, model_cfg["model_save_path"])

    print(f"MLflow experiments completed. Best Run ID: {best_run_id}, Best CV MAE: {best_mae:.4f}")
    print(f"Registered model '{reg_model_name}' version {model_version.version} with alias 'champion'")

    return best_run_id


if __name__ == "__main__":
    run_experiments()
