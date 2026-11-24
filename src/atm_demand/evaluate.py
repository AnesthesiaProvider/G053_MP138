"""
Inference and evaluation utilities for generating final holdout predictions.
"""

import os
import joblib
import pandas as pd
from .data import load_params


def generate_holdout_predictions(
    model_path: str = None,
    holdout_path: str = None,
    output_path: str = "predictions/holdout_predictions.csv",
    params: dict = None
) -> pd.DataFrame:
    """
    Loads champion model, predicts withdrawal_thousand_rs for every row of holdout_features.csv,
    and saves output with exactly two columns: record_id,prediction.
    """
    if params is None:
        params = load_params()

    if model_path is None:
        model_path = params["model"]["model_save_path"]

    if holdout_path is None:
        holdout_path = params["data"].get("raw_holdout_path", "data/raw/holdout_features.csv")
        if not os.path.exists(holdout_path):
            holdout_path = params["data"].get("holdout_path", "data/holdout_features.csv")

    model = joblib.load(model_path)
    holdout_df = pd.read_csv(holdout_path)

    features = params["data"]["features"]
    id_col = params["data"].get("id_column", "record_id")

    preds = model.predict(holdout_df[features])

    results_df = pd.DataFrame({
        "record_id": holdout_df[id_col],
        "prediction": preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Generated holdout predictions for {len(results_df)} records saved to {output_path}")
    return results_df


if __name__ == "__main__":
    generate_holdout_predictions()
