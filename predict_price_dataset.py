"""
predict_price_dataset.py

Use a trained pipeline saved in `price_model.joblib` to generate price
predictions for every row in `ads_dataset_20k.xlsx` (including the rows that
were excluded from training).

The script appends a **Predicted Price** column and writes the full 20 k‑row
DataFrame to `ads_with_predictions.xlsx`.  If a row already has a ground‑truth
Price value, the script also prints MAE / RMSE / MaxAbsError on those rows for a
quick sanity check.

Usage
-----
$ pip install pandas numpy scikit-learn joblib
$ python predict_price_dataset.py \
        --model price_model.joblib \
        --data ads_dataset_20k.xlsx \
        --output ads_with_predictions.xlsx
"""

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def add_to_path():
    """
    Adds the project root directory to Python path to enable imports from any project module.
    Should be called at the start of scripts that need to import from other project modules.
    """
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

add_to_path()

from helpers.data_manip import load_data

from sklearn.base import BaseEstimator, RegressorMixin

class LogTransformedRegressor(BaseEstimator, RegressorMixin):
    """Wrapper that handles log transformation internally"""
    def __init__(self, regressor):
        self.regressor = regressor
    
    def fit(self, X, y):
        # Transform y to log scale for training
        self.y_min_ = y.min()
        log_y = np.log(y)
        self.regressor.fit(X, log_y)
        return self
    
    def predict(self, X):
        # Get predictions and convert back to normal scale
        log_pred = self.regressor.predict(X)
        return np.exp(log_pred)

def parse_args():
    p = argparse.ArgumentParser(description="Batch‑predict property prices.")
    p.add_argument("--model", required=True, help="Path to price_model.joblib")
    p.add_argument(
        "--data", required=True, help="Path to ads_dataset_20k.xlsx (or .csv)"
    )
    p.add_argument(
        "--output",
        default="ads_with_predictions.xlsx",
        help="Where to save the file with predictions (default: ads_with_predictions.xlsx)",
    )
    return p.parse_args()


def load_features(df: pd.DataFrame, feature_list=None):
    """Return feature frame with specified features or all numeric features if not specified."""
    if feature_list is not None:
        # Use the provided feature list
        return df[feature_list].copy()
    else:
        # Fallback to original behavior
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["price", "price_per_sqm"]]
        return df[feature_cols]


def main():
    args = parse_args()

    print("Loading model …")
    model_info = joblib.load(args.model)
    # Extract the actual model from the saved dictionary
    model = model_info['model']
    features = model_info['features']
    # Turn off verbose output for prediction
    model.regressor.set_params(verbose=0)
    print(f"Model loaded (trained on {model_info['timestamp']})")
    print(f"Using features: {', '.join(features)}")

    print("Loading data …")
    data_path = Path(args.data)
    df = pd.read_parquet(data_path)
    df = df[df['price'] >= 20000]

    X = load_features(df, features)

    print("Predicting …")
    preds = model.predict(X)
    df["predicted_price"] = preds

    # --------------------------------------------------
    # Quick metrics on rows that have ground truth Price
    # --------------------------------------------------
    if "price" in df.columns:
        mask = df["price"].notna()
        if mask.any():
            y_true = df.loc[mask, "price"].values
            y_pred = preds[mask]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            max_err = np.max(np.abs(y_true - y_pred))
            print("\nMetrics on rows with ground‑truth Price:")
            print(f"  MAE       : {mae:,.0f} €")
            print(f"  RMSE      : {rmse:,.0f} €")
            print(f"  MaxAbsErr : {max_err:,.0f} €")
        else:
            print("No ground‑truth prices found – skipped metric calculation.")

    # ---------------------------
    # Save predictions to disk
    # ---------------------------
    output_path = Path(args.output)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
