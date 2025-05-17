"""
predict_price_dataset.py

Use a trained pipeline saved in `price_model.joblib` to generate price
predictions for every row in `ads_dataset_20k.xlsx` (including the rows that
were excluded from training).

Usage
-----
$ python predict_price_dataset.py --model path/to/model.joblib --data input.parquet --output predictions.xlsx
"""

import click
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def add_to_path():
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent
    sys.path.insert(0, str(PROJECT_ROOT))

add_to_path()

from helpers.data_manip import load_data
from helpers.paths import Files, Dirs
from models.transformers import TransformedRegressor

def load_features(df: pd.DataFrame, feature_list=None):
    if feature_list is not None:
        return df[feature_list].copy()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["price", "price_per_sqm"]]
        return df[feature_cols]

@click.command()
@click.option('--model', default=Files.PRICE_MODEL, type=click.Path(exists=True), help="Path to trained model .joblib file")
@click.option('--data', default=Files.CLEAN_DATASET, type=click.Path(exists=True), help="Path to input dataset (.parquet or .csv)")
@click.option('--output', default="ads_with_predictions.xlsx", help="Output path (default: ads_with_predictions.xlsx)")
def predict(model, data, output):
    """Use a trained pipeline to predict property prices and save results."""
    print("Loading model …")
    model_info = joblib.load(model)
    model = model_info['model']
    features = model_info['features']
    config = model_info.get('config', {})  # Get config if available (for backward compatibility)
    
    # Turn off verbosity for prediction
    if hasattr(model, 'regressor'):
        model.regressor.set_params(verbose=0)
    
    print(f"Model loaded (trained on {model_info['timestamp']})")
    print(f"Using features: {', '.join(features)}")
    if config:
        print("Model configuration:", config)

    print("Loading data …")
    data_path = Path(data)
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet.")

    df = df[df['price'] >= 20000]
    X = load_features(df, features)

    print("Predicting …")
    preds = model.predict(X)
    df["predicted_price"] = preds

    if "price" in df.columns:
        mask = df["price"].notna()
        if mask.any():
            y_true = df.loc[mask, "price"].values
            y_pred = preds[mask]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            max_err = np.max(np.abs(y_true - y_pred))
            print("\nMetrics on rows with ground-truth Price:")
            print(f"  MAE       : {mae:,.0f} €")
            print(f"  RMSE      : {rmse:,.0f} €")
            print(f"  MaxAbsErr : {max_err:,.0f} €")
        else:
            print("No ground-truth prices found – skipped metric calculation.")

    output_path = Path(output)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    predict()
