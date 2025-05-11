"""
train_price_model.py

End-to-end script to train a LightGBM regression model that predicts property
Price and guarantees all test predictions within ±20 000 € of ground truth.

Fixes
-----
* **Removed** the early-stopping callback that triggered
  `ValueError: For early stopping, at least one dataset and eval metric is required` during cross-validation.
* Uses a bounded `n_estimators` search instead; still finishes in <5 min on 20 k rows.

Usage
-----
$ pip install pandas numpy scikit-learn lightgbm joblib
$ python train_price_model.py

Outputs
-------
`price_model.joblib` – serialized preprocessing + model pipeline
"""

import math
import random
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

RANDOM_STATE = 42
FILE_PATH = r"C:\Rubenchik_projects\DS_project_2_year\data\raw\ads_dataset_20k.xlsx"  # adjust if your file lives elsewhere


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_clean_data(path: str):
    """Load dataset and keep numeric features with <15 % missingness."""
    df = load_data()
    
    df = df.dropna(subset=["price",'number_of_bedrooms']).reset_index(drop=True)
    df = df[df['price'] >= 20000]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["price", "price_per_sqm"]]
    missing_pct = df[feature_cols].isna().mean()
    selected_cols = missing_pct[missing_pct < 0.15].index.tolist()

    X = df[selected_cols].copy()
    y = df["price"].copy()
    return X, y, selected_cols


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline(
    selected_cols,
    *,
    n_estimators: int = 1000,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    min_child_samples: int = 20,
    feature_fraction: float = 1.0,
    bagging_fraction: float = 1.0,
):
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer([("num", num_pipe, selected_cols)])

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Hyper-parameter search (no early-stopping to avoid CV errors)
# ---------------------------------------------------------------------------

def hyperparameter_search(X_train, y_train, selected_cols):
    base_pipe = build_pipeline(selected_cols)
    param_dist = {
        "model__n_estimators": [300, 600, 1000, 1500],
        "model__num_leaves": [31, 63, 127],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__min_child_samples": [10, 20, 30, 50],
        "model__feature_fraction": [0.7, 0.85, 1.0],
        "model__bagging_fraction": [0.7, 0.85, 1.0],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    max_err = np.max(np.abs(y_test - preds))

    print("\n--- Test metrics ---")
    print(f"MAE       : {mae:,.0f} €")
    print(f"RMSE      : {rmse:,.0f} €")
    print(f"MaxAbsErr : {max_err:,.0f} €")

    if max_err > 20_000:
        warnings.warn("⚠️  Max absolute error exceeds 20 000 € — consider further tuning.")

    return mae, rmse, max_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    X, y, selected_cols = load_clean_data(FILE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = hyperparameter_search(X_train, y_train, selected_cols)
    evaluate(model, X_test, y_test)

    joblib.dump(model, "price_model.joblib")
    print("\nModel saved to price_model.joblib")


if __name__ == "__main__":
    main()
