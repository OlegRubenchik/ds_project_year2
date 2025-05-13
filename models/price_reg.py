import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
import math

def add_to_path():
    """
    Adds the project root directory to Python path to enable imports from any project module.
    Should be called at the start of scripts that need to import from other project modules.
    """
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

add_to_path()

from helpers.config import PRICE_REG_CONFIG
from helpers.data_manip import load_data

ROOT = Path(__file__).parent.parent
PROC_DIR = ROOT / 'data' / 'processed'

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

def clean_data(df):
    df = df.dropna(subset=['price'])
    df = df[df['price'] >= 20000]
    return df

def prepare_features_and_target(df):
    X = df[PRICE_REG_CONFIG]
    y = df['price']  # Now using original price, not log_price
    return X, y

def find_optimal_n_estimators(X_train, y_train):
    print("🔍 Searching for optimal n_estimators...")
    param_grid = {
        'n_estimators': [100, 300, 500, 700, 1000, 2000],
        'max_depth': [10,15],
    }
    grid_search = GridSearchCV(
        LogTransformedRegressor(RandomForestRegressor(random_state=52)),  # Wrap the base model
        param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_n = grid_search.best_params_['n_estimators']
    best_score = -grid_search.best_score_
    print(f"✅ Best n_estimators: {best_n} with MSE: {best_score:.2f}")
    return best_n

def train_model(X_train, y_train, n_estimators):
    base_model = RandomForestRegressor(n_estimators=n_estimators, random_state=52, verbose=1, max_depth=15)
    model = LogTransformedRegressor(base_model)  # Wrap the base model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, df):
    # Get predictions (already in normal scale thanks to LogTransformedRegressor)
    y_pred = model.predict(X_test)
    
    X_test_reset = X_test.reset_index(drop=True)
    results = X_test_reset.copy()
    
    # Add predictions
    results['price_actual'] = y_test.reset_index(drop=True)
    results['price_predicted'] = y_pred
    
    # Calculate errors
    results['price_absolute_error'] = (results['price_actual'] - results['price_predicted']).abs()
    results['price_relative_error'] = (results['price_absolute_error'] / results['price_actual'] * 100)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error (€): {mae:,.2f}")
    print(f"RMSE (€): {rmse:,.2f}")
    
    print("\nPrediction Accuracy:")
    # Print accuracy for every 10% increment
    for threshold in range(10, 110, 10):
        within_percent = (results['price_relative_error'] <= threshold).mean() * 100
        print(f"Within {threshold:3d}% of actual price: {within_percent:5.1f}%")
    
    # Also show the €50k metric
    within_50k = (results['price_absolute_error'] <= 50000).sum()
    total = len(results)
    print(f"\nWithin €50k of actual price: {within_50k} out of {total} ({within_50k/total:.1%})")
    
    # Add some example predictions
    print("\nExample Predictions:")
    sample_results = results.sample(5, random_state=42)
    for _, row in sample_results.iterrows():
        print(f"\nActual Price: €{row['price_actual']:,.2f}")
        print(f"Predicted Price: €{row['price_predicted']:,.2f}")
        print(f"Absolute Error: €{row['price_absolute_error']:,.2f}")
        print(f"Relative Error: {row['price_relative_error']:.1f}%")

    return results

def save_model(model, results, timestamp=None):
    """Save both model and its metadata"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create models directory if it doesn't exist
    models_dir = ROOT / 'models' / 'saved'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model with metadata
    model_info = {
        'model': model,
        'features': PRICE_REG_CONFIG,
        'timestamp': timestamp,
        'metrics': {
            'mse_log': mean_squared_error(results['price_actual'], results['price_predicted']),
            'mae_normal': mean_absolute_error(results['price_actual'], results['price_predicted'])
        }
    }
    
    model_path = models_dir / f"random_forest_model_{timestamp}.joblib"
    joblib.dump(model_info, model_path)
    print(f"\nModel saved to {model_path}")

def save_results(model, results):
    """Save both model and prediction results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    filename = f"random_forest_results_{timestamp}.csv"
    output_path = PROC_DIR / filename
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Save model
    save_model(model, results, timestamp)

def main():
    df = load_data()
    df = clean_data(df)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_model(X_train, y_train, n_estimators=1000)
    results = evaluate_model(model, X_test, y_test, df)
    save_results(model, results)

if __name__ == '__main__':
    main()
    
