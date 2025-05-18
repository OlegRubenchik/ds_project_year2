import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import joblib
import math
import matplotlib.pyplot as plt
import seaborn as sns

def add_to_path():
    """
    Adds the project root directory to Python path to enable imports from any project module.
    Should be called at the start of scripts that need to import from other project modules.
    """
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

add_to_path()

from helpers.config import PRICE_REG_CONFIG, MODEL_CONFIG
from helpers.data_manip import load_data
from helpers.paths import Dirs, Files
from models.transformers import TransformedRegressor

ROOT = Path(__file__).parent.parent
PROC_DIR = ROOT / 'data' / 'processed'

def clean_data(df):
    df = df.dropna(subset=['price'])
    df = df[df['price'] >= 20000]
    return df

def prepare_features_and_target(df):
    X = df[PRICE_REG_CONFIG]
    y = df['price']
    return X, y

def find_optimal_n_estimators(X_train, y_train):
    print("🔍 Starting grid search with parameters:")
    print(f"Base model params: {MODEL_CONFIG['base_model']['params']}")
    print(f"Grid search params: {MODEL_CONFIG['grid_search']['params']}")
    if MODEL_CONFIG['transformer']['enabled']:
        print(f"Using transformer with: {MODEL_CONFIG['transformer']['params']}")
    print("\nThis may take a while...\n")
    
    # Create base model with config params
    base_model = RandomForestRegressor(**MODEL_CONFIG['base_model']['params'])
    
    # Wrap with transformer if enabled
    if MODEL_CONFIG['transformer']['enabled']:
        model = TransformedRegressor(
            base_model,
            **MODEL_CONFIG['transformer']['params']
        )
        # Modify param grid to target the base regressor
        param_grid = {
            'regressor__' + key: value 
            for key, value in MODEL_CONFIG['grid_search']['params'].items()
        }
    else:
        model = base_model
        param_grid = MODEL_CONFIG['grid_search']['params']
    
    # Setup grid search
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=MODEL_CONFIG['grid_search']['cv'],
        verbose=2,  # More detailed output
        n_jobs=MODEL_CONFIG['grid_search']['n_jobs']
    )
    
    print("\nFitting models...")
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    
    # Remove 'regressor__' prefix from best params if using transformer
    if MODEL_CONFIG['transformer']['enabled']:
        best_params = {
            key.replace('regressor__', ''): value 
            for key, value in best_params.items()
        }
    
    print("\n✨ Grid Search Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best MSE: {best_score:.2f}")
    print(f"Best RMSE: {math.sqrt(best_score):.2f}")
    
    # Print all results sorted by score
    print("\nAll tested combinations (sorted by score):")
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results['rmse'] = np.sqrt(-cv_results['mean_test_score'])
    cv_results = cv_results.sort_values('mean_test_score', ascending=False)
    
    for idx, row in cv_results.iterrows():
        params = {k.replace('param_regressor__', ''): v 
                 for k, v in row.items() 
                 if k.startswith('param_') and v is not None}
        print(f"Parameters: {params}")
        print(f"RMSE: {row['rmse']:.2f}")
        print("---")
    
    return best_params

def train_model(X_train, y_train, model_params):
    # Create base model with config and best params
    base_params = MODEL_CONFIG['base_model']['params'].copy()
    base_params.update(model_params)
    base_model = RandomForestRegressor(**base_params)
    
    # Wrap with transformer if enabled
    if MODEL_CONFIG['transformer']['enabled']:
        model = TransformedRegressor(
            base_model,
            **MODEL_CONFIG['transformer']['params']
        )
    else:
        model = base_model
    
    model.fit(X_train, y_train)
    return model

def create_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, save_path=None):
    """Create and save a comprehensive regression metrics table."""
    # Calculate basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)
    
    # Calculate percentage errors
    abs_perc_error = np.abs((y_true - y_pred) / y_true * 100)
    mape = np.mean(abs_perc_error)
    median_ape = np.median(abs_perc_error)
    
    # Calculate thresholds
    within_10_percent = np.mean(abs_perc_error <= 10) * 100
    within_20_percent = np.mean(abs_perc_error <= 20) * 100
    within_30_percent = np.mean(abs_perc_error <= 30) * 100
    within_40_percent = np.mean(abs_perc_error <= 40) * 100
    within_50_percent = np.mean(abs_perc_error <= 50) * 100

        
    # Calculate absolute thresholds
    abs_error = np.abs(y_true - y_pred)
    within_25k = np.mean(abs_error <= 25000) * 100
    within_50k = np.mean(abs_error <= 50000) * 100
    within_100k = np.mean(abs_error <= 100000) * 100
    
    # Create metrics dictionary
    metrics = {
        'Metric': [
            'R² Score',
            'Explained Variance',
            'Mean Squared Error',
            'Root Mean Squared Error',
            'Mean Absolute Error',
            'Mean Absolute Percentage Error',
            'Median Absolute Percentage Error',
            'Within 10% of True Price',
            'Within 20% of True Price',
            'Within 30% of True Price',
            'Within 40% of True Price',
            'Within 50% of True Price',
            'Within €25k of True Price',
            'Within €50k of True Price',
            'Within €100k of True Price'
        ],
        'Value': [
            f'{r2:.4f}',
            f'{explained_var:.4f}',
            f'€{mse:,.2f}',
            f'€{rmse:,.2f}',
            f'€{mae:,.2f}',
            f'{mape:.2f}%',
            f'{median_ape:.2f}%',
            f'{within_10_percent:.1f}%',
            f'{within_20_percent:.1f}%',
            f'{within_30_percent:.1f}%',
            f'{within_40_percent:.1f}%',
            f'{within_50_percent:.1f}%',
            f'{within_25k:.1f}%',
            f'{within_50k:.1f}%',
            f'{within_100k:.1f}%'
        ]
    }
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Remove axes
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create table
    table = plt.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.6, 0.4]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Color header
    for j, cell in enumerate(table._cells[(0, j)] for j in range(len(metrics_df.columns))):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', weight='bold')
    
    # Add title
    plt.title('Regression Model Performance Metrics', pad=20, size=14, weight='bold')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\nMetrics table saved to: {save_path}")
    else:
        plt.show()
        plt.close()

def evaluate_model(model, X_test, y_test, df):
    """Evaluate the model and generate metrics."""
    # Get predictions
    y_pred = model.predict(X_test)
    
    X_test_reset = X_test.reset_index(drop=True)
    results = X_test_reset.copy()
    
    # Add predictions
    results['price_actual'] = y_test.reset_index(drop=True)
    results['price_predicted'] = y_pred
    
    # Calculate errors
    results['price_absolute_error'] = (results['price_actual'] - results['price_predicted']).abs()
    results['price_relative_error'] = (results['price_absolute_error'] / results['price_actual'] * 100)
    
    # Generate metrics table
    metrics_path = Files.PRICE_MODEL_REGRESSION_METRICS
    create_metrics_table(results['price_actual'], results['price_predicted'], metrics_path)
    
    # Print example predictions
    print("\nExample Predictions:")
    sample_results = results.sample(5, random_state=42)
    for _, row in sample_results.iterrows():
        print(f"\nActual Price: €{row['price_actual']:,.2f}")
        print(f"Predicted Price: €{row['price_predicted']:,.2f}")
        print(f"Absolute Error: €{row['price_absolute_error']:,.2f}")
        print(f"Relative Error: {row['price_relative_error']:.1f}%")

    return results

def save_model(model, results, timestamp=None, best_params=None):
    """Save both model and its metadata"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create models directory if it doesn't exist
    models_dir = Dirs.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Get actual model parameters
    if hasattr(model, 'regressor'):
        actual_params = model.regressor.get_params()
        transformer_params = {
            'transform_fn': model.transform_fn.__name__,
            'inverse_fn': model.inverse_fn.__name__
        }
    else:
        actual_params = model.get_params()
        transformer_params = None
    
    # Save model with metadata
    model_info = {
        'model': model,
        'features': PRICE_REG_CONFIG,
        'timestamp': timestamp,
        'config': MODEL_CONFIG,
        'best_params': best_params,  # Grid search best parameters
        'actual_params': actual_params,  # Actual model parameters
        'transformer_params': transformer_params,  # Transformer configuration if used
        'metrics': {
            'mse': mean_squared_error(results['price_actual'], results['price_predicted']),
            'mae': mean_absolute_error(results['price_actual'], results['price_predicted'])
        }
    }
    
    model_path = Files.PRICE_MODEL
    joblib.dump(model_info, model_path)
    print(f"\nModel saved to {model_path}")

def save_results(model, results, best_params):
    """Save both model and prediction results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    filename = f"random_forest_results_{timestamp}.csv"
    output_path = PROC_DIR / filename
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Save model
    save_model(model, results, timestamp, best_params)

def main(skip_grid_search=False, best_params=None):
    df = load_data()
    df = clean_data(df)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    if not skip_grid_search:
        best_params = find_optimal_n_estimators(X_train, y_train)
    elif best_params is None:
        # Use default parameters from config if no grid search and no params provided
        best_params = MODEL_CONFIG['grid_search']['params']
        print("Using default parameters:", best_params)
    else:
        print("Using provided parameters:", best_params)
    
    model = train_model(X_train, y_train, best_params)
    results = evaluate_model(model, X_test, y_test, df)
    save_results(model, results, best_params)

if __name__ == '__main__':
    # By default, run with grid search
    # To skip grid search: main(skip_grid_search=True, best_params=your_params)

    best_params = {
        'max_depth': 15,
        'n_estimators': 700
    }
    main(skip_grid_search=True, best_params=best_params)
    
