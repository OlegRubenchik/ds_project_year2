import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

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

def clean_data(df):
    df = df.dropna(subset=['price'])
    df = df[df['price'] >= 20000]
    return df

def prepare_features_and_target(df):
    X = df[PRICE_REG_CONFIG]
    y = df['price']
    return X, y

def find_optimal_n_estimators(X_train, y_train):
    print("🔍 Searching for optimal n_estimators...")
    param_grid = {
        'n_estimators': [100, 300, 500, 700, 1000, 2000],
        'max_depth': [10,15],
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=52),
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
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=52, verbose=1, max_depth=15)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    results = X_test_reset.copy()
    results['Actual'] = y_test_reset
    results['Predicted'] = y_pred
    results['absolute_error'] = (results['Actual'] - results['Predicted']).abs()
    results['within_50k'] = results['absolute_error'] <= 50000

    mse = mean_squared_error(y_test, y_pred)
    count = results['within_50k'].sum()
    total = len(results)
    print(f"Within 50k: {count} out of {total} ({count / total:.2%})")
    print(f"Mean Squared Error: {mse:.2f}")

    return results

def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"random_forest_results_{timestamp}.csv"
    output_path = PROC_DIR / filename
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    df = load_data()
    df = clean_data(df)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_model(X_train, y_train, n_estimators=1000)
    results = evaluate_model(model, X_test, y_test)
    save_results(results)

if __name__ == '__main__':
    # df = load_data()
    # df_cl = clean_data(df)
    # X, y = prepare_features_and_target(df_cl)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    main()
    
