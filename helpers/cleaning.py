import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Literal
from rich import print

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from insights.location_grid import create_grid_and_map

# ALL PATHS
ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / 'data' / 'processed'
CLEAN_DATASET = PROCESSED_DIR / 'clean_dataset.parquet'
RAW_DATASET_13k = ROOT / 'data' / 'raw' / 'ads_dataset_13k.xlsx'
RAW_DATASET_20k = ROOT / 'data' / 'raw' / 'ads_dataset_20k.xlsx'
DATASETS = {
    '13k': RAW_DATASET_13k,
    '20k': RAW_DATASET_20k
}

def clean_and_save(dataset: Literal['20k', '13k'], force: bool = False) -> pd.DataFrame:
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if CLEAN_DATASET.exists() and not force:
        print(f"Using existing cleaned dataset from: {CLEAN_DATASET}")
        return None
    print(f'[green]Using {dataset} dataset[/green]')
    df = pd.read_excel(DATASETS[dataset])

    
    # Cleaning logic
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Handle invalid prices
    invalid_prices = df['price'] <= 0
    if invalid_prices.any():
        print(f"\nWarning: Found {invalid_prices.sum()} properties with invalid prices (<=0)")
        print("These will be excluded from log transformation")
    
    # Add log-transformed price
    df['log_price'] = np.nan  # Initialize with NaN
    df.loc[~invalid_prices, 'log_price'] = np.log(df.loc[~invalid_prices, 'price'])
    
    # Print statistics for valid prices only
    valid_df = df[~invalid_prices]
    print("\nPrice distribution statistics (excluding invalid prices):")
    print("\nOriginal price:")
    print(valid_df['price'].describe())
    print("\nLog-transformed price:")
    print(valid_df['log_price'].describe())
    
    # Calculate and print skewness for valid prices
    original_skew = valid_df['price'].skew()
    log_skew = valid_df['log_price'].skew()
    print(f"\nSkewness (excluding invalid prices):")
    print(f"Original price: {original_skew:.2f}")
    print(f"Log price: {log_skew:.2f}")
    
    # Create grid system and add loccell column
    df = create_grid_and_map(df)

    df.index.name = 'n_sample'
    df.columns.name = 'feature_label'

    # Save as parquet
    print(f"\nSaving cleaned dataset to: {CLEAN_DATASET}")
    df.to_parquet(CLEAN_DATASET, engine='pyarrow', index=True)
    print(f"Saving cleaned dataset to Excel: {CLEAN_DATASET.with_suffix('.xlsx')}")
    df.to_excel(CLEAN_DATASET.with_suffix('.xlsx'), index=True)
    return df

if __name__ == "__main__":
    clean_and_save('13k',force=True)

