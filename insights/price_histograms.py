import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.paths import Files, Dirs

def calculate_skewness(data):
    """Calculate skewness of the data."""
    return skew(data)

def create_histogram(data, title, save_path, skewness):
    """Create and save a histogram with skewness information."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, bins=50, kde=True)
    plt.title(f'{title}\nSkewness: {skewness:.2f}')
    plt.xlabel('Price' if 'log' not in title.lower() else 'Log Price')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_price_histograms():
    """Generate histograms for price and log price with skewness information."""
    # Read the dataset
    df = pd.read_parquet(Files.CLEAN_DATASET)
    
    # Calculate price skewness
    price_skewness = calculate_skewness(df['price'])
    
    # Calculate log price and its skewness
    log_price = np.log1p(df['price'])
    log_price_skewness = calculate_skewness(log_price)
    
    # Create histograms
    create_histogram(
        data=df['price'],
        title='Price Distribution',
        save_path=Files.PRICE_HISTOGRAM,
        skewness=price_skewness
    )
    
    create_histogram(
        data=log_price,
        title='Log Price Distribution',
        save_path=Files.LOG_PRICE_HISTOGRAM,
        skewness=log_price_skewness
    )
    
    print(f"✅ Generated price histograms")
    print(f"Price skewness: {price_skewness:.2f}")
    print(f"Log price skewness: {log_price_skewness:.2f}")

if __name__ == '__main__':
    # Ensure the statistics directory exists
    Dirs.DATA_STAT_DIR.mkdir(parents=True, exist_ok=True)
    generate_price_histograms() 