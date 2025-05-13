import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.data_manip import load_data

df = load_data()

OUTDIR = Path(__file__).parent.parent / 'data' / 'insights'
OUTDIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 1. Price Distribution --- 
price = df['price'].dropna()
median_price = price.median()
percentile_25 = price.quantile(0.25)
percentile_75 = price.quantile(0.75)
percentile_95 = price.quantile(0.95)
million_plus_count = (price > 1000000).sum()
less_than_30k = (price < 30000).sum()

print("\n=== Price Distribution Analysis ===")
print(f"Median Price: ${median_price:,.2f}")
print(f"\nKey Percentiles:")
print(f"25th Percentile: ${percentile_25:,.2f}")
print(f"75th Percentile: ${percentile_75:,.2f}")
print(f"95th Percentile: ${percentile_95:,.2f}")
print(f"\nPrice Range Distribution:")
print(f"Number of properties over $1M: {million_plus_count:,}")
print(f"Number of properties under $30K: {less_than_30k:,}")
print("\n" + "="*30)