import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.data_manip import load_data
from helpers.paths import Dirs,Files

def generate_price_distribution_summary():
    df = load_data()

    Dirs.DATA_STAT_DIR.mkdir(parents=True, exist_ok=True)

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

    # --- 2. Save distribution table ---
    summary_table = pd.DataFrame({
        'Metric': [
            'Median Price',
            '25th Percentile',
            '75th Percentile',
            '95th Percentile',
            'Count > $1M',
            'Count < $30K'
        ],
        'Value': [
            round(median_price, 2),
            round(percentile_25, 2),
            round(percentile_75, 2),
            round(percentile_95, 2),
            million_plus_count,
            less_than_30k
        ]
    })

    import matplotlib.pyplot as plt

    # Create a table as a figure
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')  # Remove axes

    # Render the table
    table = ax.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust size as needed

    # Save as PNG
    png_path = Files.PRICE_DISTR_TABLE
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()

    print(f"📸 Table image saved to {png_path}")

if __name__ == '__main__':
    generate_price_distribution_summary()
