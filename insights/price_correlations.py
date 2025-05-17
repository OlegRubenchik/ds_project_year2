import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.data_manip import load_data
from helpers.paths import Dirs, Files

def generate_price_correlation_analysis():
    df = load_data()
    Dirs.DATA_STAT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Starting Correlation Analysis ===")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols[~numeric_cols.isin(['price_per_sqm', 'log_price'])]

    correlation_matrix = df[numeric_cols].corr()
    price_correlations = correlation_matrix['price'].sort_values(ascending=False)

    print("\nTop correlations with price:")
    print(price_correlations.head(10).round(2))

    print("\nBottom correlations with price:")
    print(price_correlations.tail(10).round(2))

    # --- Save heatmap ---
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f')
    plt.title('Feature Correlation Heatmap (excluding price_per_sqm)')
    plt.tight_layout()
    plt.savefig(Files.PRICE_CORR_HEATMAP)
    plt.close()
    print(f"📊 Heatmap saved to {Files.PRICE_CORR_HEATMAP}")

    # --- Save correlation table as image ---
    summary_table = price_correlations.round(2).reset_index()
    summary_table.columns = ['Feature', 'Correlation with Price']

    fig, ax = plt.subplots(figsize=(6, len(summary_table) * 0.4))
    ax.axis('off')
    table = ax.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    plt.savefig(Files.PRICE_CORR_TABLE, bbox_inches='tight')
    plt.close()
    print(f"📸 Correlation table saved to {Files.PRICE_CORR_TABLE}")

    # --- Create scatter plots for top features ---
    top_features = price_correlations[1:4].index  # Exclude 'price' itself
    top_features = pd.Index(list(top_features) + ['construction_year'])  # Include construction year

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, feature in enumerate(top_features):
        if feature not in df.columns:
            continue

        # Remove outliers
        x_low, x_high = np.percentile(df[feature].dropna(), [1, 99])
        y_low, y_high = np.percentile(df['price'].dropna(), [1, 99])

        if feature == 'construction_year':
            current_year = datetime.now().year
            mask = (df[feature] > 1800) & (df[feature] <= current_year) & \
                   (df['price'] >= y_low) & (df['price'] <= y_high)
        else:
            mask = (df[feature] >= x_low) & (df[feature] <= x_high) & \
                   (df['price'] >= y_low) & (df['price'] <= y_high)

        plot_data = df[mask]

        if plot_data.empty:
            continue

        sns.scatterplot(data=plot_data, x=feature, y='price', ax=axes[idx], alpha=0.5)
        axes[idx].set_title(f'Price vs {feature}')
        axes[idx].set_ylabel('Price (€)')
        axes[idx].set_xlabel(feature)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}€'))

        # Custom ticks for discrete features
        if feature == 'construction_year':
            axes[idx].set_xticks(range(int(plot_data[feature].min()) // 10 * 10,
                                       int(plot_data[feature].max()) // 10 * 10 + 10, 10))
            axes[idx].tick_params(axis='x', rotation=45)

    plt.suptitle('Price vs Key Features (1st–99th percentile)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(Files.PRICE_SCATTER_PLOTS)
    plt.close()
    print(f"📈 Scatter plots saved to {Files.PRICE_SCATTER_PLOTS}")

if __name__ == "__main__":
    generate_price_correlation_analysis()
