import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.data_manip import load_data

# Create output directory for plots
OUTDIR = Path(__file__).parent.parent / 'data' / 'insights'
OUTDIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load and prepare data
df = load_data()

# Print data info and check for nulls
print("\n=== Data Info ===")
print(df.info())
print("\n=== Null Values ===")
print(df.isnull().sum())

# Print basic statistics about numerical columns
print("\n=== Numerical Columns Statistics ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"\n{col} statistics:")
    print(df[col].describe())

# Select numerical columns for correlation analysis, excluding price_per_sqm
numeric_cols = numeric_cols[~numeric_cols.isin(['price_per_sqm'])]
correlation_matrix = df[numeric_cols].corr()

# Sort correlations with price
price_correlations = correlation_matrix['price'].sort_values(ascending=False)

# Print correlations with price
print("\n=== Price Correlations Analysis ===")
print("\nTop correlations with price:")
print(price_correlations.head(10))
print("\nBottom correlations with price:")
print(price_correlations.tail(10))

# Create correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f')
plt.title('Feature Correlation Heatmap (excluding price_per_sqm)')
plt.tight_layout()
plt.savefig(OUTDIR / f'correlation_heatmap_{timestamp}.png')
plt.close()

# Create scatter plots for top correlated features
top_features = price_correlations[1:4].index  # Excluding price itself and taking top 3
top_features = pd.Index(list(top_features) + ['construction_year'])  # Add construction year
fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # Changed to 2x2 layout
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    # Print feature info
    print(f"\nFeature: {feature}")
    print(f"Data type: {df[feature].dtype}")
    print(f"Unique values: {df[feature].nunique()}")
    print(f"Value counts:\n{df[feature].value_counts().head()}")
    
    # Convert to numeric if needed
    if df[feature].dtype == 'object':
        try:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            print(f"Converted {feature} to numeric")
        except:
            print(f"Could not convert {feature} to numeric")
            continue
    
    # Remove outliers for both feature and price (using 1st and 99th percentiles)
    x_low, x_high = np.percentile(df[feature].dropna(), [1, 99])
    y_low, y_high = np.percentile(df['price'].dropna(), [1, 99])
    
    # Special handling for construction year
    if feature == 'construction_year':
        # Filter out invalid years (e.g., 0 or future years)
        current_year = datetime.now().year
        mask = (df[feature] > 1800) & (df[feature] <= current_year) & \
               (df['price'] >= y_low) & (df['price'] <= y_high) & \
               df[feature].notna()
    else:
        # Filter data for plotting
        mask = (df[feature] >= x_low) & (df[feature] <= x_high) & \
               (df['price'] >= y_low) & (df['price'] <= y_high) & \
               df[feature].notna()
    
    plot_data = df[mask]
    
    if len(plot_data) == 0:
        print(f"No data to plot for {feature} after filtering")
        continue
    
    # Create scatter plot
    sns.scatterplot(data=plot_data, x=feature, y='price', ax=axes[idx], alpha=0.5)
    axes[idx].set_title(f'Price vs {feature}')
    axes[idx].set_ylabel('Price (€)')
    axes[idx].set_xlabel(feature)
    
    # Format y-axis to show prices in thousands/millions
    axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}€'))
    
    # Add grid for better readability
    axes[idx].grid(True, alpha=0.3)
    
    # Special handling for discrete features
    if feature in ['number_of_bedrooms', 'floor_number']:
        # Get unique values for x-axis
        x_unique = sorted(plot_data[feature].unique())
        axes[idx].set_xticks(x_unique)
    elif feature == 'construction_year':
        # Set reasonable x-axis ticks for years
        start_year = int(plot_data[feature].min() // 10 * 10)
        end_year = int(plot_data[feature].max() // 10 * 10 + 10)
        axes[idx].set_xticks(range(start_year, end_year, 10))
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Print the range of values being shown
    print(f"Showing data between:")
    if feature == 'construction_year':
        print(f"X-axis: {plot_data[feature].min():.0f} to {plot_data[feature].max():.0f}")
    else:
        print(f"X-axis: {x_low:,.2f} to {x_high:,.2f}")
    print(f"Y-axis: {y_low:,.2f}€ to {y_high:,.2f}€")
    print(f"Number of points plotted: {len(plot_data)} out of {len(df)}")

plt.suptitle('Price Relationships with Key Features\n(Excluding Outliers - 1st to 99th percentile)', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(OUTDIR / f'price_correlations_scatter_{timestamp}.png')
plt.close()

# Additional analysis: Box plots for categorical variables if any
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

if len(categorical_cols) > 0:
    print("\nAnalyzing categorical variables:")
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Only for categories with fewer than 10 unique values
            print(f"\nAverage price by {col}:")
            avg_prices = df.groupby(col)['price'].mean().sort_values(ascending=False)
            print(avg_prices.apply(lambda x: f'{x:,.2f}€')) 