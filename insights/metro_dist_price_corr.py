import numpy as np
import pandas as pd
from pathlib import Path
from haversine import haversine
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.paths import Dirs, Files
def generate_price_metro_dist_corr_pic():
    # File paths
    Dirs.DATA_STAT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(Files.CLEAN_DATASET, engine='pyarrow')
    metro_stations = pd.read_csv(Files.METRO_STATIONS)

    # Calculate Haversine distance to closest metro station
    def calculate_min_haversine_distance(row, metro_stations):
        apartment = (row['latitude'], row['longitude'])
        distances = metro_stations.apply(
            lambda station: haversine(apartment, (station['latitude'], station['longitude'])), axis=1)
        return distances.min()

    df['metro_distance'] = df.apply(lambda row: calculate_min_haversine_distance(row, metro_stations), axis=1)

    # Assign price classes
    def assign_price_class(row):
        if row['price'] < 100000:
            return 'Very Low'
        elif row['price'] < 200000:
            return 'Low'
        elif row['price'] < 300000:
            return 'Medium'
        elif row['price'] < 400000:
            return 'High'
        else:
            return 'Very High'

    df['price_class'] = df.apply(assign_price_class, axis=1)

    # Print first few rows
    print(df.head())

    # Average distance to metro by price class
    avg_distance_by_class = df.groupby('price_class')['metro_distance'].agg(['mean', 'count']).round(2)
    price_class_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    avg_distance_by_class = avg_distance_by_class.reindex(price_class_order)

    print("\nAverage distance to metro by price class:")
    print(avg_distance_by_class)

    # Correlation
    correlation = df['price'].corr(df['metro_distance'])
    print(f"\nCorrelation between price and metro distance: {correlation:.2f}")

    # Regression
    X = df[['metro_distance']]
    X = sm.add_constant(X)
    y = df['price']

    model = sm.OLS(y, X).fit()
    print("\nRegression summary:")
    print(model.summary())

    # Optional: Visualization
    sns.boxplot(x='price_class', y='metro_distance', data=df, order=price_class_order)
    plt.title("Metro Distance by Price Class")
    plt.xlabel("Price Class")
    plt.ylabel("Distance to Metro (km)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Files.PRICE_METRO_DIST_CORR, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_price_metro_dist_corr_pic()