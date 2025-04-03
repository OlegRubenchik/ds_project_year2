import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist



ROOT = Path(__file__).parent.parent
CLEAN_DATASET = ROOT / 'data' / 'processed' / 'clean_dataset.parquet'
METRO_STATIONS = ROOT / 'data' / 'raw' / 'athens_metro_coordinates.csv'

df = pd.read_parquet(CLEAN_DATASET, engine='pyarrow')


def calculate_min_distance_to_metro(row, metro_stations):
    # Load metro stations if a CSV file path is provided
    if isinstance(metro_stations, (str, Path)):
        metro_stations = pd.read_csv(metro_stations)
        
    # Calculate distances to all stations
    apartment_coords = np.array([[row['latitude'], row['longitude']]])
    metro_coords = metro_stations[['latitude', 'longitude']].values
    
    # Calculate distances in kilometers (approximate using Euclidean distance)
    distances = cdist(apartment_coords, metro_coords) * 111  # Convert to km (rough approximation)
    return np.min(distances)

df['metro_distance'] = df.apply(lambda row: calculate_min_distance_to_metro(row, METRO_STATIONS), axis=1)

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

print(df.head())
# Calculate average distance to metro for each price class
avg_distance_by_class = df.groupby('price_class')['metro_distance'].agg(['mean', 'count']).round(2)
avg_distance_by_class = avg_distance_by_class.sort_values('mean')

print("\nAverage distance to metro by price class:")
print(avg_distance_by_class)
# Define the desired order of price classes
price_class_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Reindex the DataFrame with the desired order
avg_distance_by_class = avg_distance_by_class.reindex(price_class_order)
