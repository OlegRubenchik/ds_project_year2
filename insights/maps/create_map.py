#TODO
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#folium map  with lats and longs from the dataset

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium

from helpers.paths import Dirs, Files

def generate_map():
    Dirs.DATA_MAPS_DIR.mkdir(exist_ok=True)

    file_path = Files.CLEAN_DATASET
    df = pd.read_parquet(file_path,engine='pyarrow')

    # Define price categories manually
    price_bins = [0, 75000, 150000, 300000,800000, float('inf')]
    price_labels = ['Low', 'Medium', 'High', 'Luxury','SuperLuxury']
    df['price_category'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)

    # Drop rows with missing coordinates or price
    df = df.dropna(subset=['latitude', 'longitude', 'price'])

    # Color mapping
    color_map = {
        'Low': 'green',
        'Medium': 'blue',
        'High': 'orange',
        'Luxury': 'red',
        'SuperLuxury': 'black'
    }

    # Create the map centered around the mean of all long and lat
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    price_map = folium.Map(location=map_center, zoom_start=11)


    sample_df = df  

    # Add dots to the map
    for _, row in sample_df.iterrows():
        price = int(row['price'])
        category = row['price_category']
        color = color_map.get(category, 'gray')
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{category} - {price}€"
        ).add_to(price_map)

    # Save the map
    price_map.save(Files.EXACT_PRICE_MAP)
    print("✅ Map saved as athens_exact_price_map.html")

if __name__ == '__main__':
    generate_map()


