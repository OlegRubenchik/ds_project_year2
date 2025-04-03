#TODO

#folium map  with lats and longs from the dataset

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from pathlib import Path

ROOT = Path(__file__).parent.parent
CLEAN_DATASET = ROOT / 'data' / 'processed' / 'clean_dataset.parquet'


file_path = CLEAN_DATASET 
df = pd.read_parquet(file_path,engine='pyarrow')

# Define price categories manually
price_bins = [0, 75000, 150000, 300000, float('inf')]
price_labels = ['Low', 'Medium', 'High', 'Luxury']
df['price_category'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)

# Drop rows with missing coordinates or price
df = df.dropna(subset=['latitude', 'longitude', 'price'])

# Color mapping
color_map = {
    'Low': 'green',
    'Medium': 'blue',
    'High': 'orange',
    'Luxury': 'red'
}

# Create the map centered around the mean of all long and lat
map_center = [df['latitude'].mean(), df['longitude'].mean()]
price_map = folium.Map(location=map_center, zoom_start=11)

# Sample data (optional for performance)
sample_df = df.sample(n=3000, random_state=42)  # Or remove `.sample` to use full dataset

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
price_map.save("athens_exact_price_map.html")
print("✅ Map saved as athens_exact_price_map.html")


