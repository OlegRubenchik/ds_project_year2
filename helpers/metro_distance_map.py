import pandas as pd
import folium
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


ROOT = Path(__file__).parent.parent
CLEAN_DATASET = ROOT / 'data' / 'processed' / 'clean_dataset.parquet'
METRO_STATIONS = ROOT / 'data' / 'raw' / 'athens_metro_coordinates.csv'

# Load the apartment data
df = pd.read_parquet(CLEAN_DATASET, engine='pyarrow')

# Load metro stations data and rename columns to match
metro_stations = pd.read_csv(METRO_STATIONS)
metro_stations.columns = ['Station', 'Latitude', 'Longitude']
print("\nMetro stations data:")
print(metro_stations.head(20))

def calculate_min_distance_to_metro(row, metro_stations):
    # Calculate distances to all stations
    apartment_coords = np.array([[row['latitude'], row['longitude']]])
    metro_coords = metro_stations[['Latitude', 'Longitude']].values
    
    # Calculate distances in kilometers (approximate using Euclidean distance)
    distances = cdist(apartment_coords, metro_coords) * 111  # Convert to km (rough approximation)
    return np.min(distances)

# Calculate minimum distance to metro for each apartment
df['metro_distance'] = df.apply(lambda row: calculate_min_distance_to_metro(row, metro_stations), axis=1)

# Create color bins for distances
distance_bins = [0, 0.5, 1, 2, float('inf')]  # distances in km
distance_labels = ['Very Close', 'Close', 'Moderate', 'Far']
df['distance_category'] = pd.cut(df['metro_distance'], bins=distance_bins, labels=distance_labels)

# Color mapping
color_map = {
    'Very Close': 'green',
    'Close': 'blue',
    'Moderate': 'orange',
    'Far': 'red'
}

# Create the map centered around the mean of all coordinates
map_center = [df['latitude'].mean(), df['longitude'].mean()]
distance_map = folium.Map(location=map_center, zoom_start=11)

# Sample data for better performance
sample_df = df.sample(n=3000, random_state=42)  # Adjust sample size as needed

# Add apartment markers
for _, row in sample_df.iterrows():
    distance = round(row['metro_distance'], 2)
    category = row['distance_category']
    color = color_map.get(category, 'gray')
    
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"Distance to nearest metro: {distance}km"
    ).add_to(distance_map)

# Add metro stations as larger markers
for _, station in metro_stations.iterrows():
    folium.CircleMarker(
        location=(station['Latitude'], station['Longitude']),
        radius=6,
        color='black',
        fill=True,
        fill_color='yellow',
        fill_opacity=1.0,
        popup=f"Metro Station: {station['Station']}"
    ).add_to(distance_map)

# Add a legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 150px; height: 130px; 
            border:2px solid grey; z-index:9999; background-color:white;
            opacity:0.8; font-size:12px; padding:10px;">
    <p><b>Distance to Metro</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> &lt; 0.5 km</p>
    <p><i class="fa fa-circle" style="color:blue"></i> 0.5-1 km</p>
    <p><i class="fa fa-circle" style="color:orange"></i> 1-2 km</p>
    <p><i class="fa fa-circle" style="color:red"></i> &gt; 2 km</p>
</div>
'''
distance_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map
distance_map.save("athens_metro_distance_map.html")
print("\n✅ Map saved as athens_metro_distance_map.html") 