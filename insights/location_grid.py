import pandas as pd
import numpy as np
import folium
from folium import plugins
import branca.colormap as cm
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Create output directory for the map
OUTDIR = Path(__file__).parent.parent / 'data' / 'insights'
OUTDIR.mkdir(parents=True, exist_ok=True)

def create_grid_and_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create grid system, generate visualization, and add grid cell information to the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with latitude, longitude, and price columns
        
    Returns:
        pd.DataFrame: Original dataframe with added loccell column
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Filter out rows with invalid coordinates
    valid_coords = df[df['latitude'].notna() & df['longitude'].notna() & 
                     df['latitude'].abs().ne(float('inf')) & 
                     df['longitude'].abs().ne(float('inf'))].copy()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create grid (20x20 cells)
    n_cells = 20
    lat_min, lat_max = valid_coords['latitude'].min(), valid_coords['latitude'].max()
    lon_min, lon_max = valid_coords['longitude'].min(), valid_coords['longitude'].max()

    lat_step = (lat_max - lat_min) / n_cells
    lon_step = (lon_max - lon_min) / n_cells

    # Initialize loccell column with NaN
    df['loccell'] = pd.NA

    # Assign grid cells only to valid coordinates
    valid_coords['grid_lat'] = ((valid_coords['latitude'] - lat_min) / lat_step).astype(int)
    valid_coords['grid_lon'] = ((valid_coords['longitude'] - lon_min) / lon_step).astype(int)
    valid_coords['loccell'] = valid_coords['grid_lat'] * n_cells + valid_coords['grid_lon']
    
    # Update the original dataframe with calculated loccell values
    df.loc[valid_coords.index, 'loccell'] = valid_coords['loccell']

    # Calculate statistics for each grid cell using only valid coordinates
    grid_stats = valid_coords.groupby('loccell').agg({
        'price': ['count', 'mean', 'std', 'min', 'max'],
        'latitude': ['mean', 'min', 'max'],
        'longitude': ['mean', 'min', 'max']
    }).round(2)

    # Filter out cells with too few properties
    min_properties = 10
    valid_grids = grid_stats[grid_stats[('price', 'count')] >= min_properties].index

    # Create the base map centered on Athens
    m = folium.Map(
        location=[valid_coords['latitude'].mean(), valid_coords['longitude'].mean()],
        zoom_start=12,
        tiles='cartodbpositron'
    )

    # Create color map based on average prices
    prices = grid_stats[('price', 'mean')]
    vmin = prices[prices > 0].quantile(0.1)  # 10th percentile
    vmax = prices.quantile(0.9)  # 90th percentile
    colormap = cm.LinearColormap(
        colors=['blue', 'green', 'yellow', 'orange', 'red'],
        vmin=vmin,
        vmax=vmax
    )

    # Add grid cells to the map
    for grid_id in valid_grids:
        grid_data = valid_coords[valid_coords['loccell'] == grid_id]
        
        # Calculate grid cell boundaries
        lat_start = lat_min + (grid_id // n_cells) * lat_step
        lon_start = lon_min + (grid_id % n_cells) * lon_step
        
        bounds = [
            [lat_start, lon_start],
            [lat_start + lat_step, lon_start],
            [lat_start + lat_step, lon_start + lon_step],
            [lat_start, lon_start + lon_step]
        ]
        
        avg_price = grid_stats.loc[grid_id, ('price', 'mean')]
        n_properties = grid_stats.loc[grid_id, ('price', 'count')]
        
        if n_properties >= min_properties:
            popup_text = f"""
            <b>Grid Cell {grid_id}</b><br>
            Properties: {n_properties}<br>
            Avg Price: €{avg_price:,.2f}<br>
            Min Price: €{grid_stats.loc[grid_id, ('price', 'min')]:,.2f}<br>
            Max Price: €{grid_stats.loc[grid_id, ('price', 'max')]:,.2f}<br>
            Std Dev: €{grid_stats.loc[grid_id, ('price', 'std')]:,.2f}
            """
            
            folium.Polygon(
                locations=bounds,
                popup=popup_text,
                color=colormap(avg_price),
                fill=True,
                fillOpacity=0.4,
                weight=1
            ).add_to(m)

    # Add only valid properties as points
    for idx, row in valid_coords.iterrows():
        grid_id = row['loccell']
        if grid_id in valid_grids:
            avg_price = grid_stats.loc[grid_id, ('price', 'mean')]
            color = colormap(avg_price)
        else:
            color = 'gray'
        
        popup_text = f"""
        <b>Property Details:</b><br>
        Price: €{row['price']:,.2f}<br>
        Grid: {row['loccell']}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            popup=popup_text,
            color=color,
            fill=True,
            fillOpacity=0.7
        ).add_to(m)

    # Add color map to the map
    colormap.add_to(m)
    colormap.caption = 'Average Price in Grid Cell'

    # Save the map
    output_path = OUTDIR / f'location_grid_map_{timestamp}.html'
    m.save(str(output_path))
    print(f"\nMap saved to: {output_path}")

    # Save grid information for future use
    grid_info = valid_coords[['latitude', 'longitude', 'price', 'loccell']]
    grid_info.to_parquet(OUTDIR / 'location_grid.parquet')
    print("\nGrid information saved to:", OUTDIR / 'location_grid.parquet')

    # Print insights about grid cells
    valid_stats = grid_stats[grid_stats[('price', 'count')] >= min_properties]
    print("\nGrid Cell Statistics:")
    print(f"\nNumber of valid grid cells: {len(valid_stats)}")
    print(f"Average properties per cell: {valid_stats[('price', 'count')].mean():.1f}")
    print("\nPrice ranges in valid cells:")
    print(f"Minimum: €{valid_stats[('price', 'min')].min():,.2f}")
    print(f"Maximum: €{valid_stats[('price', 'max')].max():,.2f}")
    print(f"Average: €{valid_stats[('price', 'mean')].mean():,.2f}")
    
    # Print information about invalid coordinates
    n_invalid = len(df) - len(valid_coords)
    if n_invalid > 0:
        print(f"\nWarning: {n_invalid} properties ({n_invalid/len(df):.1%}) had invalid coordinates and were not assigned to grid cells")

    return df

if __name__ == "__main__":
    from helpers.data_manip import load_data
    # Load data and create grid when run directly
    df = load_data()
    create_grid_and_map(df) 