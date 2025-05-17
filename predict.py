"""
predict.py

Make price predictions using trained model by providing feature values.
"""

import joblib
from pathlib import Path
from typing import Dict, Union
import pandas as pd
import numpy as np

from helpers.paths import Files
from helpers.config import PRICE_REG_CONFIG

def get_location_cell(latitude: float, longitude: float) -> int:
    """
    Get the location cell number for given coordinates using the same grid system.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        int: Location cell number
    """
    # Load grid info to get boundaries
    grid = pd.read_parquet(Files.LOC_GRID_PARQUET)
    
    # Get grid boundaries
    lat_min, lat_max = grid['latitude'].min(), grid['latitude'].max()
    lon_min, lon_max = grid['longitude'].min(), grid['longitude'].max()
    
    # Check if coordinates are within bounds
    if not (lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max):
        raise ValueError(
            f"Coordinates (lat={latitude}, lon={longitude}) are outside the known area. "
            f"Latitude should be between {lat_min:.4f} and {lat_max:.4f}, "
            f"Longitude should be between {lon_min:.4f} and {lon_max:.4f}"
        )
    
    # Use the same grid size as in location_grid.py
    n_cells = 20
    lat_step = (lat_max - lat_min) / n_cells
    lon_step = (lon_max - lon_min) / n_cells
    
    # Calculate grid cell using the same formula
    grid_lat = int((latitude - lat_min) / lat_step)
    grid_lon = int((longitude - lon_min) / lon_step)
    loccell = grid_lat * n_cells + grid_lon
    
    # Verify this cell exists in our grid
    if loccell not in grid['loccell'].unique():
        raise ValueError(
            f"Calculated cell {loccell} is not present in our grid. "
            "This area might not have enough data for reliable predictions."
        )
    
    return loccell

def predict_price(feature_values: Dict[str, Union[float, int]]) -> float:
    """
    Predict house price using the trained model.
    
    Args:
        feature_values: Dictionary with feature names and their values.
            Required features: build_area, construction_year, number_of_bedrooms,
                             floor_number, latitude, longitude
            Optional features: loccell (will be calculated from lat/lon if not provided)
    
    Returns:
        float: Predicted price in euros
        
    Example:
        >>> features = {
        ...     'build_area': 85,
        ...     'construction_year': 1990,
        ...     'number_of_bedrooms': 2,
        ...     'floor_number': 3,
        ...     'latitude': 37.9838,
        ...     'longitude': 23.7275
        ... }
        >>> price = predict_price(features)
        >>> print(f"Predicted price: €{price:,.2f}")
    """
    # Make a copy to avoid modifying the input
    features = feature_values.copy()
    
    # Calculate loccell if not provided
    if 'loccell' not in features:
        try:
            features['loccell'] = get_location_cell(features['latitude'], features['longitude'])
            print(f"Calculated loccell: {features['loccell']}")
        except Exception as e:
            raise ValueError(f"Error calculating loccell: {str(e)}")
    
    # Validate features
    missing_features = set(PRICE_REG_CONFIG) - set(features.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Load model
    model_info = joblib.load(Files.PRICE_MODEL)
    model = model_info['model']
    
    # Create DataFrame with features in correct order
    features_df = pd.DataFrame([features])[PRICE_REG_CONFIG]
    
    # Make prediction
    predicted_price = model.predict(features_df)[0]
    
    return predicted_price

if __name__ == "__main__":
    print("Choose input method:")
    print("1. Use ready values")
    print("2. Enter values manually")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        # Example usage
        example_features = {
            'build_area': 94,
            'construction_year': 1970,
            'number_of_bedrooms': 1,
            'floor_number': 0,
            'latitude': 38.020516,
            'longitude': 23.728080
        }
        features = example_features
        
    elif choice == "2":
        # Manual input
        features = {}
        features['build_area'] = float(input("Enter build area (in m²): "))
        features['construction_year'] = int(input("Enter construction year: "))
        features['number_of_bedrooms'] = int(input("Enter number of bedrooms: "))
        features['floor_number'] = int(input("Enter floor number: "))
        features['latitude'] = float(input("Enter latitude: "))
        features['longitude'] = float(input("Enter longitude: "))
    
    else:
        print("Invalid choice!")
        exit()
    
    try:
        price = predict_price(features)
        print(f"\nFeatures:")
        for feature, value in features.items():
            print(f"{feature}: {value}")
        print(f"\nPredicted price: €{price:,.2f}")
    except Exception as e:
        print(f"Error: {e}") 