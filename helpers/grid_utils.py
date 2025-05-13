import pandas as pd
from pathlib import Path
from typing import Union, Dict, Optional, List, Tuple

# Path to grid information
ROOT = Path(__file__).parent.parent
GRID_INFO_PATH = ROOT / 'data' / 'insights' / 'location_grid.parquet'

class GridAssigner:
    def __init__(self, grid_info_path: Path = GRID_INFO_PATH):
        """Initialize the grid assigner with grid information"""
        if not grid_info_path.exists():
            raise FileNotFoundError(f"Grid information file not found at {grid_info_path}")
        
        # Load grid information
        self.grid_info = pd.read_parquet(grid_info_path)
        
        # Calculate and store grid boundaries
        self.bounds = {
            'lat_min': self.grid_info['latitude'].min(),
            'lat_max': self.grid_info['latitude'].max(),
            'lon_min': self.grid_info['longitude'].min(),
            'lon_max': self.grid_info['longitude'].max()
        }
        
        # Default grid size
        self.n_cells = 20
        
        # Calculate cell sizes
        self.lat_step = (self.bounds['lat_max'] - self.bounds['lat_min']) / self.n_cells
        self.lon_step = (self.bounds['lon_max'] - self.bounds['lon_min']) / self.n_cells

    def get_cell_id(self, lat: float, lon: float) -> Optional[int]:
        """
        Get the grid cell ID for given coordinates.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            Optional[int]: Grid cell ID or None if coordinates are out of bounds
        """
        # Check if coordinates are within bounds
        if (lat < self.bounds['lat_min'] or lat > self.bounds['lat_max'] or 
            lon < self.bounds['lon_min'] or lon > self.bounds['lon_max']):
            return None
        
        # Calculate grid coordinates
        grid_lat = int((lat - self.bounds['lat_min']) / self.lat_step)
        grid_lon = int((lon - self.bounds['lon_min']) / self.lon_step)
        
        # Handle edge cases
        grid_lat = min(grid_lat, self.n_cells - 1)
        grid_lon = min(grid_lon, self.n_cells - 1)
        
        return grid_lat * self.n_cells + grid_lon
    
    def get_cell_bounds(self, cell_id: int) -> Optional[Dict[str, float]]:
        """
        Get the boundaries of a specific grid cell.
        
        Args:
            cell_id (int): Grid cell ID
            
        Returns:
            Optional[Dict[str, float]]: Dictionary with cell boundaries or None if invalid cell_id
        """
        if not (0 <= cell_id < self.n_cells * self.n_cells):
            return None
        
        grid_lat = cell_id // self.n_cells
        grid_lon = cell_id % self.n_cells
        
        return {
            'lat_min': self.bounds['lat_min'] + grid_lat * self.lat_step,
            'lat_max': self.bounds['lat_min'] + (grid_lat + 1) * self.lat_step,
            'lon_min': self.bounds['lon_min'] + grid_lon * self.lon_step,
            'lon_max': self.bounds['lon_min'] + (grid_lon + 1) * self.lon_step
        }
    
    def get_cell_stats(self, cell_id: int) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific grid cell.
        
        Args:
            cell_id (int): Grid cell ID
            
        Returns:
            Optional[Dict[str, float]]: Dictionary with cell statistics or None if invalid cell_id
        """
        cell_data = self.grid_info[self.grid_info['grid_id'] == cell_id]
        if len(cell_data) == 0:
            return None
        
        return {
            'count': len(cell_data),
            'avg_price': cell_data['price'].mean(),
            'min_price': cell_data['price'].min(),
            'max_price': cell_data['price'].max(),
            'std_price': cell_data['price'].std()
        }

def assign_to_cell(lat: float, lon: float) -> Optional[int]:
    """
    Quick utility function to assign coordinates to a grid cell.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        Optional[int]: Grid cell ID or None if coordinates are out of bounds or NaN
    """
    # Check for NaN values
    if pd.isna(lat) or pd.isna(lon):
        return None
        
    try:
        assigner = GridAssigner()
        return assigner.get_cell_id(lat, lon)
    except FileNotFoundError:
        print("Warning: Grid information file not found")
        return None

# Example usage
if __name__ == "__main__":
    # Example coordinates (Athens city center)
    test_coords = [
        (37.9838, 23.7275),  # Athens city center
        (37.9755, 23.7348),  # Acropolis
        (37.9908, 23.7033)   # Kerameikos
    ]
    
    try:
        grid = GridAssigner()
        
        print("\nTesting coordinate assignment:")
        for lat, lon in test_coords:
            cell_id = grid.get_cell_id(lat, lon)
            print(f"\nCoordinates: {lat}, {lon}")
            print(f"Assigned to cell: {cell_id}")
            
            if cell_id is not None:
                bounds = grid.get_cell_bounds(cell_id)
                stats = grid.get_cell_stats(cell_id)
                
                print("Cell boundaries:", bounds)
                if stats:
                    print(f"Properties in cell: {stats['count']}")
                    print(f"Average price: €{stats['avg_price']:,.2f}")
    
    except FileNotFoundError:
        print("Could not find grid information file. Please ensure it exists at:", GRID_INFO_PATH) 