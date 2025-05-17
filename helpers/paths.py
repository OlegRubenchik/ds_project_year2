from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = Path(__file__).parent.parent

class Dirs:
    DATA_MAPS_DIR = ROOT / 'data' / 'insights' / 'maps'
    DATA_STAT_DIR = ROOT / 'data' / 'insights' / 'statistics'


class Files:
    
    # Maps
    EXACT_PRICE_MAP = ROOT / 'data' / 'insights' / 'maps' / 'athens_exact_price_map.html'
    METRO_DISTANCE_MAP = ROOT / 'data' / 'insights' / 'maps' / 'athens_metro_distance_map.html'
    

    # Statistics
    PRICE_DISTR_TABLE = ROOT / 'data' / 'insights' / 'statistics' / 'price_distribution.png'
    PRICE_CORR_HEATMAP = ROOT / 'data' / 'insights' / 'statistics' / 'price_feature_correlation_heatmap.png'
    PRICE_SCATTER_PLOTS = ROOT / 'data' / 'insights' / 'statistics' / 'price_feature_correlations_scatter.png'
    PRICE_CORR_TABLE = ROOT / 'data' / 'insights' / 'statistics' / 'price_feature_correlations_table.png'
    PRICE_METRO_DIST_CORR = ROOT / 'data' / 'insights' / 'statistics' / 'metro_distance_by_price_class.png'

    # Parquets and CSV's
    LOC_GRID_MAP = ROOT / 'data' / 'insights' / 'maps' / 'location_grid_map.html'
    LOC_GRID_PARQUET = ROOT / 'data' / 'processed' / 'location_grid.parquet'
    CLEAN_DATASET = ROOT / 'data' / 'processed' / 'clean_dataset.parquet'
    METRO_STATIONS = ROOT / 'data' / 'raw' / 'athens_metro_coordinates.csv'

def create_directories():
    """Create all necessary directories for the project."""
    # Main data directories
    (ROOT / 'data').mkdir(exist_ok=True)
    (ROOT / 'data' / 'insights').mkdir(exist_ok=True)
    (ROOT / 'data' / 'processed').mkdir(exist_ok=True)
    (ROOT / 'data' / 'raw').mkdir(exist_ok=True)
    
    # Insights subdirectories
    Dirs.DATA_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    Dirs.DATA_STAT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created all necessary directories")