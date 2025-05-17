from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = Path(__file__).parent.parent

class Dirs:
    DATA_MAPS_DIR = ROOT / 'data' / 'insights' / 'maps'
    DATA_STAT_DIR = ROOT / 'data' / 'insights' / 'statistics'
    MODELS_DIR = ROOT / 'models' / 'saved'
    DATA_PROCESSED_DIR = ROOT / 'data' / 'processed'
    DATA_RAW_DIR = ROOT / 'data' / 'raw'


class Files:
    
    # Maps
    EXACT_PRICE_MAP = Dirs.DATA_MAPS_DIR / 'athens_exact_price_map.html'
    METRO_DISTANCE_MAP = Dirs.DATA_MAPS_DIR / 'athens_metro_distance_map.html'
    LOC_GRID_MAP = Dirs.DATA_MAPS_DIR / 'location_grid_map.html'

    # Statistics
    PRICE_DISTR_TABLE = Dirs.DATA_STAT_DIR / 'price_distribution.png'
    PRICE_CORR_HEATMAP = Dirs.DATA_STAT_DIR / 'price_feature_correlation_heatmap.png'
    PRICE_SCATTER_PLOTS = Dirs.DATA_STAT_DIR / 'price_feature_correlations_scatter.png'
    PRICE_CORR_TABLE = Dirs.DATA_STAT_DIR / 'price_feature_correlations_table.png'
    PRICE_METRO_DIST_CORR = Dirs.DATA_STAT_DIR / 'metro_distance_by_price_class.png'
    PRICE_MODEL_REGRESSION_METRICS = Dirs.DATA_STAT_DIR / 'price_model_regression_metrics.png'

    # Parquets and CSV's
    
    LOC_GRID_PARQUET = Dirs.DATA_PROCESSED_DIR / 'location_grid.parquet'
    CLEAN_DATASET = Dirs.DATA_PROCESSED_DIR / 'clean_dataset.parquet'
    METRO_STATIONS = Dirs.DATA_RAW_DIR / 'athens_metro_coordinates.csv'

    # Models
    PRICE_MODEL = Dirs.MODELS_DIR / 'price_model.joblib'

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