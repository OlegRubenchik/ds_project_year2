from pathlib import Path
import pandas as pd


f = Path(__file__)
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
PROC_DIR = DATA_DIR / 'processed'
CLEAN_DATASET = PROC_DIR / 'clean_dataset.parquet'


def load_data(file_path: Path = CLEAN_DATASET) -> pd.DataFrame:
    df = pd.read_parquet(file_path, engine='pyarrow')
    print('Loaded the dataset!')
    return df

