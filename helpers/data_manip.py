from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from helpers.paths import Files




def load_data(file_path: Path = Files.CLEAN_DATASET) -> pd.DataFrame:
    df = pd.read_parquet(file_path, engine='pyarrow')
    print('Loaded the dataset!')
    return df

