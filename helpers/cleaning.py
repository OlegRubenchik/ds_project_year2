import pandas as pd
import os
from pathlib import Path
from typing import Union

# ALL PATHS
ROOT = Path(__file__).parent.parent
CLEAN_DATASET = ROOT / 'data' / 'processed' / 'clean_dataset.parquet'

def clean(dataset: Union[Path,str,pd.DataFrame], force: bool = False) -> pd.DataFrame:
    
    if CLEAN_DATASET.exists() and not force:
        print(f"Using existing cleaned dataset from: {CLEAN_DATASET}")
        return None
    
    if isinstance(dataset, (Path, str)):
        
        df = pd.read_excel(dataset)
    else:
        df = dataset
    
    # Cleaning logic
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

    df.index.name = 'n_sample'
    df.columns.name = 'feature_label'
    #TODO
    #Implement cleaning

    # Save as parquet
    print(f"Saving cleaned dataset to: {CLEAN_DATASET}")
    df.to_parquet(CLEAN_DATASET, engine='pyarrow', index=True)
    return df

if __name__ == "__main__":
    print("Enter the path to a dataset you want to use (should be .xlsx): ")
    input_file = Path(input())
    clean(input_file,force=True)

