#find all nans in the combined_forcing parquet files

import os
import pandas as pd
from pathlib import Path

# directories
forcing_dir = Path("C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\combined_forcing")

# list of files
forcing_files = os.listdir(forcing_dir)

for forcing_file in forcing_files:
    if forcing_file.endswith('.parquet'):
        file_path = forcing_dir / forcing_file
        df = pd.read_parquet(file_path)
        
        # Check for NaNs in the DataFrame
        nan_columns = df.columns[df.isna().any()].tolist()
        
        if any(item != 'discharge_mm' for item in nan_columns):
            print(f"File: {forcing_file} has NaNs in columns: {nan_columns}")
            exit()

print("No unexpected NaNs found in any forcing files.")
        