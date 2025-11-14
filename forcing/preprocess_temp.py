# required imports
import os
import pandas as pd
import pyarrow.parquet as pq

# directories
input_temp_dir = "C:\\Data\\nwm30_forcing\\temperature"
output_temp_dir = "C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\temp"

# lists of files
# Get all .parquet files
temp_files = [f for f in os.listdir(input_temp_dir) if f.endswith('.parquet')]
temp_files = os.listdir(input_temp_dir)

# ----------------------------Temperature Data Processing----------------------------

for temp_file in temp_files:
    # if temp_file is in parquet format, convert to pandas dataframe
    if temp_file.endswith('.parquet'):
        table = pq.read_table(os.path.join(input_temp_dir, temp_file))
        df_temp_file = table.to_pandas()
        for unique_location in df_temp_file['location_id'].unique():
            df_temp_location = df_temp_file[df_temp_file['location_id'] == unique_location]
            # calculate timestep (hourly) temp (c) from temp (K)
            df_temp_location['temp_c'] = df_temp_location['value'] - 273.15
            # keep select columns
            df_temp_location = df_temp_location[['location_id', 'value_time', 'temp_c']]
            # save to parquet
            output_parquet = os.path.join(output_temp_dir, unique_location + '.parquet')
            if os.path.exists(output_parquet):
                # append to existing parquet
                table_existing = pq.read_table(output_parquet)
                df_existing = table_existing.to_pandas()
                df_combined = pd.concat([df_existing, df_temp_location])
                df_combined.to_parquet(output_parquet, index=False)
            else:
                df_temp_location.to_parquet(output_parquet)