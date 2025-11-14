# required imports
import os
import pandas as pd
import pyarrow.parquet as pq

# directories
input_precip_dir = "C:\\Data\\nwm30_forcing\\rainrate"
output_precip_dir = "C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\precip"

# lists of files
precip_files = os.listdir(input_precip_dir)

# ----------------------------Precipitation Data Processing----------------------------
unique_locations = []
for precip_file in precip_files:
    # if precip_file is in parquet format, convert to pandas dataframe
    if precip_file.endswith('.parquet'):
        table = pq.read_table(os.path.join(input_precip_dir, precip_file))
        df_met_file = table.to_pandas()
        # get list of unique locations in location_id column
        unique_locations = df_met_file['location_id'].unique()
        for unique_location in unique_locations:
            df_precip_location = df_met_file[df_met_file['location_id'] == unique_location]
            # calculate timestep (hourly) precipitation from rate in mm/sec
            df_precip_location['precip_mm'] = df_precip_location['value'] * 3600
            # keep select columns
            df_precip_location = df_precip_location[['location_id', 'value_time', 'precip_mm']]
            
            # save to parquet
            output_parquet = os.path.join(output_precip_dir, unique_location + '.parquet')
            if os.path.exists(output_parquet):
                # append to existing parquet
                table_existing = pq.read_table(output_parquet)
                df_existing = table_existing.to_pandas()
                df_combined = pd.concat([df_existing, df_precip_location])
                df_combined.to_parquet(output_parquet, index=False)
            else:
                df_precip_location.to_parquet(output_parquet)