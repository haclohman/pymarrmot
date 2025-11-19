# PET files are most limited/restrictive, so loop through PET files and match other forcings to those
# For each PET file:
#   1. read PET file
#   2. read matching precip files
#   3. read matching temp files
#   3. read matching usgs files
#   4. combine into single dataframe
import os
import pandas as pd
from pathlib import Path

# directories
pet_dir = Path("C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\pet")
precip_dir = Path("C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\precip")
temp_dir = Path("C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\temp")
usgs_dir = Path("C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\usgs")
output_dir = Path("C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\combined_forcing")

# list of PET files
pet_files = os.listdir(pet_dir)
unique_locations = []
for pet_file in pet_files:
    if pet_file.endswith('.parquet'):
        
        # read PET file
        df_pet = pd.read_parquet(os.path.join(pet_dir, pet_file))
        location_id = pet_file.replace('_pet.parquet','')
        df_pet['value_time'] = pd.to_datetime(df_pet.index)
        df_pet = df_pet.reset_index(drop=True)

        # read matching precip file
        precip_file = os.path.join(precip_dir, location_id + '.parquet')
        df_precip = pd.read_parquet(precip_file)

        # read matching temp file
        temp_file = os.path.join(temp_dir, location_id + '.parquet')
        df_temp = pd.read_parquet(temp_file)
        
        # read matching usgs file
        usgs_file = os.path.join(usgs_dir, location_id + '.parquet')
        df_usgs = pd.read_parquet(usgs_file)
        
        # join dataframes on value_time

        df_combined = pd.merge(df_pet, df_precip, on='value_time', how='inner')
        df_combined = pd.merge(df_combined, df_temp, on='value_time', how='inner')
        df_combined = pd.merge(df_combined, df_usgs, on='value_time', how='left')
        
        # drop and order columns
        df_combined = df_combined[['value_time', 'precip_mm', 'temp_c', 'PET_daily_mm_hourly', 'discharge_mm']]
        # rename columns
        df_combined.columns = ['value_time', 'precip_mm', 'temp_c', 'pet_mm', 'discharge_mm']

        # save combined dataframe to parquet
        output_file = os.path.join(output_dir, location_id + '_combined.parquet')
        df_combined.to_parquet(output_file, index=False)