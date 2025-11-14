# required imports
import os
import pandas as pd
import pyarrow.parquet as pq

# file to process USGS observation data for use in PyMARRMoT
usgs_data = "C:\\Data\\usgs_obs.parquet"
usgs_area = "C:\\Data\\usgs_basin_attr_drainage_area.all.parquet"
output_dir = "C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\usgs"

# ----------------------------USGS Data Processing----------------------------
table = pq.read_table(usgs_data)
df_usgs_data = table.to_pandas()
table = pq.read_table(usgs_area)
df_usgs_area = table.to_pandas()

for location in df_usgs_data['location_id'].unique():
    df_usgs_location = df_usgs_data[df_usgs_data['location_id'] == location]
    df_usgs_location = df_usgs_location.merge(df_usgs_area, how='left', left_on='location_id', right_on='location_id')
    # calculate discharge - convert from m3/sec to mm/hour using drainage area
    # mm/hour = (m3/sec) * (3600 sec/hour) * (1000 mm/m) / (drainage area in km2) / (1000000 m2/km2)
    df_usgs_location['discharge_mm'] = df_usgs_location['value'] * 3600 / df_usgs_location['attribute_value'] / 1000
    # keep select columns
    df_usgs_location = df_usgs_location[['value_time', 'discharge_mm']]
    
    # save to parquet
    # add 'basin' to file name to be consistent with other forcings
    location = location.replace('-','basin-')
    output_parquet = os.path.join(output_dir, location + '.parquet')
    df_usgs_location.to_parquet(output_parquet)