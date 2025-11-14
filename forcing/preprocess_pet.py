# Perfect — here’s your fully integrated, end-to-end PET workflow, now including a function that 
# automatically reads a GeoParquet of basin polygons, extracts centroid latitudes, and uses them in the PET calculations.

# This version seamlessly links your geospatial data → temperature → daily PET (via pyet) → hourly PET (cosine daylight) → Parquet outputs for each basin/station.

import numpy as np
import pandas as pd
import geopandas as gpd
import pyet
import os
import sys
from pathlib import Path

# =============================================================
# 1. Read latitudes from GeoParquet of basin polygons
# =============================================================
def get_latitudes_from_geoparquet(geo_parquet_path, id_column=None):
    """
    Read a GeoParquet file of basin polygons and return centroid latitudes (degrees).
    
    Parameters
    ----------
    geo_parquet_path : str or Path
        Path to the GeoParquet file containing polygon geometries.
    id_column : str, optional
        Column name to use as the station/basin identifier.
    
    Returns
    -------
    pandas.Series
        Series of centroid latitudes (degrees North), indexed by id_column if provided.
    """
    gdf = gpd.read_parquet(geo_parquet_path)
    
    # Ensure geographic coordinates (EPSG:4326)
    if gdf.crs is None:
        raise ValueError("GeoParquet has no CRS. Please set it (e.g., EPSG:4326).")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    
    # Compute centroids
    centroids = gdf.geometry.centroid
    latitudes = centroids.y
    
    # Return Series indexed by ID or default index
    if id_column and id_column in gdf.columns:
        return pd.Series(latitudes.values, index=gdf[id_column].values, name="latitude")
    return pd.Series(latitudes.values, index=gdf.index, name="latitude")

# =============================================================
# 2. Disaggregate daily PET → hourly PET
# =============================================================
def disaggregate_daily_to_hourly(pet_daily_df, lat):
    """
    Disaggregates daily Potential Evapotranspiration (PET) into hourly values
    using a cosine weighting method based on sunrise and sunset times.

    The method assumes PET is 0 at night and follows a sine curve
    during daylight, peaking at solar noon.

    Args:
        pet_daily_df (pd.DataFrame): 
            A DataFrame with a DatetimeIndex (YYYY-MM-DD) and a single 
            column containing daily PET values.
        lat (float): 
            Latitude in radians.

    Returns:
        pd.DataFrame: 
            A DataFrame with an hourly DatetimeIndex (YYYY-MM-DD HH:MM:SS) 
            and a single column of hourly PET values.
    """

    # --- 1. Internal Helper Function: Calculate Solar Parameters ---
    def get_solar_parameters(doy, lat_rad):
        """
        Calculates key solar parameters for a given day(s) of year and latitude.
        All calculations are vectorized.
        """
        
        # Ensure doy is a numpy array
        doy = np.array(doy)

        # Solar declination (in radians)
        # Formula from: FAO-56, Equation 24
        delta = 0.409 * np.sin(((2 * np.pi) / 365) * doy - 1.39)

        # Sunset hour angle (in radians)
        # Formula from: FAO-56, Equation 25
        cos_omega_s = -np.tan(lat_rad) * np.tan(delta)
        
        # Clip to handle polar day/night
        # cos_omega_s > 1: Polar night (no sun)
        # cos_omega_s < -1: Polar day (24h sun)
        cos_omega_s = np.clip(cos_omega_s, -1.0, 1.0)
        
        omega_s = np.arccos(cos_omega_s)

        # Daylength (in hours)
        # Formula from: FAO-56, Equation 34
        D = (24 / np.pi) * omega_s

        # Sunrise and sunset times (in local solar time, hours)
        # 12.0 is solar noon
        sunrise = 12.0 - D / 2.0
        sunset = 12.0 + D / 2.0
        
        # Handle edge cases for polar regions
        sunrise[D == 0] = 0.0  # Polar night
        sunset[D == 0] = 0.0   # Polar night
        
        return D, sunrise, sunset

    # --- 2. Internal Helper Function: Calculate Hourly Weights ---
    def hourly_cosine_weights(D, sunrise, sunset):
        """
        Generates 24 hourly weights for each day.
        The weights form a sine curve between sunrise and sunset and are
        0 at night. All weights for a day sum to 1.0.
        """
        n_days = len(D)
        
        # Create an array of hour midpoints (0.5, 1.5, ..., 23.5)
        # Shape: (24,)
        hour_midpoints = np.arange(0.5, 24.5)
        
        # Tile this array to match the number of days
        # t shape: (n_days, 24)
        t = np.tile(hour_midpoints, (n_days, 1))

        # Use broadcasting to compare t with sunrise/sunset
        # We need sunrise and sunset to be shape (n_days, 1)
        sunrise_b = sunrise[:, None]
        sunset_b = sunset[:, None]
        
        # Handle division by zero for polar night (D=0)
        # D_safe shape: (n_days, 1)
        D_safe = np.where(D == 0, 1e-9, D)[:, None]

        # Calculate the argument of the sine function
        # (pi * (t - sunrise)) / D
        arg = np.pi * (t - sunrise_b) / D_safe

        # Calculate raw weights (sine curve)
        # This is 0 at t=sunrise, 1 at t=noon, 0 at t=sunset
        raw_weights = np.sin(arg)

        # Create a mask for daylight hours
        # True only for hours *between* sunrise and sunset
        is_daytime = (t > sunrise_b) & (t < sunset_b)

        # Apply the mask: 0 for night, sin(arg) for day
        # Also clip at 0 to remove any negative artifacts
        hourly_weights_raw = np.where(is_daytime, np.maximum(0, raw_weights), 0)

        # Normalize weights so they sum to 1.0 for each day
        daily_sum = hourly_weights_raw.sum(axis=1) # Sum across the 24 hours
        
        # Avoid division by zero if a day has 0 sunlight
        daily_sum_safe = np.where(daily_sum == 0, 1.0, daily_sum)

        # Normalize by dividing each row by its sum
        # daily_sum_safe[:, None] makes it (n_days, 1) for broadcasting
        hourly_weights_normalized = hourly_weights_raw / daily_sum_safe[:, None]
        
        # W shape: (n_days, 24)
        return hourly_weights_normalized

    # --- 3. Main Disaggregation Logic ---
    
    # Ensure index is datetime
    try:
        days_index = pd.to_datetime(pet_daily_df.index)
        doy = days_index.dayofyear
        daily_pet_values = pet_daily_df.values # Shape: (n_days, 1)
        original_col_name = pet_daily_df.columns[0]
    except Exception as e:
        raise ValueError(
            "Input DataFrame `pet_daily_df` must have a DatetimeIndex "
            f"and at least one data column. Error: {e}"
        )

    # Calculate solar parameters for all days
    D, sunrise, sunset = get_solar_parameters(doy, lat)

    # Calculate hourly weights for all days
    # W shape: (n_days, 24)
    W = hourly_cosine_weights(D, sunrise, sunset)

    # Disaggregate
    # daily_pet_values (n_days, 1) * W (n_days, 24)
    # NumPy broadcasting multiplies the daily value across all 24 weights
    # pet_hourly_values shape: (n_days, 24)
    pet_hourly_values = daily_pet_values * W

    # --- 4. Format the Output DataFrame ---

    # Create an hourly timestamp for every value
    # 1. Get base days as numpy array
    days_array = days_index.values
    # 2. Get 24 hour-deltas
    hours_array = pd.timedelta_range(start="0H", periods=24, freq="h").values
    
    # 3. Add arrays using broadcasting
    # days_array[:, None] -> (n_days, 1)
    # hours_array[None, :] -> (1, 24)
    # Result: (n_days, 24) array of hourly timestamps
    hourly_timestamps = days_array[:, None] + hours_array[None, :]

    # Flatten both the timestamps and the values
    hourly_idx_flat = hourly_timestamps.reshape(-1)
    pet_hourly_flat = pet_hourly_values.reshape(-1)

    # Create the final DataFrame
    df_hourly = pd.DataFrame(
        pet_hourly_flat, 
        index=hourly_idx_flat, 
        columns=[f'{original_col_name}_hourly']
    )
    df_hourly.index.name = 'datetime'

    return df_hourly

# =============================================================
# 6. Main workflow: hourly temp → hourly PET (using GeoParquet lats)
# =============================================================
def compute_pet_from_hourly_temperature(
    temp_hourly, lat_rad, out_dir
):
    """
    Full PET workflow:
    - Read centroid latitudes from GeoParquet basins
    - Aggregate hourly temperature → daily Tmin/Tmax
    - Compute daily PET (Hargreaves, pyet)
    - Disaggregate daily PET → hourly (cosine daylight)
    - Write one Parquet per station/basin

    Parameters
    ----------
    temp_hourly : DataFrame
        Columns: ['time', 'station_id', 'temp_C']
    geo_parquet_path : str or Path
        Path to GeoParquet file of basin polygons.
    id_column : str
        Column name identifying the basin/station IDs in GeoParquet.
    out_dir : str or Path
        Directory to write hourly PET parquet files.
    """
    out_dir = Path(out_dir)
    #out_dir.mkdir(parents=True, exist_ok=True)

    # set index for temp_hourly to value_time column
    temp_hourly.set_index('value_time', inplace=True)

    # ---- Step 3: aggregate to daily Tmin/Tmax ----
    tmin = temp_hourly.resample("D").min()
    tmax = temp_hourly.resample("D").max()
    tmin_series = tmin['temp_c']
    tmax_series = tmax['temp_c']
    tmean_series = (tmin_series + tmax_series) / 2

    # ---- Step 4: daily PET ----
    pet_daily = pyet.hargreaves(tmean=tmean_series, tmax=tmax_series, tmin=tmin_series, lat=lat_rad, clip_zero=True)
    #convert to dataframe
    pet_daily = pet_daily.to_frame(name='PET_daily_mm')
    # index to datetime
    pet_daily.index = pd.to_datetime(pet_daily.index)
    return pet_daily

# Inputs
geo_parquet_path = "C:\\Data\\fff_all_basins.parquet"
temp_file_dir = "C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\temp"
pet_file_dir = "C:\\Users\\ssheeder\\Repos\\pymarrmot\\forcing\\pymarrmot\\pet"

# get latitudes of centroids
latitudes = get_latitudes_from_geoparquet(geo_parquet_path, id_column="GAGE_ID")
# lists of files
# Get all .parquet temperature files
temp_files = [f for f in os.listdir(temp_file_dir) if f.endswith('.parquet')]

for temp_file in temp_files:
    # only process USGS gage files
    if temp_file.startswith('usgsbasin-'):
        df_temp = pd.read_parquet(os.path.join(temp_file_dir, temp_file))
        
        # get the usgs gage id
        character = "-"
        index = df_temp['location_id'][0].find(character)
        if index != -1:
            usgs_gage_id = df_temp['location_id'][0][index + 1:]
        else:
            print("Character '-'not found in the usgs file name. Exiting the program.")
            sys.exit()
        
        # if the usgs gage id is in the latitudes, calculate PET
        if usgs_gage_id in latitudes.index:
            # get the latitude corresponding to the centroid of the watershed upstream of the usgs gage
            location_lat_dd = latitudes.loc[usgs_gage_id]
            location_lat_rad = np.deg2rad(location_lat_dd)

            # calculate PET and save to parquet
            pet_daily = compute_pet_from_hourly_temperature(
                temp_hourly=df_temp,
                lat_rad=location_lat_rad,
                out_dir=pet_file_dir
            )

            # disaggregate daily PET to hourly
            pet_hourly = disaggregate_daily_to_hourly(pet_daily_df=pet_daily, lat=location_lat_rad)

            # ---- Step 6: save one parquet per basin ----
            pet_file_name = temp_file.replace('.parquet', '_pet.parquet')
            pet_hourly.to_parquet(os.path.join(pet_file_dir, pet_file_name))
        else: # usgs gage id not found in latitudes
            print(f"USGS gage ID {usgs_gage_id} not found in GeoParquet latitudes. Skipping file {temp_file}.")    