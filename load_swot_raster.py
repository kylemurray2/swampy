'''
This script is used to load the SWOT L2 HR Raster NetCDF files and extract the Water Surface Elevation (WSE) time series for a given latitude/longitude.
'''

import xarray as xr
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from pyproj import Proj
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def load_example_file(fn='/d/surfaceWater/westCoastData/swot/RasterData/SWOT_L2_HR_Raster_100m_UTM10R_N_x_x_x_540_026_050F_20230603T091719_20230603T091721_PGC0_01.nc'):
    '''
    fn: Path to the SWOT L2 HR Raster NetCDF file
    '''
    # Load the NetCDF file using xarray
    ds = xr.open_dataset(fn)

    # Print the dataset to inspect its contents
    print(ds)

    # Extract the Water Surface Elevation (WSE) variable
    if 'wse' in ds.variables:
        wse = ds['wse']  # Assuming 'wse' is the name of the variable for Water Surface Elevation
        print(wse)

        # Plot the WSE data
        plt.figure(figsize=(10, 6))
        wse.plot()
        plt.title('Water Surface Elevation (WSE)')
        plt.show()
    else:
        print("The WSE variable ('wse') is not found in this dataset.")

# Note: Removed the call to load_example_file() as it is for demonstration purposes.

def point_in_bbox(lat, lon, dataset):
    """
    Check if a lat/lon point is within the bounding box of a dataset.
    """
    lat_min = dataset.latitude.min().item()
    lat_max = dataset.latitude.max().item()
    lon_min = dataset.longitude.min().item()
    lon_max = dataset.longitude.max().item()

    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

def latlon_to_utm_zone(lat, lon):
    """
    Convert lat/lon to UTM zone (number and letter).
    """
    # Determine UTM zone number (1 to 60, covering longitude)
    utm_zone_number = int((lon + 180) // 6) + 1

    # Determine UTM latitudinal band (C to X, excluding I and O)
    if -80 <= lat <= 84:  # UTM is defined between 80S and 84N
        utm_band_letters = "CDEFGHJKLMNPQRSTUVWX"  # UTM latitudinal bands
        lat_band_index = int((lat + 80) // 8)
        utm_lat_band = utm_band_letters[lat_band_index]
    else:
        raise ValueError(f"Latitude {lat} is out of UTM range (-80, 84)")

    return f"{utm_zone_number}{utm_lat_band}"

def filter_files_by_utm(lat, lon, folder_path):
    """
    Filter NetCDF files based on whether their UTM zone overlaps with the given lat/lon.
    """
    utm_zone = latlon_to_utm_zone(lat, lon)
    print(f"Target UTM zone: {utm_zone}")

    filtered_files = []

    # Regex pattern to extract UTM zone from file names (e.g., 'UTM10S')
    pattern = re.compile(r"UTM(\d{1,2}[C-X])")

    # Loop through all NetCDF files in the directory
    for file in os.listdir(folder_path):
        if file.endswith(".nc"):
            match = pattern.search(file)
            if match:
                file_utm_zone = match.group(1)
                if file_utm_zone == utm_zone:
                    print(f"File {file} matches UTM zone {utm_zone}")
                    filtered_files.append(file)

    return filtered_files

def extract_timestamp_from_filename(filename):
    """
    Extracts the timestamp from a given SWOT filename.
    """
    # Extract the timestamp using regex pattern to match the datetime part in the filename
    match = re.search(r'_(\d{8}T\d{6})_', filename)
    if match:
        timestamp_str = match.group(1)
        return pd.to_datetime(timestamp_str, format='%Y%m%dT%H%M%S')
    else:
        return None

def process_file(args):
    """
    Process a single NetCDF file to extract WSE at the given latitude and longitude.
    """
    file, lat, lon, folder_path = args
    file_path = os.path.join(folder_path, file)

    # Extract timestamp from the filename
    file_timestamp = extract_timestamp_from_filename(file)
    if file_timestamp is None:
        print(f"Could not extract timestamp from file: {file}")
        return None

    try:
        # Open the dataset
        with xr.open_dataset(file_path) as ds:
            # Check if the point is within the latitude/longitude bounds of the dataset
            lat_min = ds.latitude.min().item()
            lat_max = ds.latitude.max().item()
            lon_min = ds.longitude.min().item()
            lon_max = ds.longitude.max().item()

            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                print(f"Point is within the bounds of {file}")

                # Compute the absolute difference
                distance = abs(ds.latitude - lat) + abs(ds.longitude - lon)

                # Find the indices of the minimum distance
                lat_lon_idx = np.unravel_index(distance.argmin(), distance.shape)

                # Get WSE value at the nearest grid point
                wse_value = ds['wse'].isel(y=lat_lon_idx[0], x=lat_lon_idx[1]).values.item()

                return {'date': file_timestamp, 'wse': wse_value}
            else:
                print(f"Point is outside the bounds of {file}")
                return None
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def extract_wse_time_series(lat, lon, matching_files, folder_path):
    """
    Extract a time series of WSE data for a given latitude/longitude across multiple NetCDF files.
    """
    wse_time_series = []

    args_list = [(file, lat, lon, folder_path) for file in matching_files]

    # Use threading to process files in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, args_list)

    # Filter out None results
    wse_time_series = [res for res in results if res is not None]

    # Convert to DataFrame for easier handling
    if wse_time_series:
        wse_df = pd.DataFrame(wse_time_series)

        # Sort by date to ensure the time series is in order
        wse_df = wse_df.sort_values(by='date').reset_index(drop=True)
    else:
        print("No valid data found for the given latitude/longitude.")
        wse_df = pd.DataFrame()  # Return an empty DataFrame if no data is found

    return wse_df

# Example usage
if __name__ == "__main__":
    folder_path = './data/swot/RasterData'  # Path to the folder containing NetCDF files
    # Define the UTM zone you are working in
    # utm_proj = Proj(proj="utm", zone=10, south=True, ellps="WGS84")
    # latitude, longitude = 38.27, -121.68
    latitude, longitude = 34.58555, -119.9792  # Cachuma Reservoir, California
    # Filter the files based on UTM zone overlap with the lat/lon
    matching_files = filter_files_by_utm(latitude, longitude, folder_path)

    # Output the filtered file list
    print("Matching files:")
    for file in matching_files:
        print(file)

    # Extract the WSE time series for the given lat/lon
    wse_df = extract_wse_time_series(latitude, longitude, matching_files, folder_path)

    if not wse_df.empty:
        plt.figure()
        plt.plot(wse_df['date'], wse_df['wse'], '.')
        plt.xlabel('Date')
        plt.ylabel('Water Surface Elevation (WSE)')
        plt.title('WSE Time Series')
        plt.show()
        print(wse_df)

        # Save the time series to a CSV if needed
        wse_df.to_csv('wse_raster_time_series.csv', index=False)
    else:
        print("No data available to plot or save.")
