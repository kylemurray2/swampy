import geopandas as gpd
import pandas as pd
from pathlib import Path
import fiona, os
from multiprocessing import Pool
from matplotlib import pyplot as plt


def process_shapefile(shp_file, lake_name, date_str):
    """
    Process a single shapefile to extract WSE for a lake using selective column loading.
    
    Parameters:
    shp_file (str): Path to the shapefile.
    lake_name (str): Name of the lake to search for.
    date_str (str): Date string corresponding to the folder name (YYYYMMDD).
    
    Returns:
    dict: A dictionary containing the date and WSE value for the lake, or None if not found.
    """
    # Open only the columns 'lake_name' and 'wse' using fiona
    print(shp_file)
    with fiona.open(shp_file, 'r') as source:
        # Loop through records in the shapefile
        for feature in source:
            if lake_name.lower() in feature['properties']['lake_name'].lower():
                wse_value = feature['properties']['wse']
                return {'date': pd.to_datetime(date_str, format='%Y%m%d'), 'wse': wse_value}
    return None

def process_shapefile_for_lakes(shp_file):
    """
    Process a single shapefile and extract unique lake names.
    
    Parameters:
    shp_file (str): Path to the shapefile.

    Returns:
    set: A set of unique lake names from the shapefile.
    """
    lake_names = set()
    
    try:
        # Load the shapefile into a GeoDataFrame
        gdf = gpd.read_file(shp_file)

        # Check if 'lake_name' column exists, then add all lake names to the set
        if 'lake_name' in gdf.columns:
            lake_names.update(gdf['lake_name'].dropna().unique())
    
    except Exception as e:
        print(f"Error processing {shp_file}: {e}")
    
    return lake_names

def get_unique_lake_names_parallel(base_dir, num_processes=None):
    """
    Function to get a unique list of lake names from all shapefiles using parallel processing.

    Parameters:
    base_dir (str): Path to the base directory containing YYYYMMDD folders with shapefiles.
    num_processes (int): Number of parallel processes to use. If None, the number of CPU cores will be used.

    Returns:
    list: A sorted list of unique lake names across all shapefiles.
    """
    all_shapefiles = []
    
    # Iterate through each date folder in the base directory to find all shapefiles
    for date_folder in sorted(Path(base_dir).iterdir()):
        if date_folder.is_dir():
            shapefiles = list(date_folder.glob("*.shp"))
            all_shapefiles.extend(shapefiles)
    if num_processes is None:
        num_processes = os.cpu_count()
    # Use multiprocessing Pool to process the shapefiles in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_shapefile_for_lakes, all_shapefiles)

    # Combine the results from all processes
    unique_lake_names = set().union(*results)
    with open('lake_names.txt', 'w') as f:
        for lake_name in unique_lake_names:
            f.write(f"{lake_name}\n")
    # Convert the set to a sorted list and return
    return sorted(unique_lake_names)

def get_lake_wse_time_series_parallel(base_dir, lake_name, num_processes=None):
    """
    Create a time series of WSE for a single lake from shapefiles using parallel processing and selective column loading.
    
    Parameters:
    base_dir (str): Path to the base directory containing YYYYMMDD folders with shapefiles.
    lake_name (str): The name of the lake you are interested in.
    num_processes (int): The number of parallel processes to use. If None, it will use the number of CPU cores available.
    
    Returns:
    pd.DataFrame: DataFrame containing the date and WSE for the lake.
    """
    tasks = []

    # If num_processes is not specified, use os.cpu_count() to get the default
    if num_processes is None:
        num_processes = os.cpu_count()

    # Print the number of processes being used
    print(f"Using {num_processes} processes for parallel execution.")

    # Iterate through each date folder in the base directory
    for date_folder in sorted(Path(base_dir).iterdir()):
        if date_folder.is_dir():
            date_str = date_folder.name  # Directory name is YYYYMMDD
            shapefiles = list(date_folder.glob("*Obs*.shp"))  # Find all shapefiles

            # Add tasks to the task list for parallel processing
            for shp_file in shapefiles:
                tasks.append((shp_file, lake_name, date_str))

    # Use a pool of workers to process the shapefiles in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_shapefile, tasks)

    # Collect the results and filter out None values
    time_series_data = [res for res in results if res is not None]

    # Check if time_series_data is empty before creating a DataFrame
    if not time_series_data:
        print(f"No data found for lake_name {lake_name}")
        return pd.DataFrame()  # Return an empty DataFrame

    # Create a DataFrame from the collected data
    df = pd.DataFrame(time_series_data)

    # Sort by date to ensure the time series is in order
    df = df.sort_values(by='date').reset_index(drop=True)

    return df

def process_shapefile_for_nearest_lake(shp_file, target_point, date_str):
    """
    Process a single shapefile and find the WSE of the nearest lake to the given point.
    
    Parameters:
    shp_file (str): Path to the shapefile.
    target_point (shapely.geometry.Point): The point (latitude/longitude) to find the nearest lake to.
    date_str (str): The date string corresponding to the folder name (YYYYMMDD).

    Returns:
    dict: A dictionary containing the date and WSE value for the nearest lake, or None if not found.
    """
    try:
        # Load the shapefile into a GeoDataFrame
        gdf = gpd.read_file(shp_file)

        # Check if the 'wse' column exists and there's valid geometry
        if 'wse' in gdf.columns and gdf.geometry.notnull().all():
            # Calculate the distance of all geometries from the target_point
            gdf['distance'] = gdf.geometry.distance(target_point)

            # Find the nearest lake (the one with the smallest distance)
            nearest_lake = gdf.loc[gdf['distance'].idxmin()]

            # Extract the WSE value of the nearest lake
            wse_value = nearest_lake['wse']
            return {'date': pd.to_datetime(date_str, format='%Y%m%d'), 'wse': wse_value}

    except Exception as e:
        print(f"Error processing {shp_file}: {e}")

    return None

from shapely.geometry import Point
def get_lake_wse_time_series_by_location(base_dir, latitude, longitude, num_processes=None):
    """
    Create a time series of WSE for the lake nearest to a given latitude/longitude point
    from shapefiles using parallel processing and selective column loading.
    
    Parameters:
    base_dir (str): Path to the base directory containing YYYYMMDD folders with shapefiles.
    latitude (float): Latitude of the target location.
    longitude (float): Longitude of the target location.
    num_processes (int): The number of parallel processes to use. If None, it will use the number of CPU cores available.
    
    Returns:
    pd.DataFrame: DataFrame containing the date and WSE for the nearest lake.
    """
    tasks = []
    target_point = Point(longitude, latitude)  # Create a shapely Point from lat/lon

    # If num_processes is not specified, use os.cpu_count() to get the default
    if num_processes is None:
        num_processes = os.cpu_count()

    # Print the number of processes being used
    print(f"Using {num_processes} processes for parallel execution.")

    # Iterate through each date folder in the base directory
    for date_folder in sorted(Path(base_dir).iterdir()):
        if date_folder.is_dir():
            date_str = date_folder.name  # Directory name is YYYYMMDD
            shapefiles = list(date_folder.glob("*Obs*.shp"))  # Find all shapefiles

            # Add tasks to the task list for parallel processing
            for shp_file in shapefiles:
                tasks.append((shp_file, target_point, date_str))

    # Use a pool of workers to process the shapefiles in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_shapefile_for_nearest_lake, tasks)

    # Collect the results and filter out None values
    time_series_data = [res for res in results if res is not None]

    # Check if time_series_data is empty before creating a DataFrame
    if not time_series_data:
        print(f"No data found for the location ({latitude}, {longitude})")
        return pd.DataFrame()  # Return an empty DataFrame

    # Create a DataFrame from the collected data
    df = pd.DataFrame(time_series_data)

    # Sort by date to ensure the time series is in order
    df = df.sort_values(by='date').reset_index(drop=True)

    return df

def plot_wse_time_series(df, lake_name):
    """
    Plot the WSE time series for the given lake.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing date and WSE values.
    lake_name (str): The name of the lake for labeling the plot.
    
    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['wse'], marker='o', linestyle='-', color='b')
    plt.title(f'WSE Time Series for Lake {lake_name}')
    plt.xlabel('Date')
    plt.ylabel('Water Surface Elevation (WSE)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    base_dir = './swot/LakesData'  # Base directory containing YYYYMMDD folders
    
    unique_lake_names = get_unique_lake_names_parallel(base_dir)
    latitude = 34.579  # Example latitude
    longitude = -119.948  # Example longitude
    num_processes = 4  # Specify the number of parallel processes, or leave it None for automatic

    # Get the WSE time series for the water body near the specified lat/lon
    wse_df = get_lake_wse_time_series_by_location(base_dir, latitude, longitude, num_processes=num_processes)

    
    lake_name = 'Cachuma'  # The lake name you are interested in
    # # Get the WSE time series for the specified lake using parallel processing and selective column loading
    # wse_df = get_lake_wse_time_series_parallel(base_dir, lake_name)

    # Plot the time series if data is available
    if not wse_df.empty:
        plot_wse_time_series(wse_df, lake_name)
    else:
        print(f"No data found for lake_name {lake_name}")

    # Save to CSV if needed
    wse_df.to_csv('wse_data.csv', index=False)
