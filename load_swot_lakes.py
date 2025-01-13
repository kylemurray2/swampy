import geopandas as gpd
import pandas as pd
from pathlib import Path
import fiona, os
from multiprocessing import Pool
from matplotlib import pyplot as plt
from shapely.geometry import Point


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

def process_shapefile_for_lakes(shapefile_path):
    """
    Function to process a shapefile and extract lake names and polygons.

    Parameters:
    shapefile_path (str or Path): Path to the shapefile.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame containing lake names and polygons.
    """
    try:
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)

        # Ensure the expected columns are present
        if 'lake_name' in gdf.columns:
            # Select only the necessary columns
            gdf = gdf[['lake_name', 'geometry']]
            return gdf
        else:
            print(f"'lake_name' column not found in {shapefile_path}")
            return gpd.GeoDataFrame(columns=['lake_name', 'geometry'])
    except Exception as e:
        print(f"Error processing {shapefile_path}: {e}")
        return gpd.GeoDataFrame(columns=['lake_name', 'geometry'])

def get_unique_lake_polygons_parallel(base_dir, output_shapefile, num_processes=None):
    """
    Function to get unique lake names and polygons from all shapefiles using parallel processing,
    and save them into a new shapefile.

    Parameters:
    base_dir (str): Path to the base directory containing folders with shapefiles.
    output_shapefile (str): Path to the output shapefile to save the combined data.
    num_processes (int): Number of parallel processes to use. If None, the number of CPU cores will be used.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing unique lake names and polygons.
    """
    all_shapefiles = []
    
    # Iterate through each folder in the base directory to find all shapefiles
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".shp"):
                shapefile_path = os.path.join(root, file)
                all_shapefiles.append(shapefile_path)

    if num_processes is None:
        num_processes = os.cpu_count()
    
    # Use multiprocessing Pool to process the shapefiles in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_shapefile_for_lakes, all_shapefiles)

    # Combine the results from all processes into a single GeoDataFrame
    combined_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=results[0].crs if results else None)

    # Remove duplicates based on 'lake_name'
    unique_gdf = combined_gdf.drop_duplicates(subset=['lake_name']).reset_index(drop=True)

    # Save the unique lake names and polygons to a new shapefile
    unique_gdf.to_file(output_shapefile)

    print(f"Saved unique lakes to {output_shapefile}")

    return unique_gdf


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


def process_shapefile_for_nearest_lake(shp_file, target_point, date_str, max_distance):
    """
    Process a single shapefile and find the WSE of the nearest lake to the given point, within a maximum distance.
    
    Parameters:
    shp_file (str): Path to the shapefile.
    target_point (shapely.geometry.Point): The point (latitude/longitude) to find the nearest lake to.
    date_str (str): The date string corresponding to the folder name (YYYYMMDD).
    max_distance (float): Maximum allowable distance (in meters) from the target point.

    Returns:
    dict: A dictionary containing the date, WSE value, and lake name for the nearest lake, or None if not found.
    """
    try:
        # Load the shapefile into a GeoDataFrame
        gdf = gpd.read_file(shp_file)

        # Ensure the GeoDataFrame has a valid CRS
        if gdf.crs is None:
            print(f"Shapefile {shp_file} has no CRS, skipping.")
            return None

        # Reproject both the GeoDataFrame and the target point to a projected CRS (e.g., EPSG:3857)
        gdf = gdf.to_crs(epsg=3857)  # Project to Web Mercator
        target_point_projected = gpd.GeoSeries([target_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

        # Check if the 'wse' and 'lake_name' columns exist and there's valid geometry
        if 'wse' in gdf.columns and 'lake_name' in gdf.columns and gdf.geometry.notnull().all():
            # Calculate the distance of all geometries from the projected target_point
            gdf['distance'] = gdf.geometry.distance(target_point_projected)

            # Filter to include only lakes within the max_distance
            gdf_within_distance = gdf[gdf['distance'] <= max_distance]

            # If no lakes are within the max_distance, return None
            if gdf_within_distance.empty:
                return None

            # Find the nearest lake (the one with the smallest distance)
            nearest_lake = gdf_within_distance.loc[gdf_within_distance['distance'].idxmin()]

            # Extract the WSE value and lake name of the nearest lake
            wse_value = nearest_lake['wse']
            lake_name = nearest_lake['lake_name']
            return {'date': pd.to_datetime(date_str, format='%Y%m%d'), 'wse': wse_value, 'lake_name': lake_name}

    except Exception as e:
        print(f"Error processing {shp_file}: {e}")

    return None

def get_lake_wse_time_series_by_location(base_dir, latitude, longitude, max_distance, num_processes=None):
    """
    Create a time series of WSE for the lake nearest to a given latitude/longitude point
    from shapefiles using parallel processing and selective column loading, within a maximum distance.
    
    Parameters:
    base_dir (str): Path to the base directory containing YYYYMMDD folders with shapefiles.
    latitude (float): Latitude of the target location.
    longitude (float): Longitude of the target location.
    max_distance (float): Maximum allowable distance (in meters) from the target point.
    num_processes (int): The number of parallel processes to use. If None, it will use the number of CPU cores available.
    
    Returns:
    pd.DataFrame: DataFrame containing the date, WSE, and lake name for the nearest lake.
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
                tasks.append((shp_file, target_point, date_str, max_distance))

    # Use a pool of workers to process the shapefiles in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_shapefile_for_nearest_lake, tasks)

    # Collect the results and filter out None values
    time_series_data = [res for res in results if res is not None]

    # Check if time_series_data is empty before creating a DataFrame
    if not time_series_data:
        print(f"No data found for the location ({latitude}, {longitude}) within {max_distance} meters.")
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
    output_shp_fn = './swot/lakes_polygons.shp'
    if not os.path.isfile('lake_names.txt'):
        lake_names = get_unique_lake_polygons_parallel(base_dir,output_shp_fn)
    else:
        with open('lake_names.txt', 'r') as file:
            lake_names = [line.strip() for line in file.readlines()]  # Read each line, stripping newline characters

    # latitude,longitude = 34.579, -119.948 # Cachuma latitude,longitude
    latitude,longitude = 43.457, -124.251 # oregon site? 

    num_processes = 20  # Specify the number of parallel processes, or leave it None for automatic

    # Get the WSE time series for the water body near the specified lat/lon
    max_dist = 500 # Meters
    wse_df = get_lake_wse_time_series_by_location(base_dir, latitude, longitude, max_dist, num_processes=num_processes)

    
    lake_name = 'Cachuma'  # The lake name you are interested in
    # # Get the WSE time series for the specified lake using parallel processing and selective column loading
    # wse_df = get_lake_wse_time_series_parallel(base_dir, lake_name)

    # Plot the time series if data is available
    if not wse_df.empty:
        plot_wse_time_series(wse_df, lake_name)
    else:
        print(f"No data found for lake_name {lake_name}")

    # Save to CSV if needed
    wse_df.to_csv(f'wse_data{str(latitude)}{str(longitude)}.csv', index=False)
