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
    lake_name = 'Cachuma'  # The lake name you are interested in

    # Get the WSE time series for the specified lake using parallel processing and selective column loading
    wse_df = get_lake_wse_time_series_parallel(base_dir, lake_name)

    # Plot the time series if data is available
    if not wse_df.empty:
        plot_wse_time_series(wse_df, lake_name)
    else:
        print(f"No data found for lake_name {lake_name}")

    # Save to CSV if needed
    wse_df.to_csv('wse_data.csv', index=False)
