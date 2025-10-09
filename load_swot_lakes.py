'''
https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_LakeSP_2.0

a lakes file has the following columns:
    Index(['lake_id', 'reach_id', 'obs_id', 'overlap', 'n_overlap', 'time',
        'time_tai', 'time_str', 'wse', 'wse_u', 'wse_r_u', 'wse_std',
        'area_total', 'area_tot_u', 'area_detct', 'area_det_u', 'layovr_val',
        'xtrk_dist', 'ds1_l', 'ds1_l_u', 'ds1_q', 'ds1_q_u', 'ds2_l', 'ds2_l_u',
        'ds2_q', 'ds2_q_u', 'quality_f', 'dark_frac', 'ice_clim_f', 'ice_dyn_f',
        'partial_f', 'xovr_cal_q', 'geoid_hght', 'solid_tide', 'load_tidef',
        'load_tideg', 'pole_tide', 'dry_trop_c', 'wet_trop_c', 'iono_c',
        'xovr_cal_c', 'lake_name', 'p_res_id', 'p_lon', 'p_lat', 'p_ref_wse',
        'p_ref_area', 'p_date_t0', 'p_ds_t0', 'p_storage', 'geometry'],
        dtype='object')

Found 1 lakes with 'cachuma' in the name
['CACHUMA LAKE;LAKE CACHUMA']

you can find pass numbers form the kml in /d/surfaceWater/westCoastData/data/swot/swot_science_hr_2.0s_4.0s_Aug2021-v5_perPass.kml
'''

import geopandas as gpd
import pandas as pd
from pathlib import Path
import fiona, os
from multiprocessing import Pool
from matplotlib import pyplot as plt
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


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


def load_prior_lakes_shapefile(prior_shapefile_path):
    """
    Load a Prior lakes shapefile containing reference water body geometries.
    
    Parameters:
    prior_shapefile_path (str): Path to the Prior lakes shapefile.
    
    Returns:
    geopandas.GeoDataFrame: GeoDataFrame containing the Prior lakes data.
    """
    try:
        gdf = gpd.read_file(prior_shapefile_path)
        print(f"Loaded Prior lakes shapefile with {len(gdf)} features")
        print(f"CRS: {gdf.crs}")
        print(f"Columns: {gdf.columns.tolist()}")
        
        # Check if the necessary columns exist in the shapefile
        if 'lake_name' not in gdf.columns:
            print("Warning: 'lake_name' column not found in Prior shapefile.")
            # Try to identify an appropriate name column if it exists
            name_candidates = [col for col in gdf.columns if 'name' in col.lower()]
            if name_candidates:
                print(f"Using '{name_candidates[0]}' as lake name column")
                gdf = gdf.rename(columns={name_candidates[0]: 'lake_name'})
            else:
                print("No suitable name column found. Creating a default 'lake_name' column with index values.")
                gdf['lake_name'] = [f"Lake_{i}" for i in range(len(gdf))]
        
        return gdf
    except Exception as e:
        print(f"Error loading Prior lakes shapefile: {e}")
        return None

def find_lake_in_prior_shapefile(prior_lakes_gdf, latitude, longitude, max_distance=500):
    """
    Find a lake in the Prior shapefile that contains or is closest to the given coordinates.
    
    Parameters:
    prior_lakes_gdf (geopandas.GeoDataFrame): GeoDataFrame containing Prior lakes data.
    latitude (float): Latitude of the target location.
    longitude (float): Longitude of the target location.
    max_distance (float): Maximum allowable distance (in meters) from the target point.
    
    Returns:
    tuple: (lake_name, distance) if found, else (None, None)
    """
    if prior_lakes_gdf is None or prior_lakes_gdf.empty:
        return None, None
    
    # Create a Point from the coordinates
    point = Point(longitude, latitude)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    
    # Ensure both GeoDataFrames have the same CRS
    if prior_lakes_gdf.crs != "EPSG:4326":
        prior_lakes_gdf = prior_lakes_gdf.to_crs("EPSG:4326")
    
    # First check if the point is within any lake
    for idx, lake in prior_lakes_gdf.iterrows():
        if point_gdf.geometry.iloc[0].within(lake.geometry):
            return lake['lake_name'], 0
    
    # If not within any lake, find the closest lake within max_distance
    # Convert to projected CRS for accurate distance calculation
    prior_lakes_gdf_proj = prior_lakes_gdf.to_crs(epsg=3857)
    point_gdf_proj = point_gdf.to_crs(epsg=3857)
    
    # Calculate distances
    prior_lakes_gdf_proj['distance'] = prior_lakes_gdf_proj.geometry.distance(point_gdf_proj.geometry.iloc[0])
    
    # Find lakes within max_distance
    lakes_within_distance = prior_lakes_gdf_proj[prior_lakes_gdf_proj['distance'] <= max_distance]
    
    if not lakes_within_distance.empty:
        # Get the closest lake
        closest_lake = lakes_within_distance.iloc[lakes_within_distance['distance'].idxmin()]
        return closest_lake['lake_name'], closest_lake['distance']
    
    return None, None

def process_shapefile_for_nearest_lake(shp_file, target_point, date_str, max_distance, prior_lakes_gdf=None, preferred_lake_name=None, debug=False):
    """
    Process a single shapefile and find the WSE of the nearest lake to the given point, within a maximum distance.
    Optionally filter by a preferred lake name from Prior shapefile.
    
    Parameters:
    shp_file (str): Path to the shapefile.
    target_point (shapely.geometry.Point): The point (latitude/longitude) to find the nearest lake to.
    date_str (str): The date string corresponding to the folder name (YYYYMMDD).
    max_distance (float): Maximum allowable distance (in meters) from the target point.
    prior_lakes_gdf (geopandas.GeoDataFrame, optional): GeoDataFrame containing Prior lakes data.
    preferred_lake_name (str, optional): Name of the preferred lake from Prior shapefile.
    debug (bool): Whether to print debug information.

    Returns:
    dict: A dictionary containing the date, WSE value, and lake name for the nearest lake, or None if not found.
    """
    try:
        # Load the shapefile into a GeoDataFrame
        gdf = gpd.read_file(shp_file)

        if debug:
            print(f"\nProcessing {shp_file}")
            print(f"File contains {len(gdf)} features")
            if 'lake_name' in gdf.columns:
                print(f"Lake names in file: {', '.join(gdf['lake_name'].unique())}")
            
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
            
            if debug and not gdf_within_distance.empty:
                print(f"Lakes within {max_distance}m: {len(gdf_within_distance)}")
                print(f"Names: {', '.join(gdf_within_distance['lake_name'].unique())}")
                print(f"Distances: {', '.join(map(str, gdf_within_distance['distance'].values))}")
            
            # If preferred_lake_name is provided, use more flexible matching
            if preferred_lake_name is not None and not gdf_within_distance.empty:
                # Check for exact match
                gdf_preferred = gdf_within_distance[gdf_within_distance['lake_name'].str.lower() == preferred_lake_name.lower()]
                
                # If no exact match, try partial match
                if gdf_preferred.empty:
                    gdf_preferred = gdf_within_distance[gdf_within_distance['lake_name'].str.lower().str.contains(preferred_lake_name.lower())]
                
                # If we found matches with the preferred name
                if not gdf_preferred.empty:
                    if debug:
                        print(f"Found preferred lake: {gdf_preferred['lake_name'].iloc[0]}")
                    gdf_within_distance = gdf_preferred

            # If no lakes are within the max_distance, return None
            if gdf_within_distance.empty:
                if debug:
                    print(f"No lakes found within {max_distance}m for {date_str}")
                return None

            # Find the nearest lake (the one with the smallest distance)
            nearest_lake = gdf_within_distance.loc[gdf_within_distance['distance'].idxmin()]

            # Extract the WSE value and lake name of the nearest lake
            wse_value = nearest_lake['wse']
            lake_name = nearest_lake['lake_name']
            
            if debug:
                print(f"Selected lake: {lake_name}, WSE: {wse_value}, Distance: {nearest_lake['distance']}")
                
            return {'date': pd.to_datetime(date_str, format='%Y%m%d'), 'wse': wse_value, 'lake_name': lake_name}
        else:
            if debug:
                missing_cols = []
                if 'wse' not in gdf.columns:
                    missing_cols.append('wse')
                if 'lake_name' not in gdf.columns:
                    missing_cols.append('lake_name')
                if not gdf.geometry.notnull().all():
                    missing_cols.append('valid geometry')
                print(f"File missing required data: {', '.join(missing_cols)}")
                if 'lake_name' in gdf.columns:
                    print(f"Lake names in file: {gdf['lake_name'].unique()}")

    except Exception as e:
        print(f"Error processing {shp_file}: {e}")

    return None

def get_lake_wse_time_series_by_location(base_dir, latitude, longitude, max_distance, num_processes=None, prior_shapefile_path=None, debug=False):
    """
    Create a time series of WSE for the lake nearest to a given latitude/longitude point
    from shapefiles using parallel processing and selective column loading, within a maximum distance.
    Prioritizes files containing 'pass_num' in the filename which are known to cover the area of interest.
    
    Parameters:
    base_dir (str): Path to the base directory containing YYYYMMDD folders with shapefiles.
    latitude (float): Latitude of the target location.
    longitude (float): Longitude of the target location.
    max_distance (float): Maximum allowable distance (in meters) from the target point.
    num_processes (int): The number of parallel processes to use. If None, it will use the number of CPU cores available.
    prior_shapefile_path (str, optional): Path to the Prior lakes shapefile.
    debug (bool): Whether to print debug information.
    
    Returns:
    pd.DataFrame: DataFrame containing the date, WSE, and lake name for the nearest lake.
    """
    start_time = time.time()
    tasks = []
    target_point = Point(longitude, latitude)  # Create a shapely Point from lat/lon

    # Check for the cached file with matching pattern from lake_name
    if prior_shapefile_path:
        cache_file = None
        try:
            from pathlib import Path
            import glob
            # Try to find a cached file with similar name
            cache_pattern = f"wse_data*{latitude}*{longitude}*.csv"
            matching_files = glob.glob(cache_pattern)
            if matching_files:
                cache_file = matching_files[0]
                print(f"Found cached file {cache_file}, loading data...")
                cached_df = pd.read_csv(cache_file)
                if not cached_df.empty:
                    # Convert date column to datetime
                    cached_df['date'] = pd.to_datetime(cached_df['date'])
                    print(f"Loaded cached data with {len(cached_df)} entries from {cached_df['date'].min()} to {cached_df['date'].max()}")
                    # If cache has data, just return it
                    return cached_df
        except Exception as e:
            print(f"Error checking for cached data: {e}")
            cache_file = None

    # Load Prior lakes shapefile if provided
    prior_lakes_gdf = None
    preferred_lake_name = None
    if prior_shapefile_path and os.path.exists(prior_shapefile_path):
        prior_lakes_gdf = load_prior_lakes_shapefile(prior_shapefile_path)
        if prior_lakes_gdf is not None:
            # Find if the point is in or near a lake in the Prior shapefile
            preferred_lake_name, distance = find_lake_in_prior_shapefile(prior_lakes_gdf, latitude, longitude, max_distance)
            if preferred_lake_name:
                print(f"Found point in/near lake '{preferred_lake_name}' in Prior shapefile (distance: {distance:.2f} meters)")
                # Extract a simple name for searching (remove extra text after semicolons, commas, etc.)
                simple_name = preferred_lake_name.split(';')[0].split(',')[0].strip()
                if simple_name.lower() != preferred_lake_name.lower():
                    print(f"Will search for lake with simplified name: '{simple_name}'")
                    preferred_lake_name = simple_name
            else:
                print(f"Point ({latitude}, {longitude}) not found in any lake in Prior shapefile within {max_distance} meters")

    # If num_processes is not specified, use os.cpu_count() to get the default
    if num_processes is None:
        num_processes = os.cpu_count()
    # Use more processes for better parallelization
    num_processes = min(os.cpu_count() * 2, 32)  # Cap at 32 processes to avoid system overload

    # Print the number of processes being used
    print(f"Using {num_processes} processes for parallel execution.")

    # List all folders in base_dir
    all_date_folders = []
    for item in Path(base_dir).iterdir():
        if item.is_dir():
            try:
                # Attempt to parse the folder name as a date
                folder_name = item.name
                # Check if folder name matches YYYYMMDD pattern
                if len(folder_name) == 8 and folder_name.isdigit():
                    all_date_folders.append(item)
            except Exception as e:
                print(f"Skipping folder {item.name}: {e}")

    # Sort the folders by name (chronological order)
    all_date_folders.sort()
    
    # Print the date range being processed
    if all_date_folders:
        print(f"Processing all available date folders: {all_date_folders[0].name} to {all_date_folders[-1].name}")
        print(f"Total date folders found: {len(all_date_folders)}")
    else:
        print("No date folders found matching the criteria")
        return pd.DataFrame()

    # Iterate through each date folder and collect files with 'pass_num' in filename first
    for date_folder in all_date_folders:
        date_str = date_folder.name  # Directory name is YYYYMMDD
        
        # Look for files containing 'pass_num' in the filename
        targeted_shapefiles = list(date_folder.glob(f"*{pass_num}*.shp"))
        
        if targeted_shapefiles:
            print(f"Found {len(targeted_shapefiles)} files with 'pass_num' in folder {date_str}")
            for shp_file in targeted_shapefiles:
                tasks.append((shp_file, target_point, date_str, max_distance, prior_lakes_gdf, preferred_lake_name, debug))
        else:
            # If no targeted files, use a more general approach as fallback
            print(f"No files with 'pass_num' found in {date_str}, looking for any applicable files")
            
            # Look for Obs shapefiles as the next best option
            obs_shapefiles = list(date_folder.glob("*Obs*.shp"))
            if obs_shapefiles:
                print(f"Using {len(obs_shapefiles)} Obs files as fallback in folder {date_str}")
                for shp_file in obs_shapefiles:
                    tasks.append((shp_file, target_point, date_str, max_distance, prior_lakes_gdf, preferred_lake_name, debug))
            else:
                # If no Obs files either, try Prior files as a last resort
                prior_shapefiles = list(date_folder.glob("*Prior*.shp"))
                if prior_shapefiles:
                    print(f"Using {len(prior_shapefiles)} Prior files as last resort in folder {date_str}")
                    for shp_file in prior_shapefiles:
                        tasks.append((shp_file, target_point, date_str, max_distance, prior_lakes_gdf, preferred_lake_name, debug))
                else:
                    print(f"No usable shapefiles found in folder {date_str}")

    print(f"Collected {len(tasks)} files to process")
    
    # Use ProcessPoolExecutor for better control over parallel execution
    results = []
    
    # If debug mode is on, process first few files sequentially for better debugging
    if debug and tasks:
        print("\n===== DEBUG MODE: Processing first 5 files sequentially =====")
        sample_results = []
        for task in tasks[:5]:  # Process first 5 files
            result = process_shapefile_for_nearest_lake(*task)
            if result:
                sample_results.append(result)
        print(f"Debug sample processing found {len(sample_results)} results")
        for result in sample_results:
            print(f"  {result['date'].strftime('%Y-%m-%d')}: {result['lake_name']} - WSE: {result['wse']}")
        print("===== Continuing with parallel processing =====\n")
    
    # Process all files in parallel batches for better memory management
    batch_size = 100  # Process files in batches of 100
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1} ({len(batch_tasks)} files)")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_task = {executor.submit(process_shapefile_for_nearest_lake, *task): task for task in batch_tasks}
            
            for future in as_completed(future_to_task):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Show progress after each batch
        print(f"Progress: {min(i+batch_size, len(tasks))}/{len(tasks)} files processed, found {len(results)} valid results so far")

    # Check if results is empty before creating a DataFrame
    if not results:
        print(f"No data found for the location ({latitude}, {longitude}) within {max_distance} meters.")
        
        # If debug is enabled, test a larger search radius
        if debug:
            print("DEBUG: Testing with larger search radius (2000m)...")
            larger_radius = 2000  # meters
            test_tasks = [(task[0], task[1], task[2], larger_radius, task[4], task[5], True) for task in tasks[:10]]
            test_results = []
            for task in test_tasks:
                result = process_shapefile_for_nearest_lake(*task)
                if result:
                    test_results.append(result)
            if test_results:
                print(f"Found {len(test_results)} results with larger radius. Consider increasing max_distance.")
                for result in test_results:
                    print(f"  {result['date'].strftime('%Y-%m-%d')}: {result['lake_name']} - WSE: {result['wse']}")
        
        return pd.DataFrame()  # Return an empty DataFrame

    # Create a DataFrame from the collected data
    df = pd.DataFrame(results)

    # Sort by date to ensure the time series is in order
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Print summary of results
    print(f"Found data for {len(df)} dates between {df['date'].min().strftime('%Y-%m-%d')} and {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
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

# Example usage - Main script
# if __name__ == "__main__":

# latitude, longitude = 34.579, -119.948  # Cachuma latitude, longitude
longitude,latitude = -120.40729,41.94765
pass_num = 552
num_processes = 18  # Specify the number of parallel processes, or leave it None for automatic
debug_mode = True

base_dir = './data/swot/LakesData'  # Base directory containing YYYYMMDD folders
output_shp_fn = './data/swot/lakes_polygons.shp'
prior_shapefile_path = './data/swot/prior_lakes.shp'  # Path to the Prior lakes shapefile


if not os.path.isfile('lake_names.txt'):
    lake_names = get_unique_lake_polygons_parallel(base_dir, output_shp_fn)
else:
    with open('lake_names.txt', 'r') as file:
        lake_names = [line.strip() for line in file.readlines()]  # Read each line, stripping newline characters

# Search lake names for 'Cachuma'  not case sensitive
cachuma_lakes = [name for name in lake_names if 'Cachuma' in name.lower()]
print(f"Found {len(cachuma_lakes)} lakes with 'Cachuma' in the name")
print(cachuma_lakes)


# Increase max_distance to find lakes that might be farther away
max_dist = 1000  # Meters (increased from 500m)

# # Load example file to dataframe
# example_file = '/d/surfaceWater/westCoastData/data/swot/LakesData/20250223/SWOT_L2_HR_LakeSP_Prior_028_552_NA_20250223T062617_20250223T063405_PIC2_01.shp'
# example_df = gpd.read_file(example_file)
# print(example_df.columns)
# # Get lake names from example (lake_name)
# lake_names = example_df['lake_name'].unique()
# print(f"Found {len(lake_names)} unique lake names in the example file")
# print(lake_names)
# # See if the word 'cachuma' is in any of the lake names
# cachuma_lakes = [name for name in lake_names if 'cachuma' in name.lower()]
# print(f"Found {len(cachuma_lakes)} lakes with 'cachuma' in the name")
# print(cachuma_lakes)

# # another example file
# example_file2 = '/d/surfaceWater/westCoastData/data/swot/LakesData/old/20240224/SWOT_L2_HR_LakeSP_Prior_011_261_NA_20240224T043639_20240224T044337_PIC0_01.shp'
# example_df2 = gpd.read_file(example_file2)
# print(example_df2.columns)
# # Get lake names from example (lake_name)
# lake_names2 = example_df2['lake_name'].unique()
# print(f"Found {len(lake_names2)} unique lake names in the example file")
# print(lake_names2)
# # See if the word 'cachuma' is in any of the lake names
# cachuma_lakes2 = [name for name in lake_names2 if 'cachuma' in name.lower()]
# print(f"Found {len(cachuma_lakes2)} lakes with 'cachuma' in the name")
# print(cachuma_lakes2)

# # in example 2, find the lake closest to latitude, longitude
# example_point = Point(longitude, latitude)
# example_df2['distance'] = example_df2.geometry.distance(example_point)
# closest_lake = example_df2.loc[example_df2['distance'].idxmin()]
# print(f"Closest lake to ({latitude}, {longitude}) is {closest_lake['lake_name']} at a distance of {closest_lake['distance']:.2f} meters")


# Get the WSE time series for the water body near the specified lat/lon
wse_df = get_lake_wse_time_series_by_location(
    base_dir, 
    latitude, 
    longitude, 
    max_dist, 
    num_processes=num_processes,
    prior_shapefile_path=prior_shapefile_path,
    debug=debug_mode
)

# Get lake name from wse_df if it exists, otherwise use default name
if not wse_df.empty and 'lake_name' in wse_df.columns:
    lake_name = wse_df['lake_name'].iloc[0]
    # If lake name is too long or has multiple variants, simplify it
    if ';' in lake_name or ',' in lake_name:
        simple_name = lake_name.split(';')[0].split(',')[0].strip()
        print(f"Using simplified lake name: '{simple_name}' instead of '{lake_name}'")
        lake_name = simple_name
else:
    lake_name = 'Cachuma'  # Default lake name

# Plot the time series if data is available
if not wse_df.empty:
    plt.figure(figsize=(12, 6))
    plt.plot(wse_df['date'], wse_df['wse'], marker='o', linestyle='-', color='b')
    plt.title(f'WSE Time Series for {lake_name}')
    plt.xlabel('Date')
    plt.ylabel('Water Surface Elevation (WSE)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add date range annotation
    min_date = wse_df['date'].min().strftime('%Y-%m-%d')
    max_date = wse_df['date'].max().strftime('%Y-%m-%d')
    plt.figtext(0.5, 0.01, f"Date Range: {min_date} to {max_date} ({len(wse_df)} data points)", ha='center')
    
    plt.show()
    print(wse_df)
else:
    print(f"No data found for lake at coordinates ({latitude}, {longitude})")

# Save to CSV with date range info in filename
if not wse_df.empty:
    start_date = wse_df['date'].min().strftime('%Y%m%d')
    end_date = wse_df['date'].max().strftime('%Y%m%d')
    csv_filename = f'wse_data_{lake_name}_{start_date}_to_{end_date}.csv'
    wse_df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")
