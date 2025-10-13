import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point

def find_lakes_near_point(base_dir, latitude, longitude, max_distance=5000):
    """
    Search all SWOT LakeSP_Prior .shp files for lakes within a specified distance
    of the given latitude/longitude.
    
    Parameters:
    base_dir (str): Path to the base directory containing YYYYMMDD folders with shapefiles
    latitude (float): Latitude of the target location
    longitude (float): Longitude of the target location
    max_distance (float): Maximum distance in meters to search for lakes
    
    Returns:
    pd.DataFrame: DataFrame with date and WSE values for nearby lakes
    """
    # Create a target point from the input coordinates
    target_point = Point(longitude, latitude)
    
    # List to store results
    results = []
    
    # Walk through all folders in the base directory
    print(f"Searching for lakes within {max_distance/1000:.1f} km of coordinates ({latitude}, {longitude})")
    
    # Count the total number of folders for progress reporting
    date_folders = [f for f in Path(base_dir).iterdir() if f.is_dir() and f.name.isdigit() and len(f.name) == 8]
    total_folders = len(date_folders)
    print(f"Found {total_folders} date folders to process")
    
    # Process each date folder
    for i, date_folder in enumerate(sorted(date_folders)):
        date_str = date_folder.name  # YYYYMMDD format
        
        # Find all Prior shapefiles in this folder
        prior_files = list(date_folder.glob("*LakeSP_Prior*.shp"))
        
        if prior_files:
            # Print progress update every 10 folders
            if i % 10 == 0 or i == total_folders - 1:
                print(f"Processing folder {i+1}/{total_folders}: {date_str} ({len(prior_files)} files)")
            
            # Process each shapefile
            for shp_file in prior_files:
                try:
                    # Read the shapefile
                    gdf = gpd.read_file(shp_file)
                    
                    # Check if required columns exist
                    if 'wse' in gdf.columns and 'lake_name' in gdf.columns:
                        # Convert both GeoDataFrame and target point to the same CRS for distance calculation
                        if gdf.crs is not None:
                            # Create a GeoDataFrame with the target point
                            point_gdf = gpd.GeoDataFrame(geometry=[target_point], crs="EPSG:4326")
                            
                            # Ensure both are in the same projected CRS for accurate distance calculation
                            gdf_proj = gdf.to_crs(epsg=3857)  # Web Mercator projection
                            point_proj = point_gdf.to_crs(epsg=3857)
                            
                            # Calculate distances to the target point
                            gdf_proj['distance'] = gdf_proj.geometry.distance(point_proj.geometry.iloc[0])
                            
                            # Filter to lakes within max_distance
                            nearby_lakes = gdf_proj[gdf_proj['distance'] <= max_distance]
                            
                            # If any lakes are found within the distance
                            if not nearby_lakes.empty:
                                # Get the nearest lake
                                nearest_lake = nearby_lakes.loc[nearby_lakes['distance'].idxmin()]
                                
                                # Store the result
                                results.append({
                                    'date': pd.to_datetime(date_str, format='%Y%m%d'),
                                    'wse': nearest_lake['wse'],
                                    'lake_name': nearest_lake['lake_name'],
                                    'distance': nearest_lake['distance']
                                })
                                
                                # Optional: print when a lake is found
                                print(f"  Found lake '{nearest_lake['lake_name']}' at {nearest_lake['distance']:.0f}m")
                    
                except Exception as e:
                    print(f"Error processing {shp_file}: {e}")
        
    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        # Sort by date
        df = df.sort_values(by='date').reset_index(drop=True)
        return df
    else:
        print(f"No lakes found within {max_distance/1000:.1f} km of coordinates")
        return pd.DataFrame()

def plot_wse_time_series(df):
    """
    Plot the WSE time series.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing date and WSE values
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Get the most common lake name for the title
    if 'lake_name' in df.columns:
        lake_name = df['lake_name'].mode().iloc[0]
    else:
        lake_name = "Unknown"
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['wse'], marker='o', linestyle='-', color='b')
    plt.title(f'Water Surface Elevation Time Series for {lake_name}')
    plt.xlabel('Date')
    plt.ylabel('Water Surface Elevation (m)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add annotation with date range and points count
    min_date = df['date'].min().strftime('%Y-%m-%d')
    max_date = df['date'].max().strftime('%Y-%m-%d')
    plt.figtext(0.5, 0.01, f"Date Range: {min_date} to {max_date} ({len(df)} data points)", ha='center')
    
    plt.show()

def save_data_to_csv(df, lake_name=None):
    """
    Save the time series data to a CSV file.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data
    lake_name (str, optional): Name of the lake for the filename
    """
    if df.empty:
        return
    
    # Determine lake name for filename
    if lake_name is None and 'lake_name' in df.columns:
        lake_name = df['lake_name'].mode().iloc[0]
        # Simplify lake name for filename
        lake_name = lake_name.split(';')[0].split(',')[0].strip()
    elif lake_name is None:
        lake_name = "lake"
    
    # Create filename with date range
    start_date = df['date'].min().strftime('%Y%m%d')
    end_date = df['date'].max().strftime('%Y%m%d')
    csv_filename = f'wse_data_{lake_name}_{start_date}_to_{end_date}.csv'
    
    # Save to CSV
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

# Main script execution
if __name__ == "__main__":
    # Input parameters
    latitude = 34.579
    longitude = -119.948  # Example: Cachuma Lake
    max_distance = 5000  # 5 kilometers in meters
    base_dir = './data/swot/LakesData'  # Change this to your data directory
    
    # Find lakes near the specified coordinates
    wse_df = find_lakes_near_point(base_dir, latitude, longitude, max_distance)
    
    # Display the results
    if not wse_df.empty:
        print(f"\nFound data for {len(wse_df)} dates")
        print(wse_df.head())
        
        # Plot the time series
        plot_wse_time_series(wse_df)
        
        # Save data to CSV
        save_data_to_csv(wse_df)
    else:
        print("No data found for the specified location")