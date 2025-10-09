#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 21 16:21:35 2023

Download DSWx data

@author: km
"""

from pystac_client import Client  
from shapely import wkt
import os, json, requests, re, glob
from datetime import datetime, timedelta
from swampy import config
from urllib.request import urlopen
import concurrent.futures
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor, as_completed


nproc = int(os.cpu_count()-1)

def filter_by_cloud_cover(item, threshold=10):
    '''
    Removes images with more than <threshold>% cloud cover.
    '''
    xml_url = item.assets['metadata'].href
    try:
        response = urlopen(xml_url)
        data_json = json.loads(response.read())
    except URLError:
        print("Error fetching URL:", xml_url)
        return False

    try:
        c_cover = data_json['CloudCover']
    except KeyError:
        c_cover = data_json.get('eo:cloud_cover')

    return c_cover <= threshold


def return_granule(item):
    """Return WTR, WTR-2, and CONF asset URLs for the item with their asset types"""
    url_info = []
    
    # Define correct band numbers for each asset type
    wtr_asset = '0_B01_WTR'
    wtr2_asset = '0_B06_WTR-2'  # Changed from B01 to B06
    conf_asset = '0_B03_CONF'    # Changed from B01 to B03
    
    if wtr_asset in item.assets:
        wtr_url = item.assets[wtr_asset].href
        url_info.append((wtr_url, wtr_asset))
        
        # Get WTR-2 and CONF with correct band numbers
        if wtr2_asset in item.assets:
            url_info.append((item.assets[wtr2_asset].href, wtr2_asset))
        if conf_asset in item.assets:
            url_info.append((item.assets[conf_asset].href, conf_asset))
    
    return url_info

def check_file_exists(filename, data_dir):
    """
    Check if a specific DSWx file exists in any EPSG subdirectory of any date directory
    """
    # Extract date from filename (assuming format contains YYYYMMDD)
    date_match = re.search(r'(\d{8})', filename)
    if not date_match:
        return False
    
    date = date_match.group(1)
    date_dir = os.path.join(data_dir, date)
    
    if not os.path.exists(date_dir):
        return False
    
    # Look in all EPSG subdirectories
    epsg_dirs = glob.glob(os.path.join(date_dir, 'EPSG:*'))
    for epsg_dir in epsg_dirs:
        if os.path.exists(os.path.join(epsg_dir, filename)):
            return True
    
    return False
# ps = config.getPS()


# collections = ps.collections

def search_dswx_data(date_range, intersects_geometry, cloudy_threshold, collections):
    start, stop = date_range
    date_range = f"{start:%Y-%m-%d}/{stop:%Y-%m-%d}"

    stac_url = "https://cmr.earthdata.nasa.gov/cloudstac/POCLOUD"
    client = Client.open(stac_url)
    
    max_retries = 3
    retry_delay = 5  # seconds
    success = False
    
    for attempt in range(max_retries):
        try:
            # Use intersects parameter with the GeoJSON geometry
            search = client.search(
                collections=collections,
                intersects=intersects_geometry,  # Use the original polygon geometry
                datetime=date_range,
                max_items=100000
            )
            stac_results = list(search.get_items())
            print(f"Number of STAC results: {len(stac_results)}")
            success = True
            break
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    if not success:
        print(f"All retries failed for date range {date_range}")
        return [], []
    
    all_urls = []
    for item in stac_results:
        urls = return_granule(item)
        all_urls.extend(urls)
    
    print(f"Number of URLs found: {len(all_urls)}")
    return all_urls, stac_results


    
def searchDSWx(ps):
    '''
    Searches DSWx data given polygon AOI and date range. 
    splits the date range into 10 chunks, and accesses data in parallel
    Returns:
        filtered_urls: urls to the data from sparse-cloud days
        dswx_data_df: pandas dataframe for all search results
    '''
    
    already_dl_dates = []  # glob.glob('data/2???????')
    if len(already_dl_dates) > 0:
        already_dl_dates.sort()
        last_date_str = already_dl_dates[-1][-8:]
        last_date = datetime.strptime(last_date_str, '%Y%m%d')
        start_date = last_date + timedelta(days=1)
        print(f'already found the following dates {already_dl_dates}. Setting new start date to {start_date}.')
    else:
        start_date = datetime(int(ps.date_start[0:4]), int(ps.date_start[4:6]), int(ps.date_start[6:8]))    
    
    if ps.date_stop is None:
        stop_date = datetime.today()  
    else:
        # Convert string to datetime
        stop_date = datetime(int(ps.date_stop[0:4]), int(ps.date_stop[4:6]), int(ps.date_stop[6:8]))
        # Check if stop_date is in the future, if yes, use today's date
        today = datetime.today()
        if stop_date > today:
            print(f"Date stop {stop_date.strftime('%Y-%m-%d')} is in the future. Setting to today's date: {today.strftime('%Y-%m-%d')}")
            stop_date = today

    # Load polygon from map.wkt file
    with open('map.wkt', 'r') as f:
        polygon_wkt = f.read().strip()
    aoi = wkt.loads(polygon_wkt)
    
    # Use the original polygon geometry for search
    intersects_geometry = aoi.__geo_interface__
    
    print(f"Using original polygon geometry for search")
    
    # Calculate the total number of days and the interval
    total_days = (stop_date - start_date).days
    num_workers = 10
    
    interval = max(1, total_days // num_workers)
    
    # Generate date ranges as tuples of datetime objects
    date_ranges = []
    current_date = start_date
    while current_date < stop_date:
        range_end = min(current_date + timedelta(days=interval), stop_date)
        date_ranges.append((current_date, range_end))
        current_date = range_end + timedelta(days=1)
    
    # Display the date ranges and ensure no gaps
    prev_date = None
    for start, stop in date_ranges:
        if prev_date is not None and start != prev_date + timedelta(days=1):
            print(f"WARNING: Gap between {prev_date.strftime('%Y-%m-%d')} and {start.strftime('%Y-%m-%d')}")
        print("Start:", start.strftime("%Y-%m-%d"), "Stop:", stop.strftime("%Y-%m-%d"))
        prev_date = stop

    filtered_urls_list = []
    dswx_data = []
    
    print("\nDEBUG - searchDSWx:")
    print(f"Number of date ranges to process: {len(date_ranges)}")
    
    # Ensure date ranges don't overlap and cover full period
    total_days_covered = sum((stop - start).days + 1 for start, stop in date_ranges)
    expected_days = (stop_date - start_date).days + 1
    if total_days_covered != expected_days:
        print(f"WARNING: Date ranges cover {total_days_covered} days but should cover {expected_days} days")
    

    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_date_range = {
            executor.submit(
                search_dswx_data, 
                (start, stop),
                intersects_geometry, 
                ps.cloudy_threshold,
                ps.collections
            ): (start, stop) 
            for start, stop in date_ranges
        }
        
        # Process the results as they complete
        for future in as_completed(future_to_date_range):
            date_range = future_to_date_range[future]
            try:
                f_urls, data_list = future.result()
                print(f"DEBUG - Results for {date_range}:")
                print(f"Number of URLs: {len(f_urls)}")
                print(f"Number of data items: {len(data_list)}")
                filtered_urls_list.append(f_urls)
                dswx_data.append(data_list)
            except Exception as exc:
                print(f'ERROR - {date_range} generated an exception: {exc}')

    # Create a new list to hold all items from all collections
    filtered_urls = []
    # Iterate over each ItemCollection and extend the all_items list with its items
    for ful in filtered_urls_list:
        for f in ful:
            filtered_urls.append(f)

    dswx_data_a = []
    for dsw in dswx_data:
        for ds in dsw:
            dswx_data_a.append(ds)
    dswx_data = dswx_data_a

    if dswx_data:  # Only plot if there's data
        plot_frames(dswx_data, aoi)


    print(len(filtered_urls))
    
    # Create table of search results
    dswx_data_df = []
    for item in dswx_data:
        item.to_dict()
        fn = item.id.split('_')
        ID = fn[3]
        sensor = fn[6]
        dat = item.datetime.strftime('%Y-%m-%d')
        # spatial_coverage = intersection_percent(item, intersects_geometry)
        geom = item.geometry

        # Take all the band href information 
        band_links = [item.assets[links].href for links in item.assets.keys()]
        dswx_data_df.append([ID,sensor,dat,geom,band_links])

    dswx_data_df = pd.DataFrame(dswx_data_df, columns = ['TileID', 'Sensor', 'Date', 'Footprint','BandLinks'])

    # Save the results:
    dswx_data_df.to_csv('searchResults.csv',index=False)
    with open('filteredURLs.txt', 'w') as file:
        for item in filtered_urls:
            file.write(f"{item}\n")
        
    return filtered_urls, dswx_data_df


def dl(url,outname):   
    response = requests.get(url,stream=True,allow_redirects=True)
    
    # Open the local file for writing
    with open(outname, 'wb') as file:
        # Iterate through the content and write to the file
        for data in response.iter_content(chunk_size=int(2**14)):
            file.write(data)

            
def dlDSWx(urls, ps, dataDir): 
    '''
    Download the DSWx files given urls, only if all three asset types don't exist
    '''
    # Set the data directory for DSW data
    dataDir = './data/DSW'
    
    # Create directory if it doesn't exist
    if not os.path.isdir(dataDir):
        os.makedirs(dataDir, exist_ok=True)
        print(f"Created directory: {dataDir}")

    # Group URLs by their base filename (without asset type and band number)
    grouped_files = {}
    for url_tuple in urls:
        url, asset_type = url_tuple
        file_name = url.split('/')[-1]
        # Split the filename and remove the last two parts (B0X_asset-type.tif)
        name_parts = file_name.split('_')
        base_key = '_'.join(name_parts[:-2])  # Remove band number and asset type
        
        if base_key not in grouped_files:
            grouped_files[base_key] = []
        grouped_files[base_key].append((url, asset_type))

    dl_list = []
    outNames = []
    
    # Debug prints
    print(f"Number of grouped files: {len(grouped_files)}")
    
    # Check each group of files
    for base_key, file_group in grouped_files.items():
        # Debug prints
        print(f"\nChecking group: {base_key}")
        print(f"Number of files in group: {len(file_group)}")
        print(f"Asset types in group: {set(asset[1] for asset in file_group)}")
        
        # Only process groups that have all three asset types
        asset_types = set(asset[1] for asset in file_group)
        if len(asset_types) == 3:  # We have all three types
            print("Found group with all three asset types")
            # Check if any of the three files are missing
            missing_files = []
            for url, asset_type in file_group:
                file_name = url.split('/')[-1]
                exists = check_file_exists(file_name, dataDir)
                print(f"Checking {file_name}: {'exists' if exists else 'missing'}")
                if not exists:
                    missing_files.append((url, os.path.join(dataDir, file_name)))
            
            print(f"Number of missing files: {len(missing_files)}")
            # If any file is missing from the group, download all three
            if missing_files:
                for url, outName in missing_files:
                    dl_list.append(url)
                    outNames.append(outName)

    print(f"\nFinal download list size: {len(dl_list)}")

    if dl_list:
        print('Downloading the following files:')
        print(dl_list)
        print('downloading with ' + str(nproc) + ' cpus')

        with concurrent.futures.ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(dl, url, outName) for url, outName in zip(dl_list, outNames)]
            concurrent.futures.wait(futures)

    # check the files
    for url_tuple in urls:
        url, _ = url_tuple
        fname = os.path.join(dataDir, url.split('/')[-1])
        if os.path.isfile(fname):
            if os.path.getsize(fname) < 2**10:  # If it's smaller than 1 Kb
                print('Warning: ' + fname + ' is too small. Try again.')
                try:
                    os.remove(fname)
                except OSError:
                    print(f"Could not remove corrupted file: {fname}")
def plot_frames(dswx_data, aoi):
    '''
    Plot the footprints from the DSWx search results
    '''
    # Visualize the DSWx tile boundary and the user-defined bbox
    geom_df = []
    for d,_ in enumerate(dswx_data):
        geom_df.append(shape(dswx_data[d].geometry))
    geom_granules = gpd.GeoDataFrame({'geometry':geom_df})
    
    minlon = np.floor(min(coord[0] for coord in aoi.__geo_interface__['coordinates'][0]))
    maxlon = np.ceil(max(coord[0] for coord in aoi.__geo_interface__['coordinates'][0]))
    minlat = np.floor(min(coord[1] for coord in aoi.__geo_interface__['coordinates'][0]))
    maxlat = np.ceil(max(coord[1] for coord in aoi.__geo_interface__['coordinates'][0]))
    
    # Set up the figure and axis with a chosen map projection (e.g., PlateCarree)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([minlon, maxlon, minlat, maxlat])  # Set extent to fit around your data's bounding box
    
    # Add a basemap using cartopy features
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':',linewidth=.3)
   
    import cartopy.io.img_tiles as cimgt
   
    bg='World_Shaded_Relief'
    zoomLevel = 8
    url = 'https://server.arcgisonline.com/ArcGIS/rest/services/' + bg + '/MapServer/tile/{z}/{y}/{x}.jpg'
    image = cimgt.GoogleTiles(url=url)
    ax.add_image(image, zoomLevel,zorder=1) #zoom level
    # Plot the DSWx tile boundary polygons
    geom_granules.boundary.plot(ax=ax, color='green', linewidth=1)
    
    aoi_series = gpd.GeoSeries([aoi])

    # Plot the user-specified AOI polygon
    aoi_series.boundary.plot(ax=ax, color='#8B0000', linewidth=1,linestyle='--')
    
    plt.title("DSWx Tile Boundary and User-specified AOI")
    plt.show()
    

def extract_date_from_filename(filename):
    """Extract the date in YYYYMMDD format from the filename."""
    match = re.search(r'(\d{8})', filename)
    return match.group(1) if match else None

#
def organize_directories(data_dir):
    """Organize .tif files into date-based directories."""
    # Get all tif files and their dates in a single pass
    moves = []
    for f in os.listdir(data_dir):
        if not f.endswith('.tif'):
            continue
            
        date = re.search(r'(\d{8})', f)
        if not date:
            continue
            
        date_str = date.group(1)
        date_dir = os.path.join(data_dir, date_str)
        
        # Create directory if it doesn't exist
        os.makedirs(date_dir, exist_ok=True)
        
        # Store move operation
        source = os.path.join(data_dir, f)
        dest = os.path.join(date_dir, f)
        moves.append((source, dest))
    
    # Perform all moves in batch
    for source, dest in moves:
        os.rename(source, dest)


def main():

    ps = config.getPS()
    
    # Set the data directory for DSW data
    ps.dataDir = './data/DSW'
    
    # Create directory if it doesn't exist
    if not os.path.isdir(ps.dataDir):
        os.makedirs(ps.dataDir, exist_ok=True)
        print(f"Created directory: {ps.dataDir}")

    # Get the filtered list of urls
    filtered_urls, dswx_data_df = searchDSWx(ps)
    
    # Download the files
    dlDSWx(filtered_urls, ps, ps.dataDir)
    
    # Organize files into their respective dates directories
    organize_directories(ps.dataDir)

    return filtered_urls, dswx_data_df


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

    