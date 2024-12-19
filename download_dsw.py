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

def search_dswx_data(date_range,intersects_geometry,cloudy_threshold,collections):
    start, stop = date_range
    

    # Setup STAC API
    stac = 'https://cmr.earthdata.nasa.gov/cloudstac/'  # CMR-STAC API Endpoint
    api = Client.open(f'{stac}POCLOUD/')
    
    # Define search parameters
    search_params = {
        "collections": ps.collections,
        "intersects": intersects_geometry,
        "datetime": [start, stop],
        "max_items": 100000
    }
    
    print(f"Connecting to API for period {start} to {stop}...")
    search_dswx = api.search(**search_params)
    
    # Return the collected items
    # Filter cloudy days
    
    # print("Filtering cloudy days > " + str(cloudy_threshold) + "%...")
    
    # # Define the normal function
    # def cloud_cover_filter(item):
    #     return filter_by_cloud_cover(item, threshold=cloudy_threshold)
    
    # # Apply the filter using the defined function
    # filtered_items = list(filter(cloud_cover_filter, search_dswx.item_collection())) 
    

    
    # print(str(len(list(search_dswx.item_collection()))) + ' items were found.')
    # print(str(len(filtered_items)) + ' items meet the cloud threshold')
    
    # filtered_urls = list(map(return_granule, filtered_items))
    
    all_items = list(search_dswx.item_collection())
    all_urls = []
    for item in all_items:
        urls = return_granule(item)
        all_urls.extend(urls)
    
    return all_urls, list(search_dswx.items())


    
def searchDSWx(ps):
    '''
    Searches DSWx data given polygon AOI and date range. 
    splits the date range into 10 chunks, and accesses data in parallel
    Returns:
        filtered_urls: urls to the data from sparse-cloud days
        dswx_data_df: pandas dataframe for all search results
    '''
    
    
    already_dl_dates =[]# glob.glob('data/2???????')
    if len(already_dl_dates)>0:
        already_dl_dates.sort()
        # Extract the latest date from the already downloaded dates
        last_date_str = already_dl_dates[-1][-8:]  # Assumes format 'data/YYYYMMDD'
        last_date = datetime.strptime(last_date_str, '%Y%m%d')
        # Set the new start date to the day after the last downloaded date
        start_date = last_date + timedelta(days=1)
        # start_date = datetime(int(ps.date_start[0:4]), int(ps.date_start[4:6]), int(ps.date_start[6:8]))   
        print(f'already found the following dates { already_dl_dates}.  Setting new start date to {start_date}.')
    else:
        start_date = datetime(int(ps.date_start[0:4]), int(ps.date_start[4:6]), int(ps.date_start[6:8]))    
    
    # stop_date = datetime.today()  
    if ps.date_stop is None:
        stop_date = datetime.today()  
    else:
        stop_date = datetime(int(ps.date_stop[0:4]), int(ps.date_stop[4:6]), int(ps.date_stop[6:8]))    

    # Convert the wkt polygon to a Shapely Polygon
    aoi = wkt.loads(ps.polygon)
    intersects_geometry = aoi.__geo_interface__
    
    # Calculate the total number of days and the interval
    total_days = (stop_date - start_date).days
    num_workers = 10
    
    interval = total_days // num_workers
    if interval == 0:
        interval = 1  # Ensure at least one day is covered
    
    # Generate new start and stop dates for each search
    date_ranges = []
    for i in range(num_workers):
        new_start = start_date + timedelta(days=i * interval)
        # Ensure the last interval ends exactly on the stop_date
        if i == num_workers - 1:
            new_stop = stop_date
        else:
            new_stop = start_date + timedelta(days=(i + 1) * interval - 1)
        date_ranges.append((new_start, new_stop))
    
    # Display the new date ranges
    for start, stop in date_ranges:
        print("Start:", start.strftime("%Y-%m-%d"), "Stop:", stop.strftime("%Y-%m-%d"))
    
    
    filtered_urls_list = []
    dswx_data = []
    
    
    # Number of parallel workers (you can adjust this based on your needs and resources)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Adjust the executor.submit call to include the additional parameters
        future_to_date_range = {
            executor.submit(search_dswx_data, date_range, intersects_geometry, ps.cloudy_threshold,ps.collections): date_range 
            for date_range in date_ranges
        }
        
        # Process the results as they complete
        for future in as_completed(future_to_date_range):
            date_range = future_to_date_range[future]
            try:
                f_urls, data_list = future.result()
                filtered_urls_list.append(f_urls)
                dswx_data.append(data_list)
            except Exception as exc:
                print(f'{date_range} generated an exception: {exc}')
    

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

    # Convert the wkt polygon to a Shapely Polygon
    aoi = wkt.loads(ps.polygon)

    plot_frames(dswx_data,aoi)


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
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)

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
def plot_frames(dswx_data,aoi):
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
    zoomLevel = 10
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


def organize_directories(data_dir):
    tif_files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]

    # Extract unique dates from filenames, skipping files without dates
    unique_dates = set()
    for f in tif_files:
        date = extract_date_from_filename(f)
        if date:
            unique_dates.add(date)

    # Create directories for each unique date
    for date in unique_dates:
        os.makedirs(os.path.join(data_dir, date), exist_ok=True)

    # Move the .tif files to the respective directories
    for f in tif_files:
        date = extract_date_from_filename(f)
        if date:  # Only move files that have a valid date
            destination = os.path.join(data_dir, date, f)
            source = os.path.join(data_dir, f)
            os.rename(source, destination)


def main():

    ps = config.getPS()

    # Get the filtered list of urls
    filtered_urls,dswx_data_df = searchDSWx(ps)
    
    # Download the files
    dlDSWx(filtered_urls,ps,ps.dataDir)
    
    # Organize files into their respective dates directories
    organize_directories(ps.dataDir)

    return filtered_urls,dswx_data_df


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

    