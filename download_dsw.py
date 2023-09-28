#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 21 16:21:35 2023

Download DSWx data

@author: km
"""
from pystac_client import Client  
from shapely import wkt
import os
from datetime import datetime
from swampy import config
import json
from urllib.request import urlopen
import requests
import concurrent.futures
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
nproc = int(os.cpu_count())

def filter_by_cloud_cover(item, threshold=10):
    '''
    Removes images with more than <threshold>% cloud cover.
    '''
    xml_url = item.assets['metadata'].href
    response = urlopen(xml_url)
    data_json = json.loads(response.read()) # the XML files are formatted as JSONs (?!), so we use a JSON reader

    for item in data_json['AdditionalAttributes']:
        if item['Name'] == 'PercentCloudCover':
            break
    c_cover = int(item['Values'][0])
    if c_cover<=threshold:
        return True
    else:
        return False


def return_granule(item):
    return item.assets['0_B01_WTR'].href
    
    
def searchDSWx(ps):
    '''
    Searches DSWx data given polygon AOI and date range. 
    Returns:
        filtered_urls: urls to the data from sparse-cloud days
        dswx_data_df: pandas dataframe for all search results
    '''

    # convert to the datetime format
    start_date = datetime(int(ps.date_start[0:4]), int(ps.date_start[4:6]), int(ps.date_start[6:8]))    
    stop_date = datetime(int(ps.date_stop[0:4]), int(ps.date_stop[4:6]), int(ps.date_stop[6:8]))  
            
    # Convert the wkt polygon to a Shapely Polygon
    aoi = wkt.loads(ps.polygon)
    intersects_geometry = aoi.__geo_interface__
    
    # Search data through CMR-STAC API
    stac = 'https://cmr.earthdata.nasa.gov/cloudstac/'    # CMR-STAC API Endpoint
    print("Connecting to API...")
    api = Client.open(f'{stac}POCLOUD/')
    collections = ['OPERA_L3_DSWX-HLS_PROVISIONAL_V1']
    
    search_params = {"collections": collections,
                     "intersects": intersects_geometry,
                     "datetime": [start_date, stop_date],
                     "max_items": 10000}
    search_dswx = api.search(**search_params)
    items = search_dswx.get_all_items()
    dswx_data =list(search_dswx.items())
    plot_frames(dswx_data,aoi)

    # Filter cloudy days
    print("Filtering cloudy days...")
    filtered_items = list(filter(filter_by_cloud_cover, items))
    print(str(len(list(items))) + ' items were found.')
    print(str(len(filtered_items)) + ' items meet the cloud threshold')

    filtered_urls = list(map(return_granule, filtered_items))
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

            
def dlDSWx(urls,ps,dataDir): 
    '''
    Download the DSWx files given urls
    '''
    # Create a list of file path/names
    outNames = []
    dl_list = []
    # dataDir2 = './westCoastData'
    
    # for url in urls:
    #     fname2 = os.path.join(dataDir2, url.split('/')[-1])
    #     if os.path.isfile(fname2):
    #         os.system('ln -s ' + fname2 + ' dataDir/')
    
    for url in urls:
        fname = os.path.join(dataDir, url.split('/')[-1])

        if not os.path.isfile(fname):
            outNames.append(os.path.join(dataDir, url.split('/')[-1]))
            dl_list.append(url)
            
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
        
        
    print('Downloading the following files:')
    print(dl_list)
    print('downloading with '  + str(nproc) + ' cpus')

    with concurrent.futures.ThreadPoolExecutor(max_workers=nproc) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(dl, url, outName) for url, outName in zip(dl_list, outNames)]
        concurrent.futures.wait(futures)

    # check the files
    for url in urls:
        fname = os.path.join(dataDir, url.split('/')[-1])
        if not os.path.isfile(fname):
            print('Warning: File does not exist ' + fname)
        else:
            if os.path.getsize(fname) < 2**10: # If it's smaller than 1 Kb
                print('Warning: ' + fname + ' is too small. Try again.')
                

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
    zoomLevel = 12
    url = 'https://server.arcgisonline.com/ArcGIS/rest/services/' + bg + '/MapServer/tile/{z}/{y}/{x}.jpg'
    image = cimgt.GoogleTiles(url=url)
    ax.add_image(image, zoomLevel,zorder=1) #zoom level
    # Plot the DSWx tile boundary polygons
    geom_granules.boundary.plot(ax=ax, color='lightblue', linewidth=.5)
    
    aoi_series = gpd.GeoSeries([aoi])

    # Plot the user-specified AOI polygon
    aoi_series.boundary.plot(ax=ax, color='#8B0000', linewidth=1,linestyle='--')
    
    plt.title("DSWx Tile Boundary and User-specified AOI")
    plt.show()
    
    
def main():

    ps = config.getPS()

    # Get the filtered list of urls
    filtered_urls,dswx_data_df = searchDSWx(ps)
    
    dlDSWx(filtered_urls,ps,ps.dataDir)
    
    return filtered_urls,dswx_data_df


if __name__ == '__main__':
    '''
    Main driver.
    '''

    # main()

    