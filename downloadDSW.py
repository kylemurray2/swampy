#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 21 16:21:35 2023

Download DSWx data

@author: km
"""
from shapely import wkt

import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime
from swampy import config
# GIS imports
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
from shapely import Polygon
import fiona
# misc imports
from pystac_client import Client
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import os
import json
# web imports
from urllib.request import urlopen
import requests
import concurrent.futures

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

    # convert to the datetime format
    start_date = datetime(int(ps.date_start[0:4]), int(ps.date_start[4:6]), int(ps.date_start[6:8]))    
    stop_date = datetime(int(ps.date_stop[0:4]), int(ps.date_stop[4:6]), int(ps.date_stop[6:8]))  
            
    # Convert the wkt polygon to a Shapely Polygon
    aoi = wkt.loads(ps.polygon)
    intersects_geometry = aoi.__geo_interface__
    
    # Search data through CMR-STAC API
    stac = 'https://cmr.earthdata.nasa.gov/cloudstac/'    # CMR-STAC API Endpoint
    print("Connecting to API...")
    api = Client.open(f'{stac}/POCLOUD/')
    collections = ['OPERA_L3_DSWX-HLS_PROVISIONAL_V1']
    
    search_params = {"collections": collections,
                     "intersects": intersects_geometry,
                     "datetime": [start_date, stop_date],
                     "max_items": 10000}
    search_dswx = api.search(**search_params)
    items = search_dswx.get_all_items()

    # Filter cloudy days
    print("Filtering cloudy days...")
    filtered_items = list(filter(filter_by_cloud_cover, items))
    print(str(len(list(items))) + ' items were found.')
    print(str(len(filtered_items)) + ' items meet the cloud threshold')

    filtered_urls = list(map(return_granule, filtered_items))
    print(len(filtered_urls))
    
    return filtered_urls


def dl(url,outname):   
    response = requests.get(url,stream=True,allow_redirects=True)
    
    # Open the local file for writing
    with open(outname, 'wb') as file:
        # Iterate through the content and write to the file
        for data in response.iter_content(chunk_size=int(2**14)):
            file.write(data)

            
def dlDSWx(urls,ps,outdir): 

    # Create a list of file path/names
    outNames = []
    dl_list = []
    for url in urls:
        fname = os.path.join(outdir, url.split('/')[-1])
        if not os.path.isfile(fname):
            outNames.append(os.path.join(outdir, url.split('/')[-1]))
            dl_list.append(url)
            
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
        
    print('Downloading the following files:')
    print(dl_list)
    print('downloading with '  + str(nproc) + ' cpus')

    with concurrent.futures.ThreadPoolExecutor(max_workers=nproc) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(dl, url, outName) for url, outName in zip(dl_list, outNames)]
        concurrent.futures.wait(futures)

    # check the files
    for url in urls:
        fname = os.path.join(outdir, url.split('/')[-1])
        if not os.path.isfile(fname):
            print('Warning: File does not exist ' + fname)
        else:
            if os.path.getsize(fname) < 2**10: # If it's smaller than 1 Kb
                print('Warning: ' + fname + ' is too small. Try again.')
                

def main():

    ps = config.getPS()

    # Get the filtered list of urls
    filtered_urls = searchDSWx(ps)
    
    dlDSWx(filtered_urls,ps,ps.outdir)