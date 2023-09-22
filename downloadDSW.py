#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 21 16:21:35 2023

Download DSWx data

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime
import config
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


ps = config.getPS()


start_date = datetime(int(ps.date_start[0:4]), int(ps.date_start[4:6]), int(ps.date_start[6:8]))    
stop_date = datetime(int(ps.date_stop[0:4]), int(ps.date_stop[4:6]), int(ps.date_stop[6:8]))  








def search():
    # URL of CMR service
    STAC_URL = 'https://cmr.earthdata.nasa.gov/stac'
    
    # Setup PySTAC client
    provider_cat = Client.open(STAC_URL)
    catalog = Client.open(f'{STAC_URL}/POCLOUD/')
    
    # We would like to create mosaics for May 2023
    date_range = "2023-05-01/2023-05-30"
    
    # Load the geometry for Australia to retrieve bounding boxes for our search
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    australia_shape = world[world['name']=='Australia']
    bbox = australia_shape.iloc[0].geometry.bounds
    print(bbox)
    
    opts = {
        'bbox' : bbox, 
        'collections': ps.collections,
        'datetime' : date_range,
        # querying by cloud cover does not work (04/27/23)
        # We will instead filter results by parsing the associated XML files for each granule
        # 'query':{
        #     'eo:cloud_cover':{
        #         'lt': 10    
        #     },
        # }
    }
    
    search = catalog.search(**opts)
    items = search.get_all_items()
    
    def filter_by_cloud_cover(item, threshold=10):
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
    
    filtered_items = list(filter(filter_by_cloud_cover, items))
    print(len(filtered_items))
    
    def return_granule(item):
        return item.assets['0_B01_WTR'].href
    filtered_urls = list(map(return_granule, filtered_items))
    print(len(filtered_urls))
    
    
    output_path = Path('../data/australia')
    if not output_path.exists():
        output_path.mkdir(parents=True)

def dl(url,outname):   
    response = requests.get(url,stream=True,allow_redirects=True)
    
    # Open the local file for writing
    with open(outname, 'wb') as file:
        # Iterate through the content and write to the file
        for data in response.iter_content(chunk_size=int(2**14)):
            file.write(data)
            
def dlDSWx(): 

    # Create a list of file path/names
    outNames = []
    dlSLCs = []
    for ii in range(len(gran)):
        fname = os.path.join(outdir, gran[ii] + '.zip')
        if not os.path.isfile(fname):
            outNames.append(os.path.join(outdir, gran[ii] + '.zip'))
            dlSLCs.append(slcUrls[ii])
            
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
        
    print('Downloading the following files:')
    print(dl_list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=nproc) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(dl, url, outName) for url, outName in zip(dlSLCs, outNames)]
        concurrent.futures.wait(futures)

    # check the files
    for ii in range(len(gran)):
        fname = os.path.join(outdir, gran[ii] + '.zip')
        if not os.path.isfile(fname):
            print('Warning: File does not exist ' + fname)
        else:
            if os.path.getsize(fname) < 2**30: # If it's smaller than 1 Gb
                print('Warning: ' + fname + ' is too small. Try again.')