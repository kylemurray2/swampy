'''
Download SWOT data from podaac

From example
https://podaac.github.io/tutorials/notebooks/SearchDownload_SWOTviaCMR.html

Each dataset has it’s own unique collection ID. 

https://podaac.github.io/tutorials/notebooks/datasets/SWOTHR_s3Access_real_data_v11.html

	Short Names:
SWOT_L2_HR_Raster_1.1: https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_Raster_1.1 Rasterized water surface elevation and inundation extent in geographically fixed tiles at resolutions of 100 m and 250 m in a Universal Transverse Mercator projection grid. Provides rasters with water surface elevation, area, water fraction, backscatter, geophysical information.
SWOT_L2_LR_SSH_1.1: sea surface heights https://podaac.jpl.nasa.gov/dataset/SWOT_L2_LR_SSH_1.1


Other resources for swot:
    https://github.com/SWOT-community/SWOT-OpenToolkit/tree/main
    
    
SWOT Level 2 KaRIn High Rate Version 1.1 Datasets from calibration phase, 4/8 through 4/22:
Water Mask Pixel Cloud NetCDF - SWOT_L2_HR_PIXC_1.1 (DOI: 10.5067/SWOT-PIXC-1.1)
Water Mask Pixel Cloud Vector Attribute NetCDF - SWOT_L2_HR_PIXCVec_1.1 (DOI: 10.5067/SWOT-PIXCVEC-1.1)
River Vector Shapefile - SWOT_L2_HR_RiverSP_1.1 (DOI: 10.5067/SWOT-RIVERSP-1.1)
Lake Vector Shapefile - SWOT_L2_HR_LakeSP_1.1 (DOI: 10.5067/SWOT-LAKESP-1.1)
Raster NetCDF - SWOT_L2_HR_Raster_1.1 (DOI: 10.5067/SWOT-RASTER-1.1)

'''

import requests,json,glob,os,zipfile 
from pathlib import Path
import pandas as pd
from urllib.request import urlretrieve
from json import dumps

import earthaccess
from earthaccess import Auth, DataCollections, DataGranules, Store

from swampy import config
from SARTS import makeMap

from datetime import datetime
from shapely import wkt
from pystac_client import Client  

import s3fs
import fiona
import h5netcdf
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
# import hvplot.xarray
import earthaccess
from earthaccess import Auth, DataCollections, DataGranules, Store
from shapely.wkt import loads
from osgeo import gdal

import cartopy.crs as ccrs
from pyproj import Proj, Transformer


import podaac_data_downloader




# Get our local parameters from params.yaml file
ps = config.getPS()

# Login to earth data
auth = earthaccess.login()

#earthaccess data search
start_date = ps.date_start[0:4] + '-' + ps.date_start[4:6]+ '-' + ps.date_start[6:8] + ' 00:00:00'
stop_date = ps.date_stop[0:4] + '-' + ps.date_stop[4:6]+ '-' + ps.date_stop[6:8] + ' 23:59:59'
#'SWOT_L2_LR_SSH_1.1'# 'SWOT_L2_HR_LakeSP_1.1' #'SWOT_L2_HR_Raster_1.1'# 'SWOT_L2_HR_Raster_1.1' #SWOT_L2_LR_SSH_1.1 SWOT_L2_HR_LakeSP_1.1
short_name='SWOT_L2_HR_LakeSP_1.1'
temporal = (start_date,stop_date)
granule_name ='*NA*'# '*_261_*'# '*NA*' #'*100m_UTM11*'
bounding_box = loads(ps.polygon).bounds # WSEN
results = earthaccess.search_data(short_name=short_name, 
                                  temporal = temporal,
                                  granule_name = granule_name)

lakes_results = earthaccess.search_data(short_name = 'SWOT_L2_HR_Raster_1.1', 
                                        temporal = ('2023-04-08 00:00:00', '2026-04-25 23:59:59'),
                                        granule_name = '11S') # here we filter by Reach

# passes = []
# for r in river_results:
#    pass_number = r['umm']['SpatialExtent']['HorizontalSpatialDomain']['Track']['Passes'][0]['Pass']
#    passes.append(pass_number)

# Download data
# earthaccess.download(results, "./swot/HR_Raster")
earthaccess.download(lakes_results, "./swot/HR_Lakes")
earthaccess.download(results, "./swot/LR_SSH")
