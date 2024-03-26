'''
Download SWOT data from podaac

https://deotb6e7tfubr.cloudfront.net/s3-edaf5da92e0ce48fb61175c28b67e95d/podaac-ops-cumulus-docs.s3.us-west-2.amazonaws.com/web-misc/swot_mission_docs/pdd/SALP-ST-M-EA-17043-CN_0103_20220411_Rev1.3.pdf?A-userid=None&Expires=1705714999&Signature=B9844OVgfwVrvICJI9KxdwcyO4rl3MnYcz4IvyOsTcupX9RRaVZgtTOLKPg~IiKSZOqnp5XWR~WyfdpYG1A7MJI281-jM960r1MiNP~DVicVO0kO4rR9nl-si9ySZexMudNjf3uH59ZIOmd6RDS-CLCvbIkxQL7S3TTJsvwPooGbSnQ-N5GHvWOHK3~zO9uWoHgw2zvrUsDb2Tj8YpqzgDu5k6DeQQRoN0n4gQ3qNlPIA7ujv3rVaIoWkpaPhBs9ZEss58xC7ug82YaZL273qI93wEWbUxZtd4zVfodqt6p4pXEePRJukOKT-bQPef4UYU4UZmQuyeb-36ZgxdXfyg__&Key-Pair-Id=K3OEOUZXFQBEJ5

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
import netCDF4 as nc
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


# Get our local parameters from params.yaml file
ps = config.getPS()

# Login to earth data
auth = earthaccess.login()
#earthaccess data search
start_date = ps.date_start[0:4] + '-' + ps.date_start[4:6]+ '-' + ps.date_start[6:8] + ' 00:00:00'
stop_date = ps.date_stop[0:4] + '-' + ps.date_stop[4:6]+ '-' + ps.date_stop[6:8] + ' 23:59:59'
short_name = 'SWOT_L2_NALT_OGDR_1.0'# 'SWOT_L2_HR_Raster_1.1' #SWOT_L2_LR_SSH_1.1
temporal = (start_date,stop_date)
granule_name = '*20230731_155055_20230731*'
bounding_box =''# loads(ps.polygon).bounds # WSEN
results = earthaccess.search_data(short_name=short_name, 
                                  temporal = temporal,
                                  granule_name = granule_name)

# Download data
earthaccess.download(results, "./swot/Altimeter")

# Load/read data
hr_raster_file_list = glob.glob("./swot/Altimeter/SWOT*")
fn = hr_raster_file_list[0]

ds_nadir = xr.open_mfdataset(fn, combine='nested', concat_dim="time", decode_times=False, engine='h5netcdf', group='data_01')
ds_nadir

import cartopy.io.img_tiles as cimgt

np.where(ds_nadir.longitude <10)

plt.figure(figsize=(15, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_global()

ax.set_extent([-169,-140, 55,72], crs=ccrs.PlateCarree())

ax.coastlines()
bg = 'World_Shaded_Relief'
url = 'https://server.arcgisonline.com/ArcGIS/rest/services/' + bg + '/MapServer/tile/{z}/{y}/{x}.jpg'
image = cimgt.GoogleTiles(url=url)
ax.add_image(image,10) #zoom level
plt.scatter(x=ds_nadir.longitude, y=ds_nadir.latitude, c=ds_nadir.depth_or_elevation, marker='.',vmin=-100,vmax=1000)
plt.colorbar().set_label('Depth or Elevation (m)')


plt.figure(figsize=(15, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-169,-140, 55,72], crs=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
plt.scatter(x=ds_nadir.longitude, y=ds_nadir.latitude, c=ds_nadir.mean_sea_surface_cnescls, marker='.')
plt.colorbar().set_label('mean_sea_surface(m)')