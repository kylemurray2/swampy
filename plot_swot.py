#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:31:34 2024

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os,glob
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from pathlib import Path
import pandas as pd
from SARTS import makeMap
import cartopy.crs as ccrs
import netCDF4 as nc


# Plot a bunch of files at once
# Direct folder path of shapefiles
folder = Path("swot/HR_Lakes")
# State filename extension to look for within the folder, in this case .shp which is the shapefile
shapefiles = folder.glob("*.zip")

# Merge/Combine multiple shapefiles in folder into one
gdf = pd.concat([
    gpd.read_file(shp)
    for shp in shapefiles
]).pipe(gpd.GeoDataFrame)

# Crop to smaller latitude range
latmin,latmax = -10.75, -10.2
subset_gdf = gdf.cx[:, latmin:latmax]

bounds = subset_gdf.total_bounds
minlon, minlat, maxlon, maxlat = bounds
axm,dcrs = makeMap.mapBackground(minlon, maxlon, minlat, maxlat,zoomLevel=11,scalebar=10)
subset_gdf.plot(column='wse',ax=axm, transform=ccrs.PlateCarree(), facecolor='blue', edgecolor='black',zorder=2)


# PLOT A SSH .nc file _________________________________________________________
ssh_file_list = glob.glob("./swot/LR_SSH/SWOT*Expert*.nc")
fn = ssh_file_list[0]
ds = nc.Dataset(fn)
wse = ds.variables['depth_or_elevation'][:]
latitudes = ds.variables['latitude'][:]  # Adjust 'latitude' to your dataset's latitude variable name
longitudes = ds.variables['longitude'][:]  # Adjust 'longitude' as necessary
latitudes = np.asarray(latitudes)
longitudes = np.asarray(longitudes)
wse=np.asarray(wse)


# Define your latitude bounds
minlat, maxlat = -11, -10  # Example latitude bounds
minlon,maxlon,minlat,maxlat = np.nanmin(longitudes),np.nanmax(longitudes),np.nanmin(latitudes),np.nanmax(latitudes)
# # Find indices where latitude is within the desired range
# lat_inds = np.where((latitudes >= minlat) & (latitudes <= maxlat))



# # Crop the dataset
# cropped_wse = wse[lat_inds, :]  # Assuming 'depth_or_elevation' is 2D: [lat, lon]
# cropped_lats = latitudes[lat_inds]
# cropped_lons = longitudes[:]  # Assuming you want all longitudes

axm,dcrs = makeMap.mapBackground(minlon, maxlon, minlat, maxlat,zoomLevel=3,scalebar=10)
vmin,vmax=-10,10
cm = 'jet'


plt.figure(figsize=(15, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
plt.scatter(x=longitudes,y=latitudes,c=wse,marker='.',vmin=-6000,vmax=6000)

# PLOT A RASTER IMAGE_________________________________________________________
# Load/read data
hr_raster_file_list = glob.glob("./swot/HR_Raster/SWOT*")
fn = hr_raster_file_list[0]
ds = nc.Dataset(fn)
print(ds.variables.keys())
wse = ds.variables['wse'][:]
# wa = ds.variables['model_wet_tropo_cor'][:]; plt.figure();plt.imshow(wa)

lons = ds.variables['longitude'][:]
lats = ds.variables['latitude'][:]

plt.figure();plt.imshow(lons.mask)
plt.figure();plt.imshow(lats)

from scipy import interpolate

# Assuming 'data' is your 2D numpy array with NaNs
x = np.arange(0, wse.shape[1])
y = np.arange(0, wse.shape[0])
xx, yy = np.meshgrid(x, y)

# Interpolate using griddata, for example
lons_interp = interpolate.griddata((xx[~lons.mask], yy[~lons.mask]), lons.data[~lons.mask],
                                   (xx, yy), method='cubic')
lats_interp = interpolate.griddata((xx[~lats.mask], yy[~lats.mask]), lats.data[~lats.mask],
                                   (xx, yy), method='cubic')

mask = ~np.isnan(lons_interp)
lons_interp = interpolate.griddata((xx[mask], yy[mask]), lons_interp[mask],
                                   (xx, yy), method='nearest')
lats_interp = interpolate.griddata((xx[mask], yy[mask]), lats_interp[mask],
                                   (xx, yy), method='nearest')

vmin,vmax = 200,400
pad = 0
zoom = 10
bg = 'World_Imagery'
cm = 'jet'
title = 'Water surface elevation'

minlon=np.nanmin(lons)
minlat=np.nanmin(lats)
maxlon=np.nanmax(lons)
maxlat=np.nanmax(lats)

makeMap.mapBackground( minlon, maxlon, minlat, maxlat, zoom, title, pad=0, scalebar=False, borders=True)
plt.imshow(np.flipud(wse),extent=[minlon,maxlon,minlat,maxlat],transform=ccrs.PlateCarree(),zorder=12,vmin=vmin,vmax=vmax,cmap=cm)

