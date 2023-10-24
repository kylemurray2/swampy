#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:12:46 2023

-Download Copernicus DEM if needed
-Read the DSW image and DEM rasters.
-Create a binary mask from the DSW raster where there is water.
-For the border of the DSW, use a convolution with a 3x3 filter to find the 
edges.
-Using the binary edge mask, extract elevation values from the DEM where the 
mask is True.
-Calculate the average or median of these elevation values, which will give you 
an approximation of the water surface elevation.

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os, rasterio, sys, requests, glob
from scipy.signal import convolve2d
from swampy import config
from rasterio.warp import reproject, Resampling

from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from skimage.morphology import remove_small_objects
from datetime import datetime



def getDEM(bounds,demtype='COP30',apiKey=None, srtm=False):
    '''
    bounds 'S,N,W,E'
    set srtm=True if you want to download the SRTM DEM instaerd of the defualt copernicus
    Recommended to use the copernicus DEM (not srtm)
    '''
    home_dir = os.environ['HOME']
    if not os.path.isdir('./DEM'):
        os.mkdir('DEM')

    miny,maxy,minx,maxx = miny,maxy,minx,maxx = bounds.split(sep=',')

    if srtm:
        os.system('dem.py -a stitch -b ' + str(miny) + ' ' + str(maxy) + ' ' + str(minx) + ' ' + str(maxx) +' -r -s 1 -c -f')
        os.system('mv demL* ./DEM/')

    else:
        if apiKey == None:
            print("Didn't inlcude API key. Reading the file ~/.otkey for open topography API key...")
            
            if os.path.isfile(home_dir + '/.otkey'):
                with open(home_dir + '/.otkey') as f:
                    apiKey = f.readline().rstrip()
            else:
                print('No .otkey file found. Include API key and try again.')
                sys.exit(1)
        
        baseurl = "https://portal.opentopography.org/API/globaldem?"
        
        data = dict(demtype='COP30',
        south=miny,
        north=maxy,
        west=minx,
        east=maxx,
        outputFormat='GTiff',
        API_Key=apiKey)
        
        r = requests.get(baseurl, params=data, timeout=100, allow_redirects=True)
        open('DEM/dem.tif', 'wb').write(r.content)
        os.system('gdal_translate DEM/dem.tif -Of ISCE DEM/cop_dem_glo30_wgs84.dem')


def extract_bounds_from_wkt(wkt_string, decimals=1):
    '''
    Extract bounds from wkt_string
    for the min values it will always round down to the nearest x decimals
    for the max values it will always round up to the nearest x decimals
    
    '''
    
    def round_to_decimal(value, decimals, direction):
        factor = 10 ** decimals
        if direction == "floor":
            return np.floor(value * factor) / factor
        elif direction == "ceil":
            return np.ceil(value * factor) / factor
        else:
            return value

    # Extract the coordinates from the WKT string
    coords_text = wkt_string.split('((')[1].split('))')[0]
    coords_pairs = coords_text.split(', ')
    
    # Split each pair into lat and lon
    coords = [(float(pair.split(' ')[1]), float(pair.split(' ')[0])) for pair in coords_pairs]

    # Extract min and max lat and lon and round them using the provided decimals
    minlat = round_to_decimal(min(coords, key=lambda x: x[0])[0], decimals, "floor")
    maxlat = round_to_decimal(max(coords, key=lambda x: x[0])[0], decimals, "ceil")
    minlon = round_to_decimal(min(coords, key=lambda x: x[1])[1], decimals, "floor")
    maxlon = round_to_decimal(max(coords, key=lambda x: x[1])[1], decimals, "ceil")

    return minlat, maxlat, minlon, maxlon

        
def dlDEM(ps):

    minlat, maxlat, minlon, maxlon = extract_bounds_from_wkt(ps.polygon,decimals=0)

    demBounds = f"{int(minlat)},{int(maxlat)},{int(minlon)},{int(maxlon)}"
    
    # Download dem if it doesn't exist
    if not os.path.isdir('./DEM'):
        print("Downloading copernicus DEM with bounds " + demBounds)
        getDEM(demBounds)
        DEM = glob.glob(os.path.join(ps.workdir, 'DEM', '*wgs84.dem'))[0]
    else:
        print("DEM alread exists.. skipping download.")
        DEM = glob.glob(os.path.join(ps.workdir, 'DEM', '*wgs84.dem'))[0]

    # Updating DEM’s wgs84 xml to include the full path to the DEM
    os.system(f'fixImageXml.py -f -i {DEM} >> log')

    if len(DEM) == 0:
        print('ERROR: DEM does not exists and could not be downloaded.')
        sys.exit(1)

    return demBounds, DEM


def crop_rasters_to_bounds(dsw_path, dem_path, minlon, minlat, maxlon, maxlat):
    '''
    Crop DEM and DSW images to same area
    if the two cropped rasters have different shapes, the output_tif raster is 
    resampled to match the dimensions of the dem raster.
    '''
    with rasterio.open(dsw_path) as output_tif:
        # Transform the bounds to the CRS of the output TIF
        bounds_transformed = transform_bounds(rasterio.crs.CRS.from_epsg(4326), output_tif.crs, minlon, minlat, maxlon, maxlat)
        
        # Compute the window corresponding to the transformed bounds
        window = from_bounds(*bounds_transformed, output_tif.transform)
        
        # Crop the raster
        output_tif_data = output_tif.read(window=window)

    with rasterio.open(dem_path) as dem:
        
        transform = rasterio.transform.from_bounds(minlon, minlat, maxlon, maxlat, dem.width, dem.height)
        bounds_transformed = rasterio.transform.array_bounds(dem.height, dem.width, transform)
        # Compute the window corresponding to the transformed bounds
        window_dem = from_bounds(*bounds_transformed, dem.transform)
        
        # Crop the DEM raster
        dem_data = dem.read(window=window_dem)
        # If the cropped output and DEM data have different shapes, resample the output data
        if output_tif_data.shape[1:] != dem_data.shape[1:]:
            output_resampled = dem_data.copy()
            reproject(
                output_tif_data, 
                output_resampled,
                src_transform=rasterio.windows.transform(window, output_tif.transform),
                src_crs=output_tif.crs,
                dst_transform=rasterio.windows.transform(window_dem, dem.transform),
                dst_crs=dem.crs,
                resampling=Resampling.bilinear
            )
            output_tif_data = output_resampled

    return output_tif_data, dem_data



def extract_water_edge_elevation(dsw_path, dem_path, ps, plot_flag=True):
    # Crop images around polygon bounds  
    print('Cropping the images around the polygon...')
    minlat, maxlat, minlon, maxlon = extract_bounds_from_wkt(ps.polygon,decimals=2)
    
    
    dsw,dem = crop_rasters_to_bounds(dsw_path, dem_path, minlon, minlat, maxlon, maxlat)
    dsw =dsw[0,:,:]
    dem =dem[0,:,:]
    binary_dsw = dsw !=0

    minimumPixelsInRegion = 60
    # Remove small objects (i.e., small islands of zeros)
    cleaned_binary_dsw = remove_small_objects(binary_dsw, minimumPixelsInRegion, connectivity=1)
    inverse_cleaned_binary_dsw = ~cleaned_binary_dsw
    cleaned_inverse_dsw = remove_small_objects(inverse_cleaned_binary_dsw, minimumPixelsInRegion, connectivity=1)
    binary_dsw_clean = ~cleaned_inverse_dsw
    

    
    
    # Create a binary mask for DSW == 1
    water_mask = binary_dsw_clean == 1

    # Find the edges of the water mask using convolution
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    
    # water_mask = water_mask[0:50,0:125]
    # dem = dem[0:50,0:125]


    edges = convolve2d(water_mask, kernel, mode='same', boundary='symm')
    edge_mask = edges > 0

    # Extract the elevation values from the DEM where the edge mask is True
    water_edge_elevations = dem[edge_mask]

    # Calculate the median elevation of the water's edge
    median_elevation = np.median(water_edge_elevations)

    edge_line = np.zeros(edge_mask.shape) *np.nan
    edge_line[edge_mask] = 1

    if plot_flag:

        fig,ax = plt.subplots(2,1)
        ax[0].imshow(binary_dsw,cmap='magma');ax[0].set_title('Original water mask')
        ax[1].imshow(binary_dsw_clean,cmap='magma');ax[1].set_title('Cleaned water mask')
        plt.show()
    
        plt.figure()
        plt.imshow(dem,cmap='magma')
        plt.imshow(edge_line,cmap='Greys')
        plt.title('DEM with water edge')
        plt.show()
    
    return median_elevation


def convert_to_decimal_year(date_str):
    """Convert a YYYYMMDD string to a decimal year."""
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    
    # Start of the year
    start_of_year = datetime(date_obj.year, 1, 1)
    
    # Start of the next year
    start_of_next_year = datetime(date_obj.year + 1, 1, 1)
    
    # Compute the fractional year
    fraction_passed = (date_obj - start_of_year) / (start_of_next_year - start_of_year)
    
    return date_obj.year + fraction_passed

# Convert YYYYMMDD strings to datetime.date objects
def convert_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y%m%d').date()

def main():

    ps = config.getPS()
        
    date_dirs = glob.glob(ps.dataDir + '/2???????')
    date_dirs.sort()
    dsw_paths = glob.glob(ps.dataDir + '/2???????/mosaic.tif')
    dsw_paths.sort()
    
    dates=[]
    for date_fn in date_dirs:
        dates.append(date_fn.split('/')[-1])
    
    dec_year = [convert_to_decimal_year(date) for date in dates]
    date_objects = [convert_to_datetime(date) for date in dates]

    # If DEM is needed:
    if ps.demPath == 'none':
        demBounds, dem_path = dlDEM(ps)
    else:
        dem_path = ps.demPath
    
    elevations = []
    for dsw_path in dsw_paths:

        water_surface_elevation = extract_water_edge_elevation(dsw_path, dem_path,ps)
        print(f"The estimated water surface elevation is: {water_surface_elevation:.5f} meters.")
        elevations.append(water_surface_elevation)
    
    plt.figure()
    plt.plot(date_objects, elevations, '.')
    plt.xlabel('Time')
    plt.ylabel('Water elevation (m)')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    '''
    Main driver.
    '''
    # main()

# # Open the GeoTIFF file
# with rasterio.open(dsw_path) as src:
#     # Read the dataset into a 2D array
#     array = src.read(1)  # 1 means you are reading the first band. Adjust if your GeoTIFF has multiple bands.
    
#     # If you want to extract other metadata or properties of the GeoTIFF
#     profile = src.profile  # gets the profile of the dataset (CRS, transform, etc.)
#     transform = src.transform  # gets the affine transform of the dataset


