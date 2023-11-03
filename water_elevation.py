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
from swampy import config,utils
from rasterio.warp import reproject, Resampling

from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from skimage.morphology import remove_small_objects
from datetime import datetime

from osgeo import gdal, osr


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
    '''
    Download the DEM
    '''
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



def get_epsg(ds):
    """Retrieve EPSG code of a GDAL dataset's projection"""
    projection = ds.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    return srs.GetAuthorityCode(None)



def read_crop_DEM(ps):
    '''
    Reads the DEM file and crops based on polygon in params.yaml
    '''
    with rasterio.open(ps.demPath) as src:
        # Get the current CRS of the dataset
        current_crs = src.crs
    
        print(f"Current CRS of the dataset: {current_crs}")
    
        if current_crs.to_epsg() != 4326:
            transform, width, height = rasterio.warp.calculate_default_transform(
                current_crs, {'init': 'EPSG:4326'}, src.width, src.height, *src.bounds)
    
            # Define metadata for the output file
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': {'init': 'EPSG:4326'},
                'transform': transform,
                'width': width,
                'height': height
            })
            
            if not os.path.isdir('DEM'):
                os.mkdir('DEM')
    
            # Reproject and write to a new GeoTIFF
            with rasterio.open('DEM/output_latlon.tif', 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs={'init': 'EPSG:4326'},
                        resampling=Resampling.nearest)
                    
            demPath = 'DEM/output_latlon.tif'
            utils.update_yaml_key('params.yaml', 'demPath', demPath)
            print('Updating demPath in params.yaml with new dem')
        else:
            print("The dataset is already in EPSG:4326.")
            demPath = ps.demPath
    
    
    dem_ds = rasterio.open(demPath)
    minlat, maxlat, minlon, maxlon = extract_bounds_from_wkt(ps.polygon,decimals=2)
    transform = rasterio.transform.from_bounds(minlon, minlat, maxlon, maxlat, dem_ds.width, dem_ds.height)
    bounds_transformed = rasterio.transform.array_bounds(dem_ds.height, dem_ds.width, transform)
    window_dem = from_bounds(*bounds_transformed, dem_ds.transform)
    dem = dem_ds.read(window=window_dem)
    dem[dem<0] = np.nan
    
    return dem,dem_ds

def load_dsw(dsw_path):
    ds =  rasterio.open(dsw_path)
    array = ds.read().squeeze()
    plt.figure();plt.imshow(array)
    
    
def crop_rasters_to_bounds(dsw_path, dem_ds, minlon, minlat, maxlon, maxlat):
    '''
    Crop DSW image to same area as DEM
    The output_tif raster is then resampled to match the dimensions of the dem raster.
    '''
    with rasterio.open(dsw_path) as output_tif:
        # Transform the bounds to the CRS of the output TIF
        bounds_transformed = transform_bounds(rasterio.crs.CRS.from_epsg(4326), output_tif.crs, minlon, minlat, maxlon, maxlat)
        
        # Compute the window corresponding to the transformed bounds
        window = from_bounds(*bounds_transformed, output_tif.transform)
        
        # Crop the raster
        output_tif_data = output_tif.read(window=window)

    transform = rasterio.transform.from_bounds(minlon, minlat, maxlon, maxlat, dem_ds.width, dem_ds.height)
    bounds_transformed = rasterio.transform.array_bounds(dem_ds.height, dem_ds.width, transform)
    # Compute the window corresponding to the transformed bounds
    window_dem = from_bounds(*bounds_transformed, dem_ds.transform)
    
    # Crop the DEM raster
    dem = dem_ds.read(window=window_dem)
    # If the cropped output and DEM data have different shapes, resample the output data
    if output_tif_data.shape[1:] != dem.shape[1:]:
        output_resampled = dem.copy()
        reproject(
            output_tif_data, 
            output_resampled,
            src_transform=rasterio.windows.transform(window, output_tif.transform),
            src_crs=output_tif.crs,
            dst_transform=rasterio.windows.transform(window_dem, dem_ds.transform),
            dst_crs=dem_ds.crs,
            resampling=Resampling.bilinear
        )
        output_tif_data = output_resampled

    return output_tif_data, dem


def extract_water_edge_elevation(dsw_path, dem_ds, ps, guess, plot_flag=True):
    # Crop images around polygon bounds  
    
    # for i in range(10):
    # dsw_path = dsw_paths[i]
    
    print('Cropping the images around the polygon...')
    minlat, maxlat, minlon, maxlon = extract_bounds_from_wkt(ps.polygon,decimals=2)
    dsw,dem = crop_rasters_to_bounds(dsw_path, dem_ds, minlon, minlat, maxlon, maxlat)
    dsw =dsw[0,:,:]
    dem =dem[0,:,:]

    # plt.figure();plt.imshow(dsw,vmin=0,vmax=10);plt.title(str(i))


    binary_dsw = (dsw == 1) | (dsw == 2)
    
    # plt.figure();plt.imshow(binary_dsw)
    
    minimumPixelsInRegion = 60
    # Remove small objects (i.e., small islands of zeros)
    cleaned_binary_dsw = remove_small_objects(binary_dsw, minimumPixelsInRegion, connectivity=1)
    inverse_cleaned_binary_dsw = ~cleaned_binary_dsw
    cleaned_inverse_dsw = remove_small_objects(inverse_cleaned_binary_dsw, minimumPixelsInRegion, connectivity=1)
    binary_dsw_clean = ~cleaned_inverse_dsw
    
    # plt.figure();plt.imshow(binary_dsw_clean)

    
    if len(np.unique(binary_dsw_clean))==2:
                
        # plt.figure();plt.imshow(dsw);plt.title('binary_dsw')
        # plt.figure();plt.imshow(binary_dsw_clean);plt.title('binary_dsw_clean')
    
        
        # Create a binary mask for DSW == 1
        water_mask = binary_dsw_clean == 1
    
        # Find the edges of the water mask using convolution
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
            
        edges = convolve2d(water_mask, kernel, mode='same', boundary='symm')
        edge_mask = edges > 0

        # Extract the elevation values from the DEM where the edge mask is True
        water_edge_elevations = dem[edge_mask]
    
        water_edge_elevations_rounded = np.round(water_edge_elevations)
        # exclude any values that are the exact value of the DEM water elevation (guess)
        water_edge_elevations_rounded = water_edge_elevations_rounded[water_edge_elevations_rounded!=round(guess)]
        values, counts = np.unique(water_edge_elevations_rounded.ravel(), return_counts=True)
        mode_elevation = values[np.argmax(counts)]

        
        # plt.figure()
        # plt.plot(water_edge_elevations_rounded,'.')
        
        # Calculate the median elevation of the water's edge
        median_elevation = np.median(water_edge_elevations[water_edge_elevations!=59])
        std_elevation = np.nanstd(water_edge_elevations[water_edge_elevations!=59])

        edge_line = np.zeros(edge_mask.shape) *np.nan
        edge_line[edge_mask] = 1
    
        dem2= dem.copy()
        dem2[edge_line==1] = np.nan

        if plot_flag:
    
            fig,ax = plt.subplots(2,1)
            ax[0].imshow(binary_dsw,cmap='magma');ax[0].set_title('Original water mask')
            ax[1].imshow(binary_dsw_clean,cmap='magma');ax[1].set_title('Cleaned water mask')
            plt.title(dsw_path)
            plt.show()
        
            plt.figure()
            plt.imshow(dem2,cmap='magma')
            plt.title('DEM with water edge')
            plt.show()
    
    else:
        print(dsw_path + ' failed')
        median_elevation,mode_elevation,std_elevation = np.nan,np.nan,np.nan
    
    return median_elevation,mode_elevation,std_elevation


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




def main(plot_flag = False):

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
        demBounds, ps.demPath = dlDEM(ps)
    
    ds = rasterio.open(ps.demPath)
    dem = ds.read().squeeze()
    dem[dem<0] = np.nan
    plt.figure();plt.imshow(dem)
    
    dem,dem_ds = read_crop_DEM(ps)
    dem = dem[0,:,:]
    plt.figure();plt.imshow(dem,vmin=55,vmax=200);plt.title('DEM')
    
    # Find the elevation of water in the DEM and make a guess water mask
    values, counts = np.unique(dem.ravel(), return_counts=True)
    mode_value = values[np.argmax(counts)]
    guess = dem==mode_value
    plt.figure();plt.imshow(guess);plt.title('Water mask initial guess')
    
    elevations_medians = []
    elevations_modes = []

    for dsw_path in dsw_paths:
        median_elevation,mode_elevation,std_elevation = extract_water_edge_elevation(dsw_path, dem_ds, ps, guess, plot_flag=plot_flag)
        print(f"The estimated water surface elevation is: {mode_elevation:.5f} meters.")
        elevations_medians.append(median_elevation)
        elevations_modes.append(mode_elevation)
    
    plt.figure()
    plt.plot(date_objects, elevations_medians, '.',label='median')
    plt.plot(date_objects, elevations_modes, '.',label='mode')
    plt.xlabel('Time')
    plt.ylabel('Water elevation (m)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return date_objects,elevations_medians,elevations_modes,std_elevation
    
if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

# # Open the GeoTIFF file
# with rasterio.open(dsw_path) as src:
#     # Read the dataset into a 2D array
#     array = src.read(1)  # 1 means you are reading the first band. Adjust if your GeoTIFF has multiple bands.
    
#     # If you want to extract other metadata or properties of the GeoTIFF
#     profile = src.profile  # gets the profile of the dataset (CRS, transform, etc.)
#     transform = src.transform  # gets the affine transform of the dataset


