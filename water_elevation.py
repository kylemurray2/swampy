#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:12:46 2023

-Works with LIDAR DEM, or 
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
from scipy.ndimage import binary_dilation,generate_binary_structure

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
    
    utils.update_yaml_key('params.yaml', 'demPath', DEM)
    print('Updating demPath in params.yaml with copernicus DEM')
    
    if len(DEM) == 0:
        print('ERROR: DEM does not exists and could not be downloaded.')
        sys.exit(1)

    return demBounds, DEM


def get_epsg(ds):
    """Retrieve EPSG code of a GDAL dataset's projection"""
    projection = ds.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    return srs.GetAuthorityCode(None)


def load_gt(dsw_path):
    with rasterio.open(dsw_path) as ds:
        array = ds.read().squeeze()
    return array

def find_land_adjacent_to_water(raster):
    """
    Function to find the land pixels that are directly adjacent to water pixels in a single-band raster.
    
    Parameters:
    raster (numpy array): A numpy array representing the raster with land, water, and clouds.
    
    Returns:
    mask: True where a pixel classified as land is touching a pixel classified as water
    """
    # Create masks for land and water
    land_mask = raster == 0
    water_mask = raster == 1
    other_mask = raster == 5

    # Dilate the water mask to include potential adjacent land pixels
    dilated_water_mask = binary_dilation(water_mask)
    
    # There is an artifact around some of the clouds, where they are defined as 
    # water for 1 or 2 pixels on their edges. We can grow the clouds a few pixels
    # to make sure those artifacts are not included in the edge detection.
    n_pixels = 7  # Example for 2 pixels dilation
    structuring_element = generate_binary_structure(2, 1)  # 2D cross (1-connectivity)
    dilated_other_mask = binary_dilation(other_mask, structure=structuring_element, iterations=n_pixels)
    # plt.figure();plt.imshow(dilated_water_mask[4550:4660,7200:7350])
    
    # Find the intersection of the original land mask with the dilated water mask
    # This will give us land pixels that are adjacent to water pixels
    land_adjacent_to_water = land_mask & dilated_water_mask & ~dilated_other_mask
    
    # plt.figure();plt.imshow(land_adjacent_to_water[4550:4660,7200:7350])

    # Make sure we don't include the land pixels that are under the dilated water mask
    # This is done by subtracting the original water mask from the result
    land_adjacent_to_water = land_adjacent_to_water & ~water_mask

    return land_adjacent_to_water



def extract_water_edge_elevation(dsw_path,dem, ps, DEM_water_elevation, plot_flag=True):
    '''
    Resample DSW image to DEM grid,
    Remove small groups of pixels
    Find the edge of the water,
    Find where the edge intersects the DEM and average those elevations
    '''
    
    resampled_path = dsw_path.replace('mosaic.tif', 'mosaic_resampled.tif')
    if not os.path.isfile(resampled_path):
        print('resampling DSW to the DEM grid')
        dsw = utils.resample_to_match(dsw_path, ps.demPath, output_path=resampled_path)
    else:
        print('Using existing resampled DSW')
        dsw = load_gt(resampled_path)
    dsw=dsw.astype(int)

    # plt.figure();plt.imshow(dsw,vmin=0,vmax=3)
    # plt.figure();plt.imshow(cleaned_dsw,vmin=0,vmax=3)

    # We need to make sure we only extract land/water intersections and not cloud/land etc.
    
    # Start with binary image of water and other
    binary_dem = (dem >= (DEM_water_elevation - 0.2)) & (dem <= (DEM_water_elevation + 0.2))
    binary_dsw = (dsw == 1) | (dsw == 2)
    binary_dsw[np.where((dsw !=0) & (dsw !=1))] = np.nan
    # plt.figure();plt.imshow(binary_dsw)
    # plt.figure();plt.imshow(binary_dem)
    
    # Remove small objects (i.e., small islands of zeros)
    # This will remove islands that are smaller than 0.1% of the image.
    minimumPixelsInRegion = .001 *(binary_dem.shape[0]*binary_dem.shape[1])
    cleaned_binary_dsw = remove_small_objects(binary_dsw, minimumPixelsInRegion, connectivity=1)
    inverse_cleaned_binary_dsw = ~cleaned_binary_dsw
    cleaned_inverse_dsw = remove_small_objects(inverse_cleaned_binary_dsw, minimumPixelsInRegion, connectivity=1)
    binary_dsw_clean = ~cleaned_inverse_dsw
    
 
    
    # plt.figure();plt.imshow(binary_dsw_clean)

    
    if len(np.unique(binary_dsw_clean))==2:
                
        # plt.figure();plt.imshow(dsw);plt.title('binary_dsw')
        # plt.figure();plt.imshow(binary_dsw_clean);plt.title('binary_dsw_clean')
        # Create a binary mask for DSW == 1
        # water_mask = dsw_tri == 1
        # # Find the edges of the water mask using convolution
        # kernel = np.array([
        #     [-1, -1, -1],
        #     [-1,  8, -1],
        #     [-1, -1, -1]
        # ])
        # edges = convolve2d(water_mask, kernel, mode='same', boundary='symm')
        # edge_mask = edges > 0
        
        # Now add back in the clouds so we can make sure not to find water/cloud boundaries
        dsw_tri =np.zeros(binary_dsw_clean.shape)
        dsw_tri[binary_dsw_clean] = 1
        dsw_tri[~np.isin(dsw, [0, 1, 2, 3])] = 5 # dsw_tri is 0:land 1:water and 5:other
        # plt.figure();plt.imshow(dsw_tri,vmin=0,vmax=5)

        edge_mask = find_land_adjacent_to_water(dsw_tri)
        minpix = .001* np.sum(edge_mask)
        edge_mask = remove_small_objects(edge_mask, minpix, connectivity=1)
        # Extract the elevation values from the DEM where the edge mask is True
        water_edge_elevations = dem[edge_mask]
        water_edge_elevations_rounded = np.round(water_edge_elevations)
        # exclude any values that are the exact value of the DEM water elevation (guess)
        water_edge_elevations_rounded = water_edge_elevations_rounded[water_edge_elevations_rounded!=round(DEM_water_elevation)]
        values, counts = np.unique(water_edge_elevations_rounded.ravel(), return_counts=True)
        
        if len(counts) > 1:
            mode_elevation = values[np.argmax(counts)]
            # Calculate the median elevation of the water's edge
            median_elevation = np.nanmedian(water_edge_elevations[water_edge_elevations!=DEM_water_elevation])
            std_elevation = np.nanstd(water_edge_elevations[water_edge_elevations!=DEM_water_elevation])
    
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
    
    else:
        print(dsw_path + ' failed')
        median_elevation,mode_elevation,std_elevation = np.nan,np.nan,np.nan
    
    return median_elevation,mode_elevation,std_elevation


def convert_to_decimal_year(date_str):
    """Convert a YYYYMMDD string to a decimal year."""
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    start_of_year = datetime(date_obj.year, 1, 1)
    start_of_next_year = datetime(date_obj.year + 1, 1, 1)
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
    

    # reproject DEM to 4326 and update yaml file with that path
    with rasterio.open(ps.demPath) as dem_src:
        epsg_dem = dem_src.crs.to_string()
        
    if not epsg_dem == 'EPSG:4326':
        print('Reprojecting the DEM to EPSG:4326')
        dem_path_4326 = 'DEM/dem_4326.tif'
        utils.reproject_raster(ps.demPath, dem_path_4326, output_epsg='EPSG:4326')
        ps.demPath = dem_path_4326
        utils.update_yaml_key('params.yaml', 'demPath', dem_path_4326)
        print('Updating demPath in params.yaml with reprojected dem')
    
    
    # Crop the DEM
    dem_cropped_path = 'DEM/dem_4326_cropped.tif' 
    if not os.path.isfile(dem_cropped_path):
        print('Cropping the images around the polygon...')
        minlat, maxlat, minlon, maxlon = extract_bounds_from_wkt(ps.polygon,decimals=3)
        utils.crop_geotiff(ps.demPath,dem_cropped_path, minlon, minlat, maxlon, maxlat) # might not work if bounds are outside of the image bounds
        ps.demPath = dem_cropped_path
        utils.update_yaml_key('params.yaml', 'demPath', dem_cropped_path)
        print('Updating demPath in params.yaml with cropped dem')

    
    # Load the DEM 
    dem = load_gt(dem_cropped_path)
    dem[dem<0] = np.nan  
    non_nan_dem = dem[~np.isnan(dem)]
    values, counts = np.unique(non_nan_dem.ravel(), return_counts=True)
    DEM_water_elevation = values[np.argmax(counts)]
    guess = dem == DEM_water_elevation
    
    vmin = DEM_water_elevation -30
    vmax = DEM_water_elevation + 30
    fig,ax = plt.subplots(2,1,figsize=(7,8))
    ax[0].imshow(dem,vmin=vmin,vmax=vmax,cmap='magma');ax[0].set_title('DEM')
    ax[1].imshow(guess);ax[1].set_title('Water mask initial guess')
    plt.show()
    
    elevations_medians = []
    elevations_modes = []
    elevations_stds = []
    # dsw_path = './data/20230430/mosaic.tif'
    for dsw_path in dsw_paths:
        median_elevation,mode_elevation,std_elevation = extract_water_edge_elevation(dsw_path,dem, ps, DEM_water_elevation, plot_flag=plot_flag)
        print(f"The mode water surface elevation is: {mode_elevation:.5f} meters.")
        print(f"The median water surface elevation is: {median_elevation:.5f} meters.")

        elevations_medians.append(median_elevation)
        elevations_modes.append(mode_elevation)
        elevations_stds.append(std_elevation)
        
    plt.figure()
    plt.plot(date_objects, elevations_medians, '.',label='median')
    plt.plot(date_objects, elevations_modes, '.',label='mode')
    plt.xlabel('Time')
    plt.ylabel('Water elevation (m)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return date_objects,elevations_medians,elevations_modes,std_elevation, DEM_water_elevation
    
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


