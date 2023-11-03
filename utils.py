#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:08:52 2023

extra functions for swampy

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os,re


def update_yaml_key(file_path, key, new_value):
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            # Try to match a YAML key-value pair line
            match = re.match(rf"({key}\s*:\s*)(\S+)", line)
            if match:
                # Replace the value while preserving the key and any surrounding whitespace
                line = f"{match.group(1)}{new_value}\n"
            f.write(line)
            
            


def reproject_raster(input_path, output_path, output_epsg='EPSG:4326'):
    """
    Reproject a raster to a specified EPSG code.
    
    Args:
    - input_path (str): Path to the input raster.
    - output_path (str): Path to save the reprojected raster.
    - output_epsg (str): The EPSG code for the desired output projection. E.g., 'EPSG:4326'.
    """
    
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    
    with rasterio.open(input_path) as src:
        # Check the current CRS of the raster
        current_crs = src.crs.to_string()
        
        # If it's already in the desired EPSG, just copy the raster
        if current_crs == output_epsg:
            print(f"The raster at {input_path} is already in {output_epsg}.")
            return
        
        # Calculate the ideal transformations and dimensions for the desired EPSG
        transform, width, height = calculate_default_transform(
            src.crs, output_epsg, src.width, src.height, *src.bounds
        )
        
        # Define the metadata for the output raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': output_epsg,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject and save the raster
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=output_epsg,
                    resampling=Resampling.bilinear
                )
                
    print(f"Reprojected raster saved to {output_path}.")



def resample_to_match(img_path, reference_path, output_path=None):
    """
    Resample the image GeoTIFF to match the grid of the reference GeoTIFF and save 
    the result.
    """
    
    import rasterio
    from rasterio.warp import reproject, Resampling
    import numpy as np
    
    with rasterio.open(img_path) as dsw_src, rasterio.open(reference_path) as dem_src:
        # Read data from image and metadata from reference
        dsw_data = dsw_src.read()
        
        # Create an empty array with the same shape as the reference data
        resampled_data = np.empty_like(dem_src.read())

        # Perform the reprojection
        reproject(
            source=dsw_data,
            destination=resampled_data,
            src_transform=dsw_src.transform,
            src_crs=dsw_src.crs,
            dst_transform=dem_src.transform,
            dst_crs=dem_src.crs,
            resampling=Resampling.bilinear
        )
        
        if output_path:
            # Save the resampled data as a new GeoTIFF
            with rasterio.open(output_path, 'w', 
                               driver='GTiff', 
                               height=resampled_data.shape[1], 
                               width=resampled_data.shape[2], 
                               count=dem_src.count, 
                               dtype=resampled_data.dtype.name,  # Updated this line
                               crs=dem_src.crs, 
                               transform=dem_src.transform) as dst:
                dst.write(resampled_data)
    
    return resampled_data.squeeze()




def crop_geotiff(input_path, output_path, minlon, minlat, maxlon, maxlat):
    """
    Crop a GeoTIFF to the specified lat/lon bounds using GDAL.
    
    Args:
        input_path (str): Path to the input GeoTIFF.
        output_path (str): Path to save the cropped GeoTIFF.
        minlon (float): Minimum longitude for the bounding box.
        minlat (float): Minimum latitude for the bounding box.
        maxlon (float): Maximum longitude for the bounding box.
        maxlat (float): Maximum latitude for the bounding box.
    """
    from osgeo import gdal

    # Open the input dataset
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    
    # Perform cropping using gdal.Translate
    gdal.Translate(output_path, ds, projWin=[minlon, maxlat, maxlon, minlat])
    
    # Close the dataset
    ds = None
    
    print(f"Cropped raster saved to {output_path}.")

    # ds = gdal.Open(output_path)
    # array = ds.GetVirtualMemArray()
    # return array

