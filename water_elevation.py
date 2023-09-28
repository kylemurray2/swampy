#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:12:46 2023

Read the DSW image and DEM rasters.
Create a binary mask from the DSW raster where there is water.
For the border of the DSW, use a convolution with a 3x3 filter to find the 
edges.
Using the binary edge mask, extract elevation values from the DEM where the 
mask is True.
Calculate the average or median of these elevation values, which will give you 
an approximation of the water surface elevation.

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import rasterio
from scipy.signal import convolve2d

def extract_water_edge_elevation(dsw_path, dem_path):
    # Open the DSW raster
    with rasterio.open(dsw_path, 'r') as src:
        dsw = src.read(1)
        dsw_transform = src.transform

    # Open the DEM raster
    with rasterio.open(dem_path, 'r') as src:
        dem = src.read(1)
        dem_transform = src.transform

    # Ensure they have the same transform (they cover the same area and have the same resolution)
    assert dsw_transform == dem_transform, "DSW and DEM rasters have different transforms!"

    # Create a binary mask for DSW == 1
    water_mask = dsw == 1

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

    # Calculate the median elevation of the water's edge
    median_elevation = np.median(water_edge_elevations)

    return median_elevation

dsw_path = "path_to_DSW_raster.tif"
dem_path = "path_to_DEM_raster.tif"

water_surface_elevation = extract_water_edge_elevation(dsw_path, dem_path)
print(f"The estimated water surface elevation is: {water_surface_elevation:.2f} meters (or relevant units)")
