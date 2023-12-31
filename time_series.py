#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:31:04 2023

@author: km
"""
import rasterio, glob
import matplotlib.pyplot as plt
import numpy as np
from swampy import config

ps = config.getPS()
organize_directories(ps.dataDir)



def load_geotiffs(dataDir):
    """Load geotiff files from a folder and return a stack of images and their dates."""
    # Use glob to find all GeoTIFF files in the folder
    filepaths = sorted(glob.glob(dataDir + '/*.tif'))
    
    dates=[]
    for fn in filepaths:
        dates.append(fn.split('_')[4][0:8])
    
    stacks = []
    # Load each file and append it to the stacks list
    for path in filepaths:
        with rasterio.open(path, 'r') as src:
            stacks.append(src.read(1))
    
    # Convert list of arrays to a 3D numpy array (time, y, x)
    return dates,np.stack(stacks)


def plot_time_series(time_series, labels):
    """Plot the time series."""
    plt.figure()
    plt.plot(labels, time_series, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Pixel Value')
    plt.title('Time Series at a Given Location')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    
    ps = config.getPS()
    # Load geotiffs into a stack
    dates,stack = load_geotiffs(ps.dataDir)
    # Provide x, y coordinates
    x, y = 100, 150
    # Extract time series for the given x,y location
    time_series_data = stack[:, y, x]
    # Plot the time series
    plot_time_series(time_series_data, dates)
