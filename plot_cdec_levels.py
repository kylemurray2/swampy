#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:44:05 2023

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from swampy import water_elevation,config


def remove_outliers(data,n_sigma=3):
    '''
    remove values > mean(data) +/- n_sigma 
    '''
    mean = np.mean(data)
    std = np.std(data)
    mask = (data > mean - n_sigma*std) & (data < mean + 3*std)
    filtered_data = data[mask]
    return filtered_data


def clean_convert(column):
    '''
    Remove special characters, Replace '--' with NaN, convert to float32
    '''
    column = column.str.replace('\xa0', '').str.replace(',', '')
    column = column.replace('--', np.nan)
    column = pd.to_numeric(column, errors='coerce')
    return column.astype(np.float32)


ps = config.getPS()

if os.path.isfile('elevation_data_dict.npy'):
    date_objects,elevations_medians,elevations_modes, elevations_std,DEM_water_elevation = water_elevation.main()
    elevations_medians = np.asarray(elevations_medians)
    elevations_modes = np.asarray(elevations_modes)
    elevations_std = np.asarray(elevations_std)
    elevation_data_dict = {}
    elevation_data_dict['date_objects'] = date_objects
    elevation_data_dict['elevations_medians'] = elevations_medians
    elevation_data_dict['elevations_modes'] = elevations_modes
    elevation_data_dict['elevations_std'] = elevations_std
    elevation_data_dict['DEM_water_elevation'] = DEM_water_elevation
    np.save('elevation_data_dict.npy',elevation_data_dict)
else:
    elevation_data_dict = np.load('elevation_data_dict.npy')

# Remove outliers
elevations_medians_clean = remove_outliers(elevations_medians,n_sigma=3)
elevations_modes_clean = remove_outliers(elevations_modes,n_sigma=3)

# Get measurements and clean up data
data = pd.read_csv(ps.cdec_levels_csv)
columns_to_convert = ['RES ELE FEET', 'STORAGE AF', 'RES CHG AF', 'OUTFLOW CFS', 'INFLOW CFS', 'PPT INC INCHES', 'SPILL CFS']

for col in columns_to_convert:
    data[col] = clean_convert(data[col])

# Convert dates to datetime
data['Datetime'] = pd.to_datetime(data['DATE / TIME (PST)'])

data['RES ELE FEET'][data['RES ELE FEET']==0] = np.nan
data['RES ELE FEET']*=.3048 # feet to meters

offset = 4.6
offset_mode = 2

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data['Datetime'], data['RES ELE FEET'], '.')
plt.errorbar(elevation_data_dict['date_objects'], elevation_data_dict['elevations_medians']+offset, yerr=elevation_data_dict['elevations_std'], fmt='o',label='median',capsize=5,color='green',ecolor='red')
# plt.errorbar(elevation_data_dict['date_objects'], elevation_data_dict['elevations_modes']+offset_mode, yerr=elevation_data_dict['elevations_std'], fmt='o',label='mode',capsize=5,color='orange',ecolor='red')
plt.axhline(y=DEM_water_elevation,linestyle='-')
plt.title('Time Series of Water Levels')
plt.xlabel('Datetime')
plt.ylabel('Measured Value (meters)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()