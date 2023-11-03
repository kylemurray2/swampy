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
date_objects,elevations_medians,elevations_modes, elevations_std = water_elevation.main()
elevations_medians = np.asarray(elevations_medians)
elevations_modes = np.asarray(elevations_modes)
elevations_std = np.asarray(elevations_std)

# Remove outliers
elevations_medians = remove_outliers(elevations_medians,n_sigma=3)
elevations_modes = remove_outliers(elevations_modes,n_sigma=3)

# Get measurements and clean up data
data = pd.read_csv(ps.cdec_levels_csv)
columns_to_convert = ['RES ELE FEET', 'STORAGE AF', 'RES CHG AF', 'OUTFLOW CFS', 'INFLOW CFS', 'PPT INC INCHES', 'SPILL CFS']

for col in columns_to_convert:
    data[col] = clean_convert(data[col])

# Convert dates to datetime
data['Datetime'] = pd.to_datetime(data['DATE / TIME (PST)'])

data['RES ELE FEET'][data['RES ELE FEET']==0] = np.nan
data['RES ELE FEET']*=.3048 # feet to meters


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data['Datetime'], data['RES ELE FEET'], '.')

# # Draw a vertical line where 'Amount Diverted' is nonzero
# for index, row in data.iterrows():
#     if abs(row['Amount Diverted (AF)']) > 500:
#         plt.axvline(x=row['Datetime'], color='green', linestyle='--', alpha=0.7)
# plt.errorbar(x, y, yerr=yerr, fmt='o', color='blue', ecolor='red', capsize=5)

plt.errorbar(date_objects, elevations_medians+5, yerr=elevations_std, fmt='o',label='median',capsize=5,color='green',ecolor='red')
# plt.plot(date_objects, elevations_modes, '.',label='mode')

plt.title('Time Series of Water Levels')
plt.xlabel('Datetime')
plt.ylabel('Measured Value (meters)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()