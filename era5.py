#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:27:25 2024

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import cdsapi
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from swampy import config
import matplotlib.collections as mcollections
from SARTS import makeMap
import cartopy.crs as ccrs
import imageio

def download_era5_data(north, west, south, east, start_date, end_date, download_path='downloaded_data.nc'):
    c = cdsapi.Client()
    
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate list of all dates in the range
    date_generated = [start + timedelta(days=x) for x in range(0, (end-start).days + 1)]
    
    # Format dates for the API request
    dates = [date.strftime('%Y-%m-%d') for date in date_generated]
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'total_precipitation',    # Total precipitation
                'evaporation',            # Evaporation
                'volumetric_soil_water_layer_1',  # Soil moisture for the top 7 cm of soil
                'snow_depth_water_equivalent'    # Snow depth water equivalent
            ],
            'year': [date[:4] for date in dates],
            'month': [date[5:7] for date in dates],
            'day': [date[8:10] for date in dates],
            'time': [
                '00:00', '01:00', '02:00', '03:00',
                '04:00', '05:00', '06:00', '07:00',
                '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00',
                '16:00', '17:00', '18:00', '19:00',
                '20:00', '21:00', '22:00', '23:00'
            ],
            'area': [
                north,  # North latitude
                west,   # West longitude
                south,  # South latitude
                east    # East longitude
            ],
            'grid': [0.25, 0.25],  # Grid resolution; adjust as needed
        },
        download_path
    )


def calculate_daily_averages(nc_file):
    ds = xr.open_dataset(nc_file)
    daily = ds.resample(time='D').mean()
    return daily



# Define the longitude and latitude bounds for the west coast of the US
min_longitude = -125
max_longitude = -114
min_latitude = 32
max_latitude = 49

ps = config.getPS()


start_2023 = ps.date_start[0:4] + '-' +  ps.date_start[4:6] + '-' +  ps.date_start[6:8]
start_2024 = '2024-01-01'

end_2023 = '2023-12-31'
end_2024 = datetime.today().strftime('%Y-%m-%d')

fname_2023 = 'era5/2023_data.nc'
fname_2024 = 'era5/2024_data.nc'

if not os.path.isfile(fname_2023):
    download_era5_data(max_latitude,min_longitude, min_latitude,max_longitude, start_2023,end_2023, fname_2023)
if not os.path.isfile(fname_2024):
    download_era5_data(max_latitude,min_longitude, min_latitude,max_longitude, start_2024,end_2024, fname_2024)

daily_averages = calculate_daily_averages(fname_2023)
print(daily_averages)

def combine_datasets(file1, file2, combined_file):
    # Open the datasets
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)
    # Align the second dataset by selecting one version
    ds2 = ds2.sel(expver=1)  # assuming '1' is a valid index or key for expver
    # Ensure that the spatial coordinates match
    if not (all(ds1.latitude == ds2.latitude) and all(ds1.longitude == ds2.longitude)):
        raise ValueError("Latitude and Longitude coordinates do not match.")
    # Concatenate datasets along the time dimension
    combined_ds = xr.concat([ds1, ds2], dim='time')
    # Save combined dataset to a new NetCDF file
    combined_ds.to_netcdf(combined_file)
    return combined_ds



combined_fname = 'era5/combined_2023_2024_data.nc'
ds = combine_datasets(fname_2023, fname_2024, combined_fname)


def plot_era5_data(nc_file):
    # Load the dataset
    ds = xr.open_dataset(nc_file)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle('ERA5 Daily Averages', fontsize=16)
    ds['tp'].mean('time').plot(ax=axes[0, 0], cmap='Blues', cbar_kwargs={'label': 'm'})
    axes[0, 0].set_title('Total Precipitation (m)')
    ds['e'].mean('time').plot(ax=axes[0, 1], cmap='Purples', cbar_kwargs={'label': 'm of water equivalent'})
    axes[0, 1].set_title('Evaporation (m of water equivalent)')
    ds['swvl1'].mean('time').plot(ax=axes[1, 0], cmap='Greens', cbar_kwargs={'label': 'm³/m³'})
    axes[1, 0].set_title('Soil Moisture (m³/m³)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make space for the title
    plt.show()

plot_era5_data(combined_fname)

# Assuming ds is your Dataset and tp is a DataArray within ds
latitudes = ds['latitude'].values
longitudes = ds['longitude'].values

minlat = latitudes.min()
maxlat = latitudes.max()
minlon = longitudes.min()
maxlon = longitudes.max()

def animateERA5(output_fn,variable,fps):
    '''
    Make animated map using an era5 variable like tp
    '''
    writer = imageio.get_writer(output_fn, fps=fps)
    ax,fig,_ = makeMap.mapBackground(minlon, maxlon, minlat, maxlat)  # Assuming this sets up the map
    
    # Loop through each time step to plot data
    for timestep in variable.time:
        # Clear only the data layer, correctly reference QuadMesh from matplotlib.collections
        [artist.remove() for artist in ax.get_children() if isinstance(artist, mcollections.QuadMesh)]
        variable.sel(time=timestep).plot(ax=ax, transform=ccrs.PlateCarree(), x='longitude', y='latitude', 
                                   add_colorbar=False, cmap='Blues', zorder=10)
        ax.set_title(f"Total {variable.attrs['long_name']}: {timestep.dt.strftime('%Y-%m-%d %H:%M').item()}")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(image)
    writer.close()
    plt.close(fig)


tp = ds['tp']
weekly_sum_tp = tp.resample(time='1W').sum()
first_days_of_week = weekly_sum_tp.time.values
weekly_sum_tp = weekly_sum_tp.where(weekly_sum_tp>0.0)
tp_sampled = tp.isel(time=slice(0, 100))
tp_sampled = tp.where(tp_sampled >0)

e = ds['e']
weekly_sum_e = e.resample(time='1W').sum()
first_days_of_week = weekly_sum_e.time.values
weekly_sum_e = weekly_sum_e.where(weekly_sum_e>0.0)
e_sampled = e.isel(time=slice(0, 100))
e_sampled = e.where(e_sampled >0)

swvl1 = ds['swvl1']
weekly_sum_swvl1 = swvl1.resample(time='1W').mean()
first_days_of_week = weekly_sum_swvl1.time.values
weekly_sum_swvl1 = weekly_sum_swvl1.where(weekly_sum_swvl1>0)
swvl1_sampled = swvl1.isel(time=slice(0, 100))
swvl1_sampled = swvl1.where(swvl1_sampled >.001)

output_fn_tp = 'era5/precipitation_animation.gif'
output_fn_e = 'era5/evaporation_animation.gif'
output_fn_swvl1 = 'era5/soil_moisture_animation.gif'

fps = 12

animateERA5(output_fn_tp,tp_sampled,fps)
animateERA5(output_fn_e,e_sampled,fps)
animateERA5(output_fn_swvl1,swvl1_sampled,fps)
