#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:06:50 2023

Sorts tif files into YYYYMMDD/EPSG directories
Stitches all images from each date together
Outputs mosaic.tif in each date directory

Keep track of the date and time of each Opera file that comprises a stitched image

@author: km
"""

import rasterio, os, glob
import numpy as np
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from swampy import config
from itertools import repeat
from datetime import datetime, timedelta

def organize_by_crs(crs, file_list,output_path):
    current_output_path = output_path/crs
    if not current_output_path.exists():
        current_output_path.mkdir()
    
    for f in file_list:
        f.rename(current_output_path/f.name)
        
def organize_files(ps,dataDir):
    dataDir = Path(dataDir)   
    # Organize and mosaic granules
    files_by_crs = defaultdict(list)
    for f in [f for f in dataDir.iterdir() if f.is_dir()]:
        files_by_crs[f.name] = list(f.glob("OPERA*_WTR.tif"))
        
    # Organize downloaded into folders by CRS 
    for i, f in enumerate(list(dataDir.glob('OPERA*.tif'))):
        with rasterio.open(f) as ds:
            files_by_crs[ds.profile['crs'].to_string()].append(f)
        
    _ = list(map(organize_by_crs, files_by_crs.keys(), files_by_crs.values(),repeat(dataDir)))

    return files_by_crs

from datetime import datetime

# Define a reference date (epoch) from which to calculate minutes
epoch = datetime(2023, 1, 1)

def acquisition_time_to_minutes(acquisition_time_str):
    """
    Convert acquisition time string to minutes since the epoch.
    """
    acquisition_time_str = acquisition_time_str.replace('Z', '')  # Remove the trailing 'Z'
    acquisition_time = datetime.strptime(acquisition_time_str, "%Y%m%dT%H%M%S")
    return int((acquisition_time - epoch).total_seconds() / 60)

def minutes_to_acquisition_time(minutes_since_epoch):
    """
    Convert minutes since the epoch to acquisition time string.
    """
    epoch = datetime(2023, 1, 1)
    acquisition_time = epoch + timedelta(minutes=minutes_since_epoch)
    acquisition_time_str = acquisition_time.strftime("%Y%m%dT%H%M%S") + 'Z'
    return acquisition_time_str


def merge_with_tracking(file_batch):
    """
    Merge the list of files and track the source file acquisition time for each pixel.
    """
    # Initialize with the first file
    with rasterio.open(file_batch[0]) as src:
        merged_img = src.read(1)
        merged_transform = src.transform
        merged_crs = src.crs

    # Initialize the tracking array with the first file's acquisition time


    acquisition_time_str = str(file_batch[0]).split('/')[-1].split('_')[4]
    initial_time = acquisition_time_to_minutes(acquisition_time_str)
    file_index_array = np.full_like(merged_img, initial_time, dtype=np.int32)

    # Process each file in the batch
    for file in file_batch[1:]:  # Start from the second file
        with rasterio.open(file) as src:
            src_data = src.read(1)
            src_transform = src.transform

            # Extract acquisition time from the file name and convert to minutes since epoch
            acquisition_time_str = str(file).split('/')[-1].split('_')[4]
            acquisition_time_minutes = acquisition_time_to_minutes(acquisition_time_str)

            # Create an empty array for reprojected source data
            src_reprojected = np.empty_like(merged_img)

            # Reproject source data to the merged image's CRS and transform
            reproject(
                source=src_data,
                destination=src_reprojected,
                src_transform=src_transform,
                dst_transform=merged_transform,
                src_crs=src.crs,
                dst_crs=merged_crs,
                resampling=Resampling.nearest
            )

            # Update the merged image and the tracking array based on the minimum value
            min_mask = (src_reprojected < merged_img) & (src_reprojected > 0)
            merged_img[min_mask] = src_reprojected[min_mask]
            file_index_array[min_mask] = acquisition_time_minutes

    return merged_img, merged_transform, file_index_array

def reprojectDSWx_mosaics(epsg_code, file_batch, output_filename, dswx_colormap, resolution_reduction_factor = 1):
    '''
    Takes a list of files in the same CRS and mosaic them, and then reproject 
    it to EPSG:4326.
    '''
    dst_crs = 'EPSG:4326'
    merged_img, merged_transform = merge(file_batch, method='min')
    merged_output_bounds = rasterio.transform.array_bounds(merged_img.shape[-2], merged_img.shape[-1] , merged_transform)

    kwargs = {
        "src_crs": epsg_code, 
        "dst_crs": dst_crs, 
        "width":merged_img.shape[-1], 
        "height": merged_img.shape[-2], 
        "left": merged_output_bounds[0],
        "bottom": merged_output_bounds[1],
        "right": merged_output_bounds[2],
        "top": merged_output_bounds[3],
        "dst_width": merged_img.shape[-1]//resolution_reduction_factor, 
        "dst_height":merged_img.shape[-2]//resolution_reduction_factor  
    }
    
    dst_transform, width, height = calculate_default_transform(**kwargs)

    with rasterio.open(file_batch[0]) as src:
        dst_kwargs = src.profile.copy()
        dst_kwargs.update({
            'height':height,
            'width':width,
            'transform':dst_transform,
            'crs':dst_crs
        })
        
        with rasterio.open(output_filename, 'w', **dst_kwargs) as dst:
            reproject(
                source = merged_img, 
                destination = rasterio.band(dst, 1), 
                src_transform = merged_transform,
                dst_transform = dst_transform,
                src_crs = src.crs,
                dst_crs = dst_crs,
                resampling=Resampling.nearest
            )
            dst.write_colormap(1, dswx_colormap)

    return output_filename

def reprojectDSWx(epsg_code, file_batch, output_filename, dswx_colormap, resolution_reduction_factor=1):
    '''
    Takes a list of files in the same CRS and mosaic them, and then reproject 
    it to EPSG:4326, while keeping track of acquisition times.
    '''
    dst_crs = 'EPSG:4326'

    # # Merge images and get the tracking array
    # if file_batch[0].split('/')[-1] == 'mosaic.tif':
    #     stmo = True
    # else:
    #     stmo=False
        
    merged_img, merged_transform, file_index_array = merge_with_tracking(file_batch)
    merged_output_bounds = rasterio.transform.array_bounds(merged_img.shape[-2], merged_img.shape[-1], merged_transform)

    kwargs = {
        "src_crs": epsg_code,
        "dst_crs": dst_crs,
        "width": merged_img.shape[-1],
        "height": merged_img.shape[-2],
        "left": merged_output_bounds[0],
        "bottom": merged_output_bounds[1],
        "right": merged_output_bounds[2],
        "top": merged_output_bounds[3],
        "dst_width": merged_img.shape[-1] // resolution_reduction_factor,
        "dst_height": merged_img.shape[-2] // resolution_reduction_factor
    }

    dst_transform, width, height = calculate_default_transform(**kwargs)

    with rasterio.open(file_batch[0]) as src:
        dst_kwargs = src.profile.copy()
        dst_kwargs.update({
            'height': height,
            'width': width,
            'transform': dst_transform,
            'crs': dst_crs,
            'count': src.count + 1,  # Adding an extra band for acquisition time
            'dtype': 'int32',  # Ensure the data type can hold acquisition times
            'driver': 'GTiff'  # Ensure the correct driver is used
        })
        
        with rasterio.open(output_filename, 'w', **dst_kwargs) as dst:
            # Reproject and write the merged image
            reproject(
                source=merged_img,
                destination=rasterio.band(dst, 1),
                src_transform=merged_transform,
                dst_transform=dst_transform,
                src_crs=epsg_code,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            dst.write_colormap(1, dswx_colormap)
            print(np.unique(file_index_array))
            # Reproject and write the tracking array as the second band
            tracking_reprojected = np.empty((height, width), dtype=np.int32)
            reproject(
                source=file_index_array,
                destination=tracking_reprojected,
                src_transform=merged_transform,
                dst_transform=dst_transform,
                src_crs=epsg_code,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            dst.write(tracking_reprojected, 2)

    return output_filename


# plt.figure();plt.imshow(file_index_array)
# plt.figure();plt.imshow(merged_img[0,:,:])
# plt.figure();plt.imshow(file_index_array)
# plt.figure();plt.imshow(src_reprojected)

# file_batch = [
#     'data/20230526/EPSG:32609/OPERA_L3_DSWx-HLS_T09UYP_20230526T190655Z_20230528T142338Z_L8_30_v1.0_B01_WTR.tif',
#     'data/20230526/EPSG:32609/OPERA_L3_DSWx-HLS_T09UYP_20230526T192909Z_20230528T155519Z_S2B_30_v1.0_B01_WTR.tif'
# ]
# output_filename = 'data/20230526/EPSG:32609/mosaic.tif'
# epsg_code = 'EPSG:32609'

# with rasterio.open(file_batch[0]) as ds:
#     dswx_colormap = ds.colormap(1)

# reprojectDSWx(epsg_code, file_batch, output_filename, dswx_colormap)


# with rasterio.open(output_filename) as src:
#     data = src.read(1)
#     acquisition_times = src.read(2)

# plt.figure()
# plt.imshow(acquisition_times)
# print(np.unique(acquisition_times))

# minutes_to_acquisition_time(209969)






def main():
    ps = config.getPS()
        
    date_dirs = glob.glob(ps.dataDir + '/2???????')
    date_dirs.sort()
    dates=[]
    for date_fn in date_dirs:
        dates.append(date_fn.split('/')[-1])
    
    
    for date_dir in date_dirs:
           
        files_by_crs = organize_files(ps,date_dir)
        # Get a list of the tif files
        file_list = glob.glob(os.path.join(date_dir, 'EPSG*','*_WTR.tif'))
        final_mosaic_path =Path(date_dir)
        if not final_mosaic_path.exists():
            final_mosaic_path.mkdir()
            
        if len(file_list)==0:
            print('No tif files found in the date directory ' + date_dir)
            
        elif len(file_list)==1:    
                # If there is only one, then just link to the original file as the output
                source_name = glob.glob(date_dir + '/EPSG*/O*tif')
                source_name_abs = os.path.abspath(source_name[0])
                link_name = os.path.join(str(final_mosaic_path),'mosaic1.tif')
                
                if not os.path.isfile(link_name):
                    os.symlink(source_name_abs, link_name)
                else:
                    print('Link already exists for ' + link_name)
                    
        else:
            if os.path.isfile( os.path.join(str(final_mosaic_path),'mosaic.tif')):
                print('Final stitched image already exists for ' + str(final_mosaic_path))
            else:           
                print('Reprojecting and Stitching ' + date_dir)
                # Get a colormap from one of the files
                with rasterio.open(file_list[0]) as ds:
                    dswx_colormap = ds.colormap(1)
                    
                output_path = Path(date_dir)
                
                resolution_reduction_factor = 1
                for key in files_by_crs.keys():
                    mosaic_folder = (output_path/key/'mosaics')
                    mosaic_folder.mkdir(parents=True, exist_ok=True)
                    filenames = list((output_path/key).glob('O*.tif'))
                    output_filename = 'temp_{}_{}.tif' #os.path.join(output_path, key,'mosaic.tif')

                    if len(filenames)> 10:
                        nchunks = 4
                        filename_chunks = np.array_split(filenames, nchunks)

                        function_inputs = []
                        count = 0
                        for chunk in filename_chunks:
                            if len(chunk) > 0:
                                input_tuple = (key, chunk, mosaic_folder/output_filename.format(key, str(count).zfill(4)), dswx_colormap, resolution_reduction_factor)
                                function_inputs.append(input_tuple)
                                count += 1
                                
                        with Pool() as pool:
                            output_files = pool.starmap(reprojectDSWx, function_inputs)
        
                    else:                    
                        reprojectDSWx(key, filenames, mosaic_folder/output_filename.format(key, 'all'), dswx_colormap)        
                    
                # Now stitch together all of the different EPSG mosaics
                mosaic_list = glob.glob(os.path.join(date_dir,'EPSG*','mosaics','*tif'))
                reprojectDSWx_mosaics('EPSG:4326', mosaic_list, Path(final_mosaic_path / 'mosaic.tif'),dswx_colormap)

if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

