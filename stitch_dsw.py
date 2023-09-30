#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:06:50 2023



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
    for i, f in enumerate(list(dataDir.glob('*.tif'))):
        with rasterio.open(f) as ds:
            files_by_crs[ds.profile['crs'].to_string()].append(f)
    

    
    _ = list(map(organize_by_crs, files_by_crs.keys(), files_by_crs.values(),repeat(dataDir)))

    return files_by_crs


def reprojectDSWx(epsg_code, file_batch, output_filename, dswx_colormap, resolution_reduction_factor = 1):
    '''
    Takes a list of files in the same CRS and mosaic them, and then reproject 
    it to EPSG:4326.
    '''
    dst_crs = 'EPSG:4326'
    
    # Mosaic the file_batch 
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


def main():
    ps = config.getPS()
        

    
    # os.system('mv ' + ps.dataDir + '/EPSG*/* ./ ' + ps.dataDir + '/')
    # os.system('rm -r ' + ps.dataDir + '/EPSG*')
    # os.system('rm -r ' + ps.dataDir + '/outputs')

    # First, organize the files into CRS dirs and return files by crs dict
    
    
    date_dirs = glob.glob(ps.dataDir + '/2???????')
    date_dirs.sort()
    dates=[]
    for date_fn in date_dirs:
        dates.append(date_fn.split('/')[-1])
    
    
    for date_dir in date_dirs:
    
        files_by_crs = organize_files(ps,date_dir)
        # Get a list of the tif files
        file_list = glob.glob(os.path.join(ps.dataDir,'????????', 'EPSG*','*_WTR.tif'))
       
        if len(file_list)==0:
            print('No tif files found in the date directory ' + date_dir)
        else:
            # Get a colormap from one of the files
            with rasterio.open(file_list[0]) as ds:
                dswx_colormap = ds.colormap(1)
                
            output_path = Path(date_dir)
            
            resolution_reduction_factor = 1
            nchunks = 10
            for key in files_by_crs.keys():
                mosaic_folder = (output_path/key/'mosaics')
                mosaic_folder.mkdir(parents=True, exist_ok=True)
                filenames = list((output_path/key).glob('*.tif'))
                filename_chunks = np.array_split(filenames, nchunks)
                
                output_filename = 'temp_{}_{}.tif'
                function_inputs = []
                count = 0
                for chunk in filename_chunks:
                    if len(chunk) > 0:
                        input_tuple = (key, chunk, mosaic_folder/output_filename.format(key, str(count).zfill(4)), dswx_colormap, resolution_reduction_factor)
                        function_inputs.append(input_tuple)
                        count += 1
                        
                with Pool() as pool:
                    output_files = pool.starmap(reprojectDSWx, function_inputs)# 
                    
            
            # Now get the mosaic chunks from outdir/EPSG:????/mosaics/*tif and merge those:    
            mosaic_list = glob.glob(os.path.join(date_dir,'EPSG*','mosaics','*tif'))
            
            final_mosaic_path =Path(date_dir + '/outputs')
            if not final_mosaic_path.exists():
                final_mosaic_path.mkdir()
            
            if len(mosaic_list)>1:
                # If there are multiple mosaic files, then merge them
                reprojectDSWx('EPSG:4326', mosaic_list, Path(final_mosaic_path / 'final_mosaic.tif'),dswx_colormap)
            else:
                # If there is only one, then just link to the original file as the output
                source_name = glob.glob(date_dir + '/EPSG*/*tif')
                source_name_abs = os.path.abspath(source_name[0])
                if len(source_name) > 1:
                    print("Something went wrong")
                else:
                    link_name = os.path.join(str(final_mosaic_path),'final_mosaic.tif')
                    os.symlink(source_name_abs, link_name)
