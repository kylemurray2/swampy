#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:06:50 2023

Sorts tif files into YYYYMMDD/EPSG directories
Stitches all images from each date together
Outputs mosaic_[type].tif files in each date directory

@author: km
"""

import rasterio
import os
import glob
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from collections import defaultdict
from pathlib import Path
import config
import gc

os.chdir('/Volumes/NAS_NC/haw/Documents/research/surfaceWater/westCoastData')

def check_small_files(data_dir, min_size_bytes=1024):
    """Check for and delete files smaller than min_size_bytes (default 1KB)"""
    small_files = []
    data_dir = Path(data_dir)
    
    for tif_file in data_dir.rglob('*.tif'):
        if tif_file.stat().st_size < min_size_bytes:
            small_files.append((tif_file, tif_file.stat().st_size))
            tif_file.unlink()
            
    return small_files

def organize_files(date_dir):
    """Organize files by CRS and type"""
    date_dir = Path(date_dir)   
    files_by_crs = defaultdict(lambda: {'WTR': [], 'WTR-2': [], 'CONF': []})
    
    # Handle files in the main directory
    for f in date_dir.glob('OPERA*.tif'):
        try:
            with rasterio.open(f) as ds:
                crs = ds.profile['crs'].to_string()
                
                # Determine file type
                if '_WTR.' in f.name:
                    file_type = 'WTR'
                elif '_WTR-2.' in f.name:
                    file_type = 'WTR-2'
                elif '_CONF.' in f.name:
                    file_type = 'CONF'
                else:
                    continue
                
                # Create EPSG directory and move file
                current_output_path = date_dir/crs
                current_output_path.mkdir(exist_ok=True)
                new_path = current_output_path/f.name
                f.rename(new_path)
                files_by_crs[crs][file_type].append(new_path)
                
        except Exception as e:
            print(f"Warning: Error processing file {f}: {str(e)}")
            continue

    # Add any files already in EPSG directories
    for epsg_dir in [d for d in date_dir.iterdir() if d.is_dir() and 'EPSG' in d.name]:
        crs = epsg_dir.name
        for file_type in ['WTR', 'WTR-2', 'CONF']:
            pattern = f"OPERA*_{file_type}.tif"
            files_by_crs[crs][file_type].extend(list(epsg_dir.glob(pattern)))

    return files_by_crs

def reprojectDSWx(epsg_code, file_batch, output_filename, colormap=None, bounds=None, dst_transform=None, dst_width=None, dst_height=None):
    """Reproject and merge a batch of files"""
    if not file_batch:
        return None
        
    dst_crs = 'EPSG:4326'
    
    # Get bounds and validate files
    bounds = bounds or None
    valid_files = []
    for file in file_batch:
        try:
            with rasterio.open(file) as src:
                if src.width > 0 and src.height > 0:
                    if bounds is None:
                        bounds = src.bounds
                    else:
                        bounds = (
                            min(bounds[0], src.bounds[0]),
                            min(bounds[1], src.bounds[1]),
                            max(bounds[2], src.bounds[2]),
                            max(bounds[3], src.bounds[3])
                        )
                    valid_files.append(file)
        except Exception as e:
            print(f"Warning: Error reading file {file}: {str(e)}")
            
    if not valid_files:
        return None

    # Merge files
    merged_img, merged_transform = merge(valid_files, method='min')
    
    with rasterio.open(valid_files[0]) as src:
        dst_transform, width, height = calculate_default_transform(
            src_crs=epsg_code,
            dst_crs=dst_crs,
            width=merged_img.shape[-1],
            height=merged_img.shape[-2],
            left=bounds[0],
            bottom=bounds[1],
            right=bounds[2],
            top=bounds[3]
        )

        dst_kwargs = src.profile.copy()
        dst_kwargs.update({
            'height': height,
            'width': width,
            'transform': dst_transform,
            'crs': dst_crs,
            'count': 1
        })
        
        with rasterio.open(output_filename, 'w', **dst_kwargs) as dst:
            reproject(
                source=merged_img,
                destination=rasterio.band(dst, 1),
                src_transform=merged_transform,
                dst_transform=dst_transform,
                src_crs=epsg_code,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            if colormap:
                dst.write_colormap(1, colormap)

    return output_filename

def process_date_directory(date_dir):
    """Process a single date directory"""
    print(f"\nProcessing directory: {date_dir}")
    
    # Organize files by CRS and type
    files_by_crs = organize_files(date_dir)
    if not files_by_crs:
        print(f"No files found in {date_dir}")
        return

    # Process each CRS
    for crs, files_by_type in files_by_crs.items():
        print(f"Processing CRS: {crs}")
        mosaic_folder = Path(date_dir)/crs/'mosaics'
        mosaic_folder.mkdir(parents=True, exist_ok=True)
        
        # Process each layer type independently
        for layer_type in ['WTR', 'WTR-2', 'CONF']:
            files = files_by_type[layer_type]
            if not files:
                print(f"No {layer_type} files found in {crs}")
                continue
            
            # Calculate bounds for this layer type
            layer_bounds = None
            for file in files:
                try:
                    with rasterio.open(file) as src:
                        if src.width > 0 and src.height > 0:
                            if layer_bounds is None:
                                layer_bounds = src.bounds
                            else:
                                layer_bounds = (
                                    min(layer_bounds[0], src.bounds[0]),  # left
                                    min(layer_bounds[1], src.bounds[1]),  # bottom
                                    max(layer_bounds[2], src.bounds[2]),  # right
                                    max(layer_bounds[3], src.bounds[3])   # top
                                )
                except Exception as e:
                    print(f"Warning: Error reading file {file}: {str(e)}")
                    continue

            if layer_bounds is None:
                print(f"No valid bounds found for {layer_type}")
                continue

            print(f"Processing {len(files)} {layer_type} files")
            print(f"{layer_type} bounds: {layer_bounds}")
            
            # Calculate transform for this layer
            with rasterio.open(files[0]) as src:
                dst_transform, width, height = calculate_default_transform(
                    src_crs=crs,
                    dst_crs='EPSG:4326',
                    width=src.width,
                    height=src.height,
                    left=layer_bounds[0],
                    bottom=layer_bounds[1],
                    right=layer_bounds[2],
                    top=layer_bounds[3]
                )
            
            # Process in batches
            batch_size = 4
            output_files = []
            
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                output_file = mosaic_folder/f'temp_{crs}_{layer_type}_{i//batch_size}.tif'
                
                # Get colormap from first file if WTR
                colormap = None
                if layer_type == 'WTR':
                    with rasterio.open(batch[0]) as src:
                        colormap = src.colormap(1)
                
                result = reprojectDSWx(crs, batch, output_file, colormap, 
                                     bounds=layer_bounds, dst_transform=dst_transform,
                                     dst_width=width, dst_height=height)
                if result:
                    output_files.append(result)
                gc.collect()
            
            # Create final mosaic for this type
            if output_files:
                final_output = Path(date_dir)/f'mosaic_{layer_type.lower()}.tif'
                print(f"Creating final {layer_type} mosaic...")
                
                with rasterio.open(output_files[0]) as src:
                    profile = src.profile.copy()
                    colormap = src.colormap(1) if layer_type == 'WTR' else None
                    
                    with rasterio.open(final_output, 'w', **profile) as dst:
                        data, _ = merge(output_files, method='min')
                        dst.write(data[0], 1)
                        if colormap:
                            dst.write_colormap(1, colormap)
                        del data
                        gc.collect()
                
                print(f"{layer_type} mosaic created successfully")

def main():
    try:
        ps = config.getPS()
        date_dirs = sorted(glob.glob(os.path.join(ps.dataDir, '2???????')))
        
        if not date_dirs:
            print("No date directories found in", ps.dataDir)
            return
            
        print(f"Found {len(date_dirs)} date directories to process")
        
        # Check for and remove small files
        small_files_found = []
        for date_dir in date_dirs:
            small_files = check_small_files(date_dir)
            if small_files:
                small_files_found.extend(small_files)
        
        if small_files_found:
            print("\nWarning: Found and deleted the following small files (<1KB):")
            for f, size in small_files_found:
                print(f"  - {f} ({size} bytes)")
            return
        
        # Process each date directory
        for date_dir in date_dirs:
            try:
                process_date_directory(date_dir)
            except Exception as e:
                print(f"Error processing directory {date_dir}: {str(e)}")
                continue

    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()

