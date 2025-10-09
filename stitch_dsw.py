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
import multiprocessing as mp
from functools import partial
import numpy as np
import math
from pyproj import Transformer
from rasterio.transform import from_bounds, array_bounds


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
    """
    Reproject and merge a batch of files to EPSG:4326 while preserving approximate native ground resolution.
    """
    if not file_batch:
        return None

    dst_crs = 'EPSG:4326'

    # 1. Collect valid files and determine overall bounding box in source CRS
    valid_files = []
    src_res_x = None
    src_res_y = None

    if bounds is None:
        bounds_agg = None
    else:
        bounds_agg = list(bounds)

    for f in file_batch:
        try:
            with rasterio.open(f) as src:
                if src.width == 0 or src.height == 0:
                    continue
                # Capture the first file's resolution
                if src_res_x is None and src_res_y is None:
                    src_res_x, src_res_y = src.res

                # Update overall bounding box in source CRS
                if bounds_agg is None:
                    bounds_agg = list(src.bounds)
                else:
                    bounds_agg[0] = min(bounds_agg[0], src.bounds.left)
                    bounds_agg[1] = min(bounds_agg[1], src.bounds.bottom)
                    bounds_agg[2] = max(bounds_agg[2], src.bounds.right)
                    bounds_agg[3] = max(bounds_agg[3], src.bounds.top)

                valid_files.append(f)
        except Exception as e:
            print(f"Warning: Error reading file {f}: {str(e)}")

    if not valid_files:
        return None

    # 2. Merge in the original CRS (no reprojection yet)
    merged_img, merged_transform = merge(valid_files, method='min')

    # If we never got a resolution from the first file, just bail
    if src_res_x is None or src_res_y is None:
        print("No valid resolution found; skipping reproject.")
        return None

    # 3. Approximate the native resolution in degrees (if original is not already EPSG:4326)
    #    We do this by transforming the center of the bounding box from the source CRS -> EPSG:4326
    left, bottom, right, top = bounds_agg
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (bottom + top)

    # Create a transformer from source EPSG to EPSG:4326
    transformer = Transformer.from_crs(epsg_code, dst_crs, always_xy=True)
    center_lon, center_lat = transformer.transform(center_x, center_y)

    # Approximate: 1 degree lat ~ 111320 m; 1 degree lon ~ 111320 * cos(lat) m
    # so we convert the original x resolution from meters -> degrees at center_lat
    # (this means "preserve ground resolution" near that latitude).
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))

    # If the source CRS is already lat/lon, then this step just reuses src_res_x/y directly.
    # Otherwise, do approximate conversion:
    if epsg_code.upper().startswith("EPSG:4326"):
        xres_degrees, yres_degrees = src_res_x, src_res_y
    else:
        xres_degrees = src_res_x / meters_per_degree_lon
        yres_degrees = src_res_y / meters_per_degree_lat

    # 4. Use calculate_default_transform but override the resolution
    with rasterio.open(valid_files[0]) as src_ds:
        dst_transform, width, height = calculate_default_transform(
            src_crs=epsg_code,
            dst_crs=dst_crs,
            width=merged_img.shape[-1],
            height=merged_img.shape[-2],
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            resolution=(xres_degrees, yres_degrees)  # Force preservation of ground resolution
        )

        # 5. Write out the reprojected mosaic
        dst_kwargs = src_ds.profile.copy()
        dst_kwargs.update({
            'height': height,
            'width': width,
            'transform': dst_transform,
            'crs': dst_crs,
            'count': 1,
            'nodata': src_ds.nodata
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

def process_batch(batch_info):
    """Process a single batch of files (to be run in parallel)"""
    files, crs, layer_type, output_file, layer_bounds, dst_transform, width, height = batch_info
    
    colormap = None
    if layer_type == 'WTR':
        with rasterio.open(files[0]) as src:
            colormap = src.colormap(1)
    
    return reprojectDSWx(crs, files, output_file, colormap, 
                        bounds=layer_bounds, dst_transform=dst_transform,
                        dst_width=width, dst_height=height)

def process_date_directory(date_dir):
    """Process a single date directory"""
    print(f"\nProcessing directory: {date_dir}")
    
    files_by_crs = organize_files(date_dir)
    if not files_by_crs:
        print(f"No files found in {date_dir}")
        return

    # Process each CRS
    for crs, files_by_type in files_by_crs.items():
        print(f"Processing CRS: {crs}")
        mosaic_folder = Path(date_dir)/crs/'mosaics'
        mosaic_folder.mkdir(parents=True, exist_ok=True)
        
        for layer_type in ['WTR', 'WTR-2', 'CONF']:
            files = files_by_type[layer_type]
            if not files:
                continue
            
            # Calculate bounds once for all files
            layer_bounds = None
            with rasterio.Env():
                for file in files:
                    try:
                        with rasterio.open(file) as src:
                            bounds = src.bounds
                            layer_bounds = bounds if layer_bounds is None else (
                                min(layer_bounds[0], bounds[0]),
                                min(layer_bounds[1], bounds[1]),
                                max(layer_bounds[2], bounds[2]),
                                max(layer_bounds[3], bounds[3])
                            )
                    except Exception as e:
                        print(f"Warning: Error reading file {file}: {str(e)}")

            if layer_bounds is None:
                continue

            # Calculate transform once
            with rasterio.open(files[0]) as src:
                dst_transform, width, height = calculate_default_transform(
                    src_crs=crs, dst_crs='EPSG:4326',
                    width=src.width, height=src.height,
                    left=layer_bounds[0], bottom=layer_bounds[1],
                    right=layer_bounds[2], top=layer_bounds[3]
                )
            
            # Prepare batches for parallel processing
            batch_size = max(1, len(files) // mp.cpu_count())
            batches = []
            
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                output_file = mosaic_folder/f'temp_{crs}_{layer_type}_{i//batch_size}.tif'
                batches.append((
                    batch, crs, layer_type, output_file, 
                    layer_bounds, dst_transform, width, height
                ))
            
            # Process batches in parallel
            with mp.Pool() as pool:
                output_files = []
                for result in pool.imap_unordered(process_batch, batches):
                    if result is not None:
                        output_files.append(result)
                    # Optional: Add progress indication
                    print(".", end="", flush=True)
                print()  # New line after progress dots
            
            # Create final mosaic
            if output_files:
                final_output = Path(date_dir)/f'mosaic_{layer_type.lower()}.tif'
                print(f"Creating final {layer_type} mosaic...")
                
                with rasterio.Env():
                    with rasterio.open(output_files[0]) as src:
                        profile = src.profile.copy()
                        colormap = src.colormap(1) if layer_type == 'WTR' else None
                        
                        with rasterio.open(final_output, 'w', **profile) as dst:
                            data, _ = merge(output_files, method='min')
                            dst.write(data[0], 1)
                            if colormap:
                                dst.write_colormap(1, colormap)
                            del data
                
                # Clean up temporary files
                for temp_file in output_files:
                    Path(temp_file).unlink()


def stitch_date_range(start_date, end_date, data_dir, layer_type='wtr'):
    """
    Stitch together mosaics from a range of dates
    Args:
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
        data_dir (str): Base directory containing date folders
        layer_type (str): Type of layer to stitch ('wtr', 'wtr-2', or 'conf')
    """
    # Convert dates to integers for comparison
    start = int(start_date)
    end = int(end_date)
    
    # Find all date directories within range
    date_dirs = []
    for date_dir in sorted(glob.glob(os.path.join(data_dir, '2???????'))):
        date = int(os.path.basename(date_dir))
        if start <= date <= end:
            date_dirs.append(date_dir)
    
    if not date_dirs:
        print(f"No date directories found between {start_date} and {end_date}")
        return None

    # Collect all mosaic files for the specified layer type
    mosaic_files = []
    layer_type = layer_type.lower()
    for date_dir in date_dirs:
        mosaic_file = os.path.join(date_dir, f'mosaic_{layer_type}.tif')
        if os.path.exists(mosaic_file):
            mosaic_files.append(mosaic_file)

    if not mosaic_files:
        print(f"No mosaic files found for layer type '{layer_type}' in the date range")
        return None

    print(f"Stitching {len(mosaic_files)} mosaic files...")
    
    # Create output filename
    output_file = os.path.join(data_dir, f'mosaic_{start_date}_to_{end_date}_{layer_type}.tif')
    
    # Get the source files metadata
    src_files_to_mosaic = []
    for f in mosaic_files:
        src = rasterio.open(f)
        if src.crs != 'EPSG:4326':
            print(f"Warning: File {f} is not in EPSG:4326. Skipping...")
            src.close()
            continue
        src_files_to_mosaic.append(src)
    
    if not src_files_to_mosaic:
        print("No valid files found in EPSG:4326")
        return None
    
    # Get the source resolution from the first file
    with rasterio.open(mosaic_files[0]) as src:
        res = src.res
        print(f"Source resolution: {res}")
    
    # Merge settings
    merge_kwargs = {
        'method': 'min',
        'res': res,  # Use the original resolution
        'precision': 10
    }
    
    try:
        # Merge all files and get the correct transform
        mosaic, out_trans = merge(src_files_to_mosaic, **merge_kwargs)
        
        # Get metadata from first file
        with rasterio.open(mosaic_files[0]) as src:
            profile = src.profile.copy()
            colormap = src.colormap(1) if layer_type == 'wtr' else None
        
        # Update the profile with merged data specifications
        profile.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_trans,
            'crs': 'EPSG:4326'
        })
        
        # Write the merged result
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(mosaic)
            if colormap:
                dst.write_colormap(1, colormap)
    
    finally:
        # Clean up
        for src in src_files_to_mosaic:
            src.close()
    
    print(f"Created date range mosaic: {output_file}")
    return output_file


def create_difference_image(mosaic1_path, mosaic2_path, output_path):
    """
    Create a difference image between two mosaics by aligning them on a common grid.
    The grid is defined by the intersection of the two images' extents, and the
    pixel size is chosen as the finer (smallest) of the two resolutions.
    
    Args:
        mosaic1_path (str): Path to first mosaic (earlier date)
        mosaic2_path (str): Path to second mosaic (later date)
        output_path (str): Path for output difference image
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds, array_bounds

    # Open the two source mosaics
    with rasterio.open(mosaic1_path) as src1, rasterio.open(mosaic2_path) as src2:
        # Get geographic bounds for each (left, bottom, right, top)
        bounds1 = array_bounds(src1.height, src1.width, src1.transform)
        bounds2 = array_bounds(src2.height, src2.width, src2.transform)
        
        # Compute the intersection of the two extents
        intersection_left = max(bounds1[0], bounds2[0])
        intersection_bottom = max(bounds1[1], bounds2[1])
        intersection_right = min(bounds1[2], bounds2[2])
        intersection_top = min(bounds1[3], bounds2[3])
        
        if intersection_left >= intersection_right or intersection_bottom >= intersection_top:
            raise ValueError("No overlapping geographic area found between the two mosaics.")
        
        # Choose target pixel resolution based on the finer (smaller) pixel size
        pixel_width1 = abs(src1.transform.a)
        pixel_width2 = abs(src2.transform.a)
        pixel_height1 = abs(src1.transform.e)
        pixel_height2 = abs(src2.transform.e)
        target_pixel_width = min(pixel_width1, pixel_width2)
        target_pixel_height = min(pixel_height1, pixel_height2)
        
        # Determine target dimensions from the intersection extent and chosen resolution.
        target_width = int(round((intersection_right - intersection_left) / target_pixel_width))
        target_height = int(round((intersection_top - intersection_bottom) / target_pixel_height))
        
        # Build the new affine transform for the target grid.
        target_transform = from_bounds(
            intersection_left, intersection_bottom, intersection_right, intersection_top,
            target_width, target_height
        )
        
        # Prepare empty arrays for the reprojected data (reading band 1 only).
        dst1 = np.empty((target_height, target_width), dtype=np.float32)
        dst2 = np.empty((target_height, target_width), dtype=np.float32)
        
        # Reproject mosaic1 onto the target grid.
        reproject(
            source=rasterio.band(src1, 1),
            destination=dst1,
            src_transform=src1.transform,
            src_crs=src1.crs,
            dst_transform=target_transform,
            dst_crs=src1.crs,  # Both are assumed to be in the same CRS already.
            resampling=Resampling.nearest
        )
        
        # Reproject mosaic2 onto the target grid.
        reproject(
            source=rasterio.band(src2, 1),
            destination=dst2,
            src_transform=src2.transform,
            src_crs=src2.crs,
            dst_transform=target_transform,
            dst_crs=src2.crs,
            resampling=Resampling.nearest
        )
    
    # Compute the difference (later mosaic minus earlier mosaic)
    diff = dst2 - dst1
    
    # Use the profile from mosaic1 and update with new dimensions, transform, and data type.
    with rasterio.open(mosaic1_path) as src1:
        profile = src1.profile.copy()
    profile.update({
        'height': target_height,
        'width': target_width,
        'transform': target_transform,
        'dtype': 'float32'
    })
    
    # Write out the difference as a new GeoTIFF.
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff, 1)


def main():
    try:
        ps = config.getPS()
        
        # Ensure we use the DSW data directory
        ps.dataDir = './data/DSW'
        
        # Create directory if it doesn't exist
        os.makedirs(ps.dataDir, exist_ok=True)
        
        print(f"Using DSW data directory: {ps.dataDir}")
        
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

        # Example usage of date range stitching
        # Uncomment and modify dates as needed:
        date1 = '20240801'
        date2 = '20241008'
        date3 = '20241009'
        date4 = '20241015'  
        stitch_date_range(date1, date2, ps.dataDir, 'wtr')
        stitch_date_range(date3, date4, ps.dataDir, 'wtr')

        # After stitching date ranges, create difference image
        mosaic1_path = os.path.join(ps.dataDir, f'mosaic_{date1}_to_{date2}_wtr.tif')
        mosaic2_path = os.path.join(ps.dataDir, f'mosaic_{date3}_to_{date4}_wtr.tif')
        diff_output = os.path.join(ps.dataDir, f'difference_{date1}_to_{date4}_wtr.tif')
        
        create_difference_image(mosaic1_path, mosaic2_path, diff_output)
        print(f"Created difference image: {diff_output}")

    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()