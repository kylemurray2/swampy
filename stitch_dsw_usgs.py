import os
import re
import tarfile
import rasterio
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import multiprocessing as mp
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import math
from pyproj import Transformer, CRS
import gc
import sys
import hashlib

# Check if we have a config file, otherwise use default directory
try:
    import config
    use_config = True
except ImportError:
    use_config = False
    print("No config module found, using default directories")

# USGS DSW value interpretations
# 0: Not water
# 1: Water – high confidence
# 2: Water – moderate confidence
# 3: Partial Surface Water - Conservative 
# 4: Partial Surface Water - Aggressive
# 9: Cloud, cloud shadow
# 255: Fill (NoData)

def custom_merge(file_list, bounds=None, res=None, nodata=None, precision=10):
    """
    Custom merge function for USGS DSW files that preserves the semantic meaning of values.
    Priority order (high to low): 1, 2, 3, 4, 0, 9, 255
    
    Args:
        file_list: List of files to merge
        bounds: Output bounds (left, bottom, right, top)
        res: Output resolution (x_res, y_res)
        nodata: NoData value
        precision: Precision for coordinate comparison
        
    Returns:
        tuple: (merged_data, output_transform)
    """
    if not file_list:
        return None, None
    
    # Open all input files
    sources = [rasterio.open(f) for f in file_list]
    
    # Determine output dimensions and transform using rasterio's merge function
    # First get the bounds from all sources
    if bounds is None:
        # Use bounds from all input files
        all_bounds = [src.bounds for src in sources]
        # Get the union of all bounds
        left = min([b.left for b in all_bounds])
        bottom = min([b.bottom for b in all_bounds])
        right = max([b.right for b in all_bounds])
        top = max([b.top for b in all_bounds])
        bounds = (left, bottom, right, top)
    else:
        left, bottom, right, top = bounds
    
    # Determine resolution to use
    if res is None:
        # Use the minimum resolution (max detail) from all sources
        res_xs = [src.res[0] for src in sources]
        res_ys = [src.res[1] for src in sources]
        res = (min(res_xs), min(res_ys))
    
    # Calculate dimensions
    width = int(round((right - left) / res[0]))
    height = int(round((top - bottom) / res[1]))
    
    # Create transform for output
    dest_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    
    # Create output array filled with NoData value (255 for USGS DSW)
    if nodata is None:
        nodata = 255
    dest = np.full((1, height, width), nodata, dtype=np.uint8)
    
    # Read each input file and merge with priority logic
    for src in sources:
        # Determine the window in the output raster for this input
        try:
            window = rasterio.windows.from_bounds(
                *src.bounds, dest_transform, precision=precision
            )
            
            # Convert window to integers
            window = window.round_lengths().round_offsets()
            
            # Ensure window is within output dimensions
            row_start = max(0, int(window.row_off))
            row_stop = min(height, int(window.row_off + window.height))
            col_start = max(0, int(window.col_off))
            col_stop = min(width, int(window.col_off + window.width))
            
            # Skip if window is empty
            if row_start >= row_stop or col_start >= col_stop:
                continue
            
            # Get the window within the source raster
            src_window = rasterio.windows.Window(
                col_off=max(0, -int(window.col_off)),
                row_off=max(0, -int(window.row_off)),
                width=col_stop - col_start,
                height=row_stop - row_start,
            )
            
            # Read the data
            try:
                data = src.read(1, window=src_window)
            except Exception as e:
                print(f"Error reading from source: {e}")
                continue
                
            # Get the current output data in this window
            curr_data = dest[0, row_start:row_stop, col_start:col_stop]
            
            # Apply priority merge logic - ensure values are preserved correctly
            
            # First, create a new array to hold our merged result
            merged = curr_data.copy()
            
            # For each pixel, apply our priority rules
            # Priority order: 1 > 2 > 3 > 4 > 0 > 9 > 255 (NoData)
            
            # Apply the highest priority value (1) - Water high confidence
            mask = (data == 1)
            if mask.any():
                merged[mask] = 1
                
            # Apply the next highest priority value (2) - Water moderate confidence
            # But only where we haven't already set high confidence water
            mask = (data == 2) & (merged != 1)
            if mask.any():
                merged[mask] = 2
                
            # Apply value 3 - Partial Surface Water - Conservative
            # But only where we haven't already set higher priority water
            mask = (data == 3) & ((merged != 1) & (merged != 2))
            if mask.any():
                merged[mask] = 3
                
            # Apply value 4 - Partial Surface Water - Aggressive
            # But only where we haven't already set higher priority water
            mask = (data == 4) & ((merged != 1) & (merged != 2) & (merged != 3))
            if mask.any():
                merged[mask] = 4
                
            # Apply value 0 - Not water
            # But only where we haven't already set any water
            mask = (data == 0) & ((merged != 1) & (merged != 2) & (merged != 3) & (merged != 4))
            if mask.any():
                merged[mask] = 0
                
            # Apply value 9 - Cloud
            # But only where we haven't already set any water or not-water
            mask = (data == 9) & ((merged != 0) & (merged != 1) & (merged != 2) & (merged != 3) & (merged != 4))
            if mask.any():
                merged[mask] = 9
                
            # Fill in NoData (255) where it's still NoData in the output
            # We don't need to test for this as we initialize with 255
            mask = (data != 255) & (merged == 255)
            if mask.any():
                merged[mask] = data[mask]
                
            # Update the output
            dest[0, row_start:row_stop, col_start:col_stop] = merged
        except Exception as e:
            print(f"Error processing source window: {e}")
            continue
        
    # Close all input files
    for src in sources:
        src.close()
        
    return dest, dest_transform

def check_small_files(data_dir, min_size_bytes=1024):
    """Check for and delete files smaller than min_size_bytes (default 1KB)"""
    small_files = []
    data_dir = Path(data_dir)
    
    for tif_file in data_dir.rglob('*.tif'):
        if tif_file.stat().st_size < min_size_bytes:
            small_files.append((tif_file, tif_file.stat().st_size))
            tif_file.unlink()
            
    return small_files

def get_crs_from_file(file_path):
    """
    Extract CRS from a GeoTIFF file and return a shortened identifier
    that won't cause "filename too long" errors
    """
    try:
        with rasterio.open(file_path) as src:
            crs = src.crs
            
            # Check if it's a standard EPSG code
            if crs.is_epsg_code:
                return f"EPSG_{crs.to_epsg()}"
            
            # Get WKT string representation
            wkt = crs.wkt
            
            # For Albers or other projections, use a shorter identifier
            if "Albers" in wkt:
                # Extract key parameters to create a unique but short name
                # If it's the standard Albers Equal Area CONUS projection
                if ('latitude_of_center",23' in wkt and 
                    'longitude_of_center",-96' in wkt and
                    'standard_parallel_1",29.5' in wkt and
                    'standard_parallel_2",45.5' in wkt):
                    return "ALBERS_CONUS"
                else:
                    # Create a hash of the WKT string to use as an identifier
                    hash_object = hashlib.md5(wkt.encode())
                    return f"ALBERS_{hash_object.hexdigest()[:8]}"
            elif "UTM" in wkt:
                # Extract UTM zone if possible
                zone_match = re.search(r'UTM zone (\d+)', wkt)
                if zone_match:
                    zone = zone_match.group(1)
                    return f"UTM_{zone}"
                else:
                    hash_object = hashlib.md5(wkt.encode())
                    return f"UTM_{hash_object.hexdigest()[:8]}"
            else:
                # For other projections, use a hash of the WKT string
                hash_object = hashlib.md5(wkt.encode())
                return f"PROJ_{hash_object.hexdigest()[:8]}"
    except Exception as e:
        print(f"Warning: Error reading CRS from {file_path}: {str(e)}")
        return None

def examine_values(file_path, label=""):
    """Print unique values in a file to help debug"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            unique_vals = np.unique(data)
            print(f"{label} unique values: {unique_vals}")
            # Also print counts of each value
            for val in unique_vals:
                count = np.sum(data == val)
                print(f"Value {val}: {count} pixels ({(count/data.size)*100:.2f}%)")
    except Exception as e:
        print(f"Error examining values in {file_path}: {str(e)}")

def process_date_directory(date_dir):
    """Process a single date directory of USGS DSW files"""
    print(f"\nProcessing directory: {date_dir}")
    date_dir = Path(date_dir)
    
    # Find all USGS DSW files in the directory directly
    inwam_files = list(date_dir.glob('*INWAM*.TIF'))
    intsm_files = list(date_dir.glob('*INTSM*.TIF'))
    wtr_files = list(date_dir.glob('*WTR*.TIF'))
    
    # Process each product type separately
    file_groups = [
        ("INWAM", inwam_files),
        ("INTSM", intsm_files),
        ("WTR", wtr_files)
    ]
    
    for product_type, files in file_groups:
        if not files:
            continue
            
        print(f"Processing {product_type} files: found {len(files)} files")
        
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
        
        # Create output directory for temporary files if needed
        temp_dir = date_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Prepare batches for parallel processing
        batch_size = max(1, len(files) // mp.cpu_count())
        batches = []
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            output_file = temp_dir / f'temp_{product_type}_{i//batch_size}.tif'
            batches.append((
                batch, "ALBERS_CONUS", output_file, layer_bounds
            ))
        
        # Process batches in parallel
        with mp.Pool() as pool:
            output_files = []
            for result in pool.imap_unordered(process_batch, batches):
                if result is not None:
                    output_files.append(result)
                # Add progress indication
                print(".", end="", flush=True)
            print()  # New line after progress dots
        
        # Create final mosaic
        if output_files:
            final_output = date_dir / f'mosaic_{product_type}.tif'
            print(f"Creating final mosaic for {product_type}...")
            
            with rasterio.Env():
                with rasterio.open(output_files[0]) as src:
                    profile = src.profile.copy()
                    colormap = None
                    try:
                        colormap = src.colormap(1)
                    except:
                        pass
                    
                    # Use our custom merge for the final output too
                    merged_data, merged_transform = custom_merge(output_files)
                    
                    # Update profile with the merged dimensions and transform
                    profile.update({
                        'height': merged_data.shape[1],
                        'width': merged_data.shape[2],
                        'transform': merged_transform,
                        'count': 1,
                        'nodata': 255  # Use 255 as NoData for DSW
                    })
                    
                    with rasterio.open(final_output, 'w', **profile) as dst:
                        dst.write(merged_data)
                        if colormap:
                            dst.write_colormap(1, colormap)
            
            # Check values in the final output
            examine_values(final_output, f"Final {product_type} mosaic")
            
            # Clean up temporary files
            for temp_file in output_files:
                Path(temp_file).unlink()
            
            # Also create a generic mosaic.tif for backward compatibility
            if product_type == "INWAM":
                compat_output = date_dir / "mosaic.tif"
                # Just copy the file
                if not compat_output.exists():
                    import shutil
                    shutil.copy(final_output, compat_output)
                    print(f"Created backward-compatible mosaic.tif")

def organize_by_date(data_dir):
    """Organize USGS DSW files into date-based directories"""
    data_dir = Path(data_dir)
    
    # Find all TIF files in the data directory
    tif_files = list(data_dir.glob('*.TIF'))
    
    # Organize files by date
    for tif_file in tif_files:
        date = extract_date_from_usgs_filename(tif_file)
        if date:
            date_dir = data_dir / date
            date_dir.mkdir(exist_ok=True)
            new_path = date_dir / tif_file.name
            if not new_path.exists():
                tif_file.rename(new_path)
                print(f"Moved {tif_file.name} to {date_dir}")
            else:
                print(f"File {tif_file.name} already exists in {date_dir}")

def extract_date_from_usgs_filename(filename):
    """Extract date from USGS filename"""
    # LC09_CU_002009_20230319_20230324_02_INWAM.TIF
    # Looking for the first date (acquisition date)
    match = re.search(r'_(\d{8})_', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def compare_values(source_file, mosaic_file, output_dir=None):
    """
    Compare values between a source file and a mosaic to identify differences.
    This helps diagnose why mosaicked values might differ from source values.
    
    Args:
        source_file: Path to an original source file
        mosaic_file: Path to the mosaic file
        output_dir: Optional directory to save comparison results
    """
    print(f"\nComparing values: {os.path.basename(source_file)} -> {os.path.basename(mosaic_file)}")
    
    try:
        # Open source file
        with rasterio.open(source_file) as src:
            src_data = src.read(1)
            src_profile = src.profile
            src_bounds = src.bounds
            src_transform = src.transform
            src_unique = np.unique(src_data)
            
            # Open mosaic file
            with rasterio.open(mosaic_file) as mosaic:
                # Determine the window in the mosaic that corresponds to the source file
                window = rasterio.windows.from_bounds(
                    *src_bounds, mosaic.transform
                )
                
                # Round to integers
                window = window.round_lengths().round_offsets()
                
                # Ensure window is within output dimensions
                row_start = max(0, int(window.row_off))
                row_stop = min(mosaic.height, int(window.row_off + window.height))
                col_start = max(0, int(window.col_off))
                col_stop = min(mosaic.width, int(window.col_off + window.width))
                
                # Skip if window is empty
                if row_start >= row_stop or col_start >= col_stop:
                    print("  Source file area doesn't overlap with mosaic")
                    return
                
                # Read the data from the mosaic
                mosaic_window = mosaic.read(1, window=Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=col_stop-col_start,
                    height=row_stop-row_start
                ))
                
                # Reproject the source data to match the mosaic window
                src_resampled = np.empty(mosaic_window.shape, dtype=np.uint8)
                reproject(
                    source=src_data,
                    destination=src_resampled,
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=mosaic.transform,
                    dst_crs=mosaic.crs,
                    dst_resolution=(mosaic.res[0], mosaic.res[1]),
                    resampling=Resampling.nearest
                )
                
                # Compare values
                print("  Source unique values:", src_unique)
                print("  Reprojected source unique values:", np.unique(src_resampled))
                print("  Mosaic window unique values:", np.unique(mosaic_window))
                
                # Calculate differences
                different_pixels = (src_resampled != mosaic_window)
                diff_count = different_pixels.sum()
                total_pixels = src_resampled.size
                
                print(f"  Differences: {diff_count} pixels ({diff_count/total_pixels:.2%} of area)")
                
                if diff_count > 0:
                    # Show what values were changed
                    for src_val in np.unique(src_resampled):
                        # Where source has this value, what does mosaic have?
                        mask = src_resampled == src_val
                        mosaic_vals = mosaic_window[mask]
                        mosaic_unique = np.unique(mosaic_vals)
                        
                        if len(mosaic_unique) > 1:  # Values were changed
                            print(f"  Source value {src_val} -> Mosaic values: {mosaic_unique}")
                            for mosaic_val in mosaic_unique:
                                if mosaic_val != src_val:
                                    count = (mosaic_vals == mosaic_val).sum()
                                    print(f"    {src_val} -> {mosaic_val}: {count} pixels")
                
                # Save comparison visualization if requested
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Create a visualization showing where values differ
                    diff_vis = np.zeros(src_resampled.shape, dtype=np.uint8)
                    diff_vis[different_pixels] = 1  # Mark differences
                    
                    # Save as a GeoTIFF
                    vis_file = output_dir / f"diff_{os.path.basename(source_file)}"
                    vis_profile = src_profile.copy()
                    vis_profile.update({
                        'count': 1,
                        'height': diff_vis.shape[0],
                        'width': diff_vis.shape[1],
                        'transform': mosaic.transform
                    })
                    
                    with rasterio.open(vis_file, 'w', **vis_profile) as dst:
                        dst.write(diff_vis, 1)
                    
                    print(f"  Saved difference visualization to {vis_file}")
                    
    except Exception as e:
        print(f"Error comparing files: {str(e)}")

def find_file_by_pattern(directory, pattern):
    """Find files matching a pattern in a directory"""
    matches = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if re.search(pattern, filename):
                matches.append(os.path.join(root, filename))
    return matches

def process_batch(batch_info):
    """Process a single batch of files (to be run in parallel)"""
    files, folder_name, output_file, layer_bounds = batch_info
    
    if not files:
        return None
    
    try:
        print(f"Merging {len(files)} files...")
        
        # All files should be DSW products (INWAM, INTSM, WTR, etc.)
        merged_img, merged_transform = custom_merge(files, bounds=layer_bounds)
        print("Using custom DSW merge with priority order.")
        
        # Debug output
        if merged_img is not None:
            unique_vals = np.unique(merged_img[0])
            print(f"Merged unique values: {unique_vals}")
        else:
            print("Merging produced no output.")
            return None
        
        # Write out the merged result
        with rasterio.open(files[0]) as src:
            profile = src.profile.copy()
            colormap = None
            try:
                colormap = src.colormap(1)
            except:
                pass
            
            # Update profile with the merged dimensions and transform
            profile.update({
                'height': merged_img.shape[1],
                'width': merged_img.shape[2],
                'transform': merged_transform,
                'count': 1,
                'nodata': 255  # Use 255 as NoData for DSW
            })
            
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(merged_img)
                if colormap:
                    dst.write_colormap(1, colormap)
        
        # Examine the output file
        examine_values(output_file, "Output batch mosaic")
        
        return output_file
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return None

def main():
    pattern = '????????'
    try:
        # Determine data directory
        if use_config:
            ps = config.getPS()
            # Use dataDir_usgs instead of dataDir if it exists
            if hasattr(ps, 'dataDir_usgs'):
                data_dir = ps.dataDir_usgs
            else:
                data_dir = ps.dataDir
        else:
            # Default location for USGS DSW data
            data_dir = os.path.join(os.getcwd(), 'data', 'usgs_dsw')
            
            # Create the directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
        
        # Add command line arguments
        debug_mode = False
        date_to_process = None
        
        if len(sys.argv) > 1:
            # Check for debug mode
            if '--debug' in sys.argv:
                debug_mode = True
                print("Running in debug mode - will perform detailed value comparisons")
            
            # Check for specific date to process
            for arg in sys.argv[1:]:
                if re.match(r'^\d{8}$', arg):  # YYYYMMDD format
                    date_to_process = arg
                    print(f"Will only process date: {date_to_process}")
                    date_dirs = [os.path.join(data_dir, date_to_process)]
                    if not os.path.exists(date_dirs[0]):
                        print(f"Date directory {date_to_process} not found")
                        return
                    break
            else:
                if date_to_process is None:
                    date_dirs = sorted(glob.glob(os.path.join(data_dir, pattern)))
        else:
            date_dirs = sorted(glob.glob(os.path.join(data_dir, pattern)))
        
        print(f"Using data directory: {data_dir}")
        
        # Organize files by date first
        organize_by_date(data_dir)
        
        # Find all date directories if not already specified
        if 'date_dirs' not in locals():
            date_dirs = sorted(glob.glob(os.path.join(data_dir, pattern)))
        
        if not date_dirs:
            print("No date directories found in", data_dir)
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
        
        # Process each date directory
        for date_dir in date_dirs:
            try:
                process_date_directory(date_dir)
                
                # If debug mode, compare source files with mosaic
                if debug_mode:
                    # Check both the INWAM-specific mosaic and the generic mosaic
                    for mosaic_file in [
                        os.path.join(date_dir, 'mosaic_INWAM.tif'),
                        os.path.join(date_dir, 'mosaic.tif')
                    ]:
                        if os.path.exists(mosaic_file):
                            # Find source files (up to 3) for comparison
                            inwam_files = list(Path(date_dir).glob('*INWAM*.TIF'))
                            if inwam_files:
                                # Limit to 3 files for comparison
                                for src_file in inwam_files[:3]:
                                    compare_values(str(src_file), mosaic_file, 
                                                  output_dir=os.path.join(date_dir, 'debug'))
            except Exception as e:
                print(f"Error processing directory {date_dir}: {str(e)}")
                continue
        
        # Example of date range stitching - uncomment and modify as needed
        # date1 = '20230101'
        # date2 = '20230131'
        # date3 = '20230201'
        # date4 = '20230228'
        # stitch_date_range(date1, date2, data_dir)
        # stitch_date_range(date3, date4, data_dir)
        
        # # Create difference image
        # mosaic1_path = os.path.join(data_dir, f'mosaic_{date1}_to_{date2}.tif')
        # mosaic2_path = os.path.join(data_dir, f'mosaic_{date3}_to_{date4}.tif')
        # diff_output = os.path.join(data_dir, f'difference_{date1}_to_{date4}.tif')
        # create_difference_image(mosaic1_path, mosaic2_path, diff_output)
        
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()


