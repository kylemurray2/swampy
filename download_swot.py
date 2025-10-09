'''
info about lakes dataset 
https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_LakeSP_2.0


'''

import os
import requests
from datetime import datetime
from shapely import wkt
from pystac_client import Client
from swampy import config
import xarray as xr
import earthaccess
import re
from concurrent.futures import ThreadPoolExecutor

def search_swot_data(collection, date_range, intersects_geometry):
    """
    Search for SWOT data from NASA Earthdata using earthaccess library.

    Parameters:
        collection (str): SWOT collection identifier to search.
        date_range (str): Date range string in ISO8601 interval format (e.g., "2023-03-01T00:00:00Z/2023-03-31T23:59:59Z").
        intersects_geometry (dict): GeoJSON geometry for spatial intersection.

    Returns:
        list: A list of granules matching the search criteria.
    """
    # Login to Earthdata (will use .netrc if available or prompt for credentials)
    auth = earthaccess.login()
    
    # Extract a simple bounding box from the intersects_geometry polygon
    if "coordinates" in intersects_geometry:
        # Get the coordinates from the polygon
        polygon_coords = intersects_geometry["coordinates"][0]
        
        # Calculate bounding box
        lons = [p[0] for p in polygon_coords]
        lats = [p[1] for p in polygon_coords]
        
        # Extract individual coordinate components
        min_lon = min(lons)
        min_lat = min(lats)
        max_lon = max(lons)
        max_lat = max(lats)
        
        print(f"Search bbox: [{min_lon}, {min_lat}, {max_lon}, {max_lat}]")
    else:
        # If for some reason intersects_geometry isn't properly formatted, use it directly
        print("Using original geometry for search is not supported with earthaccess")
        print("Please provide a valid GeoJSON polygon")
        return []
    
    # Parse the date range into start and end dates
    start_date, end_date = date_range.split('/')
    
    # Search for granules using earthaccess with the correct bounding_box format
    # The order expected by earthaccess is: lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat
    results = earthaccess.search_data(
        short_name=collection,
        bounding_box=(min_lon, min_lat, max_lon, max_lat),  # Pass as tuple with correct order
        temporal=(start_date, end_date),
        count=1000
    )
    
    print(f"Found {len(results)} granules for SWOT data.")
    return results

def return_swot_asset(granule):
    """
    Return the download URL for the 100m SWOT asset with a NetCDF (.nc) extension.
    
    Parameters:
        granule: An earthaccess granule object
        
    Returns:
        str or None: URL to the 100m NetCDF file, or None if not found
    """
    try:
        # First try the data_links method if it exists
        if hasattr(granule, 'data_links') and callable(granule.data_links):
            for url in granule.data_links():
                if "100m" in url and url.lower().endswith('.nc'):
                    return url
        
        # If that doesn't exist, try other ways to access URLs
        # Option 1: Check if granule has 'links' attribute
        elif hasattr(granule, 'links'):
            for link in granule.links:
                if isinstance(link, dict) and 'href' in link:
                    url = link['href']
                    if "100m" in url and url.lower().endswith('.nc'):
                        return url
        
        # Option 2: Use earthaccess.download_granules API directly
        else:
            # In some versions, granules must be downloaded through the API
            # Print available attributes to help debug
            print(f"Granule attributes: {dir(granule)}")
            print(f"Granule as string: {str(granule)}")
            return None
            
    except Exception as e:
        print(f"Error in return_swot_asset: {e}")
        return None
    
    return None

def dl(url, outname):
    """
    Download the file at URL and save it to outname using earthaccess.
    """
    try:
        # Use earthaccess to download the file with built-in authentication
        earthaccess.download(url, outname)
        print(f"Successfully downloaded {outname}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def safe_open_dataset(fn):
    try:
        # Handle different file types
        if fn.endswith('.nc'):
            # Open NetCDF files with xarray
            ds = xr.open_dataset(fn, chunks={'x': 100, 'y': 100})
            return ds
        elif fn.endswith('.shp'):
            # Use geopandas for shapefiles
            import geopandas as gpd
            gdf = gpd.read_file(fn)
            return gdf
        elif fn.endswith('.xml') or fn.endswith('.shp.xml'):
            # Skip XML files or parse them if needed
            print(f"Skipping XML file: {fn}")
            return None
        else:
            print(f"Unsupported file type: {fn}")
            return None
    except Exception as e:
        print(f"Error opening {fn}: {e}")
        return None

def parse_swot_filename(filename):
    """
    Parse a SWOT filename to extract orbit/cycle information.
    
    SWOT filename format: SWOT_L2_HR_LakeSP_Unassigned_005_261_NA_20231022T000607_20231022T001305_PGC0_01.shp
    where:
    - 005 is the cycle number
    - 261 is the pass/track number
    
    Parameters:
    filename (str): The SWOT filename to parse
    
    Returns:
    dict: Dictionary containing extracted information
    """
    try:
        # Split by underscore
        parts = filename.split('_')
        
        # Extract cycle and pass numbers
        if len(parts) >= 7:
            cycle_num = parts[5]
            pass_num = parts[6]
            
            # Extract date and time
            date_time = None
            for part in parts:
                if part.startswith('20') and 'T' in part:
                    date_time = part
                    break
            
            return {
                'cycle': cycle_num,
                'pass': pass_num,
                'datetime': date_time
            }
        else:
            return None
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

if __name__ == "__main__":
    ps = config.getPS()

    # Use the download folder specified for SWOT data from the configuration.
    base_folder = ps.dataDir_swot
    
    # Create the base folder if it doesn't exist
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    
    # Load the AOI polygon from map.wkt
    with open('map.wkt', 'r') as f:
        polygon_wkt = f.read().strip()
    aoi = wkt.loads(polygon_wkt)
    intersects_geometry = aoi.__geo_interface__

    # Print information about the polygon for verification
    bounds = aoi.bounds  # Get the bounding box as (minx, miny, maxx, maxy)
    centroid = aoi.centroid
    area_km2 = aoi.area * 111 * 111  # Rough conversion from degrees to km²

    print(f"Polygon information:")
    print(f"  Centroid: {centroid.y:.5f}, {centroid.x:.5f}")
    print(f"  Bounds: {bounds}")
    print(f"  Approximate area: {area_km2:.2f} km²")
    print(f"  Vertex count: {len(aoi.exterior.coords)}")

    # Convert dates from "YYYYMMDD" (from params.yaml) to ISO8601 strings.
    start_time_iso = datetime.strptime(ps.date_start, "%Y%m%d").strftime("%Y-%m-%dT00:00:00Z")
    if ps.date_stop:
        end_time_iso = datetime.strptime(ps.date_stop, "%Y%m%d").strftime("%Y-%m-%dT23:59:59Z")
    else:
        end_time_iso = datetime.utcnow().strftime("%Y-%m-%dT23:59:59Z")
    date_range = f"{start_time_iso}/{end_time_iso}"

    # Use the first SWOT collection specified in the configuration.
    dataset = ps.swot_collections[0]

    # Different handling based on collection type
    is_lakes_collection = dataset == "SWOT_L2_HR_LakeSP_2.0"
    
    if is_lakes_collection:
        print(f"Processing Lakes Collection: {dataset}")
        # Set up the LakesData folder structure
        lakes_folder = os.path.join(base_folder, 'LakesData')
        if not os.path.isdir(lakes_folder):
            os.makedirs(lakes_folder, exist_ok=True)
    else:
        print(f"Processing Raster Collection: {dataset}")
        # Set up the RasterData subfolder
        raster_folder = os.path.join(base_folder, 'RasterData')
        if not os.path.isdir(raster_folder):
            os.makedirs(raster_folder, exist_ok=True)
        print(f"Using raster data folder: {raster_folder}")

    # Replace the hardcoded TARGET_PASSES list with the pass_num from configuration
    TARGET_PASSES = [ps.pass_num] if hasattr(ps, 'pass_num') else ['261']  # Default to '261' if not specified

    # Update the print statement to show what pass number we're using
    print(f"Filtering for pass numbers: {TARGET_PASSES}")

    # Perform the search for SWOT data.
    granules = search_swot_data(dataset, date_range, intersects_geometry)

    # Authenticate with Earthdata (needed for downloads)
    auth = earthaccess.login()

    # Track how many files were downloaded vs already existed
    downloaded_count = 0
    existing_count = 0
    skipped_count = 0

    # Loop over each returned granule and download the appropriate assets
    for granule in granules:
        if is_lakes_collection:
            try:
                # Find and get all data links
                links = []
                if hasattr(granule, 'data_links') and callable(granule.data_links):
                    links = granule.data_links()
                
                # Filter links for Prior files with specified pass numbers
                matching_links = []
                for link in links:
                    filename = link.split('/')[-1]
                    info = parse_swot_filename(filename)
                    if info and info['pass'] in TARGET_PASSES and "Prior" in filename:
                        matching_links.append(link)
                
                if not matching_links:
                    print(f"No Prior files with pass {TARGET_PASSES} found for granule {getattr(granule, 'producer_granule_id', str(granule))}")
                    skipped_count += 1
                    continue
                
                print(f"Found {len(matching_links)} Prior files with pass {TARGET_PASSES}")
                
                # Use concurrent.futures for parallel downloads
                download_tasks = []
                
                for link in matching_links:
                    filename = link.split('/')[-1]
                    
                    # Extract date from filename using the pattern NA_YYYYMMDD
                    date_match = re.search(r'NA_(\d{8})T', filename)
                    
                    if date_match:
                        file_date_str = date_match.group(1)  # This will be YYYYMMDD
                        
                        # Create date folder if it doesn't exist
                        date_folder = os.path.join(lakes_folder, file_date_str)
                        if not os.path.isdir(date_folder):
                            os.makedirs(date_folder, exist_ok=True)
                        
                        # Check if the corresponding .shp file already exists
                        base_name = os.path.splitext(filename)[0]  # Remove .zip or other extension
                        shp_filename = f"{base_name}.shp"
                        shp_path = os.path.join(date_folder, shp_filename)
                        
                        if os.path.exists(shp_path):
                            print(f"Shapefile {shp_filename} already exists, skipping download")
                            existing_count += 1
                        else:
                            out_path = os.path.join(date_folder, filename)
                            
                            if not os.path.exists(out_path):
                                # Add download task to the list
                                download_tasks.append({
                                    'link': link,
                                    'date_folder': date_folder,
                                    'out_path': out_path,
                                    'filename': filename,
                                    'file_type': "261-File" # For logging purposes
                                })
                            else:
                                # File exists but possibly wasn't extracted yet
                                if filename.lower().endswith('.zip'):
                                    # Check if we need to extract the zip file
                                    # by seeing if any related shapefiles exist
                                    existing_related_shp = False
                                    if os.path.exists(date_folder):
                                        prefix = os.path.splitext(filename)[0]
                                        existing_related_shp = any(f.startswith(prefix) and f.endswith('.shp') 
                                                 for f in os.listdir(date_folder))
                                    
                                    if not existing_related_shp:
                                        print(f"ZIP file exists but needs extraction: {filename}")
                                        try:
                                            import zipfile
                                            with zipfile.ZipFile(out_path, 'r') as zip_ref:
                                                zip_ref.extractall(date_folder)
                                            print(f"Extraction complete!")
                                        except Exception as e:
                                            print(f"Error extracting {out_path}: {e}")
                                    else:
                                        existing_count += 1
                                        print(f"File {filename} already extracted, skipping")
                                else:
                                    existing_count += 1
                                    print(f"File {filename} already exists, skipping download")
                    else:
                        print(f"Could not extract date from filename: {filename}")
                        # Skip this file as we can't determine the date folder
                        continue
                
                # Now perform parallel downloads using ThreadPoolExecutor
                if download_tasks:
                    print(f"Downloading {len(download_tasks)} files in parallel...")
                    
                    def download_and_extract(task):
                        """Helper function to download and extract a file"""
                        try:
                            print(f"Downloading file with '261': {task['filename']} to {task['date_folder']}")
                            earthaccess.download([task['link']], local_path=task['date_folder'])
                            
                            # Unzip if it's a zip file
                            if task['filename'].lower().endswith('.zip'):
                                print(f"Extracting contents of {task['filename']}...")
                                import zipfile
                                with zipfile.ZipFile(task['out_path'], 'r') as zip_ref:
                                    zip_ref.extractall(task['date_folder'])
                            
                            return True  # Success
                        except Exception as e:
                            print(f"Error in primary download method for {task['filename']}: {e}")
                            
                            # Fallback to direct request
                            try:
                                session = requests.Session()
                                response = session.get(task['link'], stream=True, auth=earthaccess.auth.session.auth)
                                response.raise_for_status()
                                
                                with open(task['out_path'], 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192): 
                                        f.write(chunk)
                                
                                # Unzip if it's a zip file
                                if task['filename'].lower().endswith('.zip'):
                                    print(f"Extracting contents of {task['filename']}...")
                                    import zipfile
                                    with zipfile.ZipFile(task['out_path'], 'r') as zip_ref:
                                        zip_ref.extractall(task['date_folder'])
                                
                                return True  # Success with fallback method
                            except Exception as e2:
                                print(f"Fallback download also failed for {task['filename']}: {e2}")
                                return False  # Both methods failed
                    
                    # Use ThreadPoolExecutor for parallel downloads - more efficient for I/O operations
                    max_workers = min(32, len(download_tasks))  # Use up to 32 threads
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(download_and_extract, download_tasks))
                    
                    # Count successful downloads
                    downloaded_count += sum(results)
                    
                    print(f"Completed parallel downloads: {sum(results)} successful, {len(download_tasks) - sum(results)} failed")
                
            except Exception as e:
                print(f"Error processing lakes granule: {e}")
                continue
        else:
            # For Raster collection, download 100m NetCDF files
            asset_url = return_swot_asset(granule)
            if asset_url is None:
                # Use a more reliable attribute for identification
                granule_id = getattr(granule, 'producer_granule_id', 
                            getattr(granule, 'granule_id', 
                            getattr(granule, 'name', 
                            str(granule))))  # Fallback to string representation
                
                print(f"Skipping granule {granule_id} as it does not contain a 100m asset.")
                skipped_count += 1
                continue
            
            # Get the filename from the URL    
            filename = asset_url.split('/')[-1]
            out_name = os.path.join(raster_folder, filename)
            
            # Check if the file already exists
            if not os.path.exists(out_name):
                print(f"Downloading {filename} to {out_name}")
                
                try:
                    # Download URLs directly to the RasterData folder
                    earthaccess.download([asset_url], local_path=raster_folder)
                    downloaded_count += 1
                except Exception as e:
                    print(f"Error downloading {asset_url}: {e}")
                    # Fallback to direct download if earthaccess fails
                    try:
                        session = requests.Session()
                        response = session.get(asset_url, stream=True, auth=earthaccess.auth.session.auth)
                        response.raise_for_status()
                        
                        with open(out_name, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192): 
                                f.write(chunk)
                        downloaded_count += 1
                    except Exception as e2:
                        print(f"Fallback download also failed: {e2}")
            else:
                existing_count += 1
                print(f"File {filename} already exists, skipping download")

    # Print a summary of the download process
    print(f"\nDownload Summary:")
    print(f"  Collection type: {'Lakes' if is_lakes_collection else 'Raster'}")
    print(f"  Files already existing: {existing_count}")
    print(f"  Files newly downloaded: {downloaded_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Total processed files: {existing_count + downloaded_count + skipped_count}")

    # Example usage of the dataset if any files exist
    if is_lakes_collection:
        # For Lakes data, check if any date directories exist
        date_dirs = [d for d in os.listdir(lakes_folder) if os.path.isdir(os.path.join(lakes_folder, d)) and d.isdigit()]
        if date_dirs:
            print(f"\nFound {len(date_dirs)} date directories in {lakes_folder}")
            # Count the number of Prior files
            prior_files_count = 0
            for date_dir in date_dirs:
                date_path = os.path.join(lakes_folder, date_dir)
                prior_files = [f for f in os.listdir(date_path) if "Prior" in f and 
                              (f.endswith('.shp') or f.endswith('.nc'))]
                prior_files_count += len(prior_files)
            
            print(f"Total Prior files: {prior_files_count}")
            
            # Try to open a sample file
            if prior_files_count > 0:
                # Find the first Prior file that's a shapefile (.shp) or NetCDF (.nc)
                for date_dir in date_dirs:
                    date_path = os.path.join(lakes_folder, date_dir)
                    prior_files = [f for f in os.listdir(date_path) if "Prior" in f and 
                                  (f.endswith('.shp') or f.endswith('.nc'))]
                    if prior_files:
                        example_file = os.path.join(date_path, prior_files[0])
                        print(f"Opening example file: {os.path.basename(example_file)}")
                        dataset = safe_open_dataset(example_file)
                        if dataset is not None:
                            print("Dataset opened successfully.")
                        break
        else:
            print(f"\nNo date directories found in {lakes_folder}")
    else:
        # For Raster data
        raster_files = [f for f in os.listdir(raster_folder) if f.endswith('.nc')]
        if raster_files:
            print(f"\nFound {len(raster_files)} SWOT raster files in {raster_folder}")
            example_file = os.path.join(raster_folder, raster_files[0])
            print(f"Opening example file: {os.path.basename(example_file)}")
            dataset = safe_open_dataset(example_file)
            if dataset is not None:
                print("Dataset opened successfully.")
        else:
            print(f"\nNo SWOT raster files found in {raster_folder}")
