import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import bounds
from rasterio.transform import array_bounds
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
import makeMap
import glob
import gc
import rasterio.warp
from rasterio.mask import mask
from shapely import box, wkt
import cartopy.crs as ccrs


def plot_area_of_interest(area_name,minlon,maxlon,minlat,maxlat,zoomLevel=12,title='Area of interest'):
    """Plot the area of interest with a red boundary box."""

    bg = 'World_Imagery'
    pad = .1
    figsize = (10, 8)   
    
    ax, fig, map_crs = makeMap.mapBackground(minlon, maxlon, minlat, maxlat, 
                                           zoomLevel=zoomLevel, title=title, 
                                           bg=bg, pad=pad, scalebar=5, 
                                           borders=False, figsize=figsize)
    
    ax.plot([minlon, maxlon, maxlon, minlon, minlon],
            [minlat, minlat, maxlat, maxlat, minlat], 
            color='red', transform=map_crs)


def plot_dsw(gdf, index=0, image_type='wtr-2', zoomLevel=8,title='Surface Water'):
    '''
    plot the dsw image for a given index and image type
    
    :param image_type: One of 'conf', 'wtr', or 'wtr-2'
    '''

    # Convert to GeoDataFrame if needed
    if not isinstance(gdf, gpd.GeoDataFrame):
        # Create geometry from bounds if available
        if 'bounds' in gdf.columns:
            gdf['geometry'] = gdf['bounds'].apply(
                lambda b: box(b[0], b[1], b[2], b[3])
            )
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
        else:
            raise ValueError("Input DataFrame must either be a GeoDataFrame or have a 'bounds' column")

    bounds = gdf.geometry.iloc[index].bounds
    print(bounds)
    minlon, minlat, maxlon, maxlat = bounds[0], bounds[1], bounds[2], bounds[3]

    # Correct color mapping:
    # 0=land, 1=water, 2=partial water, 252=snow/ice, 253=cloud, 254=ocean, 255=no data
    colors = ['#ffeb3b', 'blue', 'lightblue', 'gray', 'white', 'darkblue', 'none']
    n_bins = 7
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 252.5, 253.5, 254.5, 255.5], n_bins)

    bg = 'World_Imagery'
    pad = 0
    figsize = (10, 8)   

    # Get the background map and figure
    ax, fig, data_crs = makeMap.mapBackground(minlon, maxlon, minlat, maxlat, 
                                            zoomLevel=zoomLevel, title=title, 
                                            bg=bg, pad=pad, scalebar=None, 
                                            borders=False, figsize=figsize)

    print(f'data_crs is {data_crs}')

    # Determine which column to use based on data type
    if 'image_wtr-2' in gdf.columns:  # DSW data
        image_data = gdf[f'image_{image_type}'].iloc[index]
    elif image_type == 'water_class':  # SWOT data
        image_data = gdf['water_class'].iloc[index]
    else:  # Try using the specified column directly
        image_data = gdf[image_type].iloc[index]

    im = ax.imshow(image_data, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], 
                   cmap=cmap, norm=norm, alpha=0.7,transform=ccrs.PlateCarree(),zorder=20)

    # Create colorbar showing only the relevant categories with their colors
    cmap_categories = ListedColormap(['#ffeb3b', 'blue', 'lightblue', 'gray', 'white', 'darkblue'])
    
    # Create boundaries that match the data values
    boundaries = [-0.5, 0.5, 1.5, 2.5, 252.5, 253.5, 254.5]
    norm_categories = BoundaryNorm(boundaries, len(colors)-1)  # -1 for 'none' color
    
    # Create tick locations at the center of each color band
    tick_locs = [0, 1, 2, 252, 253, 254]
    tick_labels = ['Land', 'Water', 'Partial Water', 'Ice', 'Cloud', 'Ocean']
    
    scalar_mappable = cm.ScalarMappable(norm=norm_categories, cmap=cmap_categories)
    cbar = plt.colorbar(scalar_mappable, ax=ax, label='Water Class', 
                       ticks=tick_locs, shrink=0.5)
    cbar.ax.set_yticklabels(tick_labels)
    
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.show()




def extract_dsw(bbox=None, geometry=None, mosaics_folder='mosaics', land_cover_file='path/to/land_cover.tif'):
    """
    Extract surface water information for a given bounding box or polygon, masking out ocean areas.
    Now includes three types of mosaic files: confidence, water, and water-2.
    
    :param bbox: Tuple of (minx, miny, maxx, maxy) in lat/lon coordinates
    :param geometry: A shapely geometry object defining the area of interest
    :param mosaics_folder: Folder containing the DSW mosaics
    :param land_cover_file: Path to the land cover raster file
    :return: GeoDataFrame with surface water information for each time step
    """
    if geometry is not None:
        pass  # Use the provided geometry directly
    elif bbox:
        geometry = box(*bbox)
    else:
        raise ValueError("Either bbox or geometry must be provided.")
    
    # Get all mosaic files and sort them by date
    mosaic_types = ['mosaic_conf.tif', 'mosaic_wtr.tif', 'mosaic_wtr-2.tif']
    dates = set()
    for mosaic_type in mosaic_types:
        dates.update([os.path.dirname(f).split('/')[-1] 
                     for f in glob.glob(f'{mosaics_folder}/2*/{mosaic_type}')])
    dates = sorted(list(dates))
    
    # List to store results
    results = []
    
    for date in dates:
        print(f'Processing date: {date}')
        try:
            date_data = {'date': date}
            
            for mosaic_type in mosaic_types:
                mosaic_file = f'{mosaics_folder}/{date}/{mosaic_type}'
                if not os.path.exists(mosaic_file):
                    print(f"Missing file: {mosaic_file}")
                    continue
                
                with rasterio.open(mosaic_file) as src:
                    # Read the entire image
                    out_image = src.read(1)  # Read first band
                    
                    # Create geometry from the raster bounds
                    raster_bounds = src.bounds
                    raster_geometry = box(raster_bounds.left, raster_bounds.bottom, 
                                       raster_bounds.right, raster_bounds.top)
                    
                    # Add the image to the date_data dictionary
                    # Remove 'mosaic_' and '.tif' from the key name
                    key = mosaic_type.replace('mosaic_', '').replace('.tif', '')
                    date_data[f'image_{key}'] = out_image
                    
                    # Store the profile
                    if 'profile' not in date_data:
                        date_data['profile'] = src.profile.copy()
                        date_data['bounds'] = raster_bounds
                        date_data['geometry'] = raster_geometry
            
            # Only append if we have at least one image
            if len(date_data) > 3:  # more than just date, geometry, and profile
                results.append(date_data)
                
        except Exception as e:
            print(f"Error processing date {date}: {str(e)}")
            continue
    
    # Create a GeoDataFrame from the results
    gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")
    return gdf


def create_monthly_dsw_mosaic(gdf_dsw, root_dir):
    """
    Create a monthly mosaic GeoDataFrame from weekly DSW acquisitions.
    If a cached version exists, load it instead of recomputing.
    
    :param gdf_dsw: GeoDataFrame containing DSW data
    :param root_dir: Root directory for saving/loading the pickle file
    """
    # Define pickle file path
    pickle_path = os.path.join(root_dir, 'data', 'monthly_dsw_mosaic.pkl')
    
    # Check if pickle file exists
    if os.path.exists(pickle_path):
        print(f"Loading monthly DSW mosaic from {pickle_path}")
        return pd.read_pickle(pickle_path)
    
    print("Creating new monthly DSW mosaic...")
    try:
        # Convert 'date' to datetime if not already
        gdf_dsw['date'] = pd.to_datetime(gdf_dsw['date'])
        
        print(f"Initial data shape: {gdf_dsw.shape}")
        print(f"Date range: {gdf_dsw['date'].min()} to {gdf_dsw['date'].max()}")
        
        # Group by year and month
        grouped = gdf_dsw.groupby(gdf_dsw['date'].dt.to_period('M'))
        print(f"Number of groups: {len(grouped)}")
        
        monthly_records = []

        for period, group in grouped:
            print(f"\nProcessing period: {period}")
            print(f"Group size: {len(group)}")
            
            # Find the overall bounds for this month
            all_bounds = [row['bounds'] for _, row in group.iterrows()]
            overall_bounds = (
                min(b[0] for b in all_bounds),  # left
                min(b[1] for b in all_bounds),  # bottom
                max(b[2] for b in all_bounds),  # right
                max(b[3] for b in all_bounds)   # top
            )
            
            # Get the first valid profile to use as a template
            template_profile = None
            for _, row in group.iterrows():
                if 'profile' in row:
                    template_profile = row['profile'].copy()
                    break
            
            if template_profile is None:
                print("No valid profile found for this period")
                continue
            
            # Calculate pixel size from the template profile
            transform = template_profile['transform']
            pixel_width = abs(transform[0])
            pixel_height = abs(transform[4])
            
            # Calculate dimensions for the mosaic
            width = int((overall_bounds[2] - overall_bounds[0]) / pixel_width)
            height = int((overall_bounds[3] - overall_bounds[1]) / pixel_height)
            
            # Create new transform for the mosaic
            new_transform = rasterio.transform.from_bounds(
                overall_bounds[0], overall_bounds[1],
                overall_bounds[2], overall_bounds[3],
                width, height
            )
            
            # Update the profile for the mosaic
            mosaic_profile = template_profile.copy()
            mosaic_profile.update({
                'height': height,
                'width': width,
                'transform': new_transform
            })
            
            # Process each layer type independently
            layer_types = ['conf', 'wtr', 'wtr-2']
            monthly_images = {}
            
            for layer_type in layer_types:
                print(f"\nProcessing {layer_type} layer")
                
                # Initialize mosaic array with no data
                mosaic = np.full((height, width), 255, dtype=np.uint8)
                
                for idx, row in group.iterrows():
                    img_key = f'image_{layer_type}'
                    if img_key not in row:
                        continue
                        
                    img = row[img_key]
                    if not isinstance(img, np.ndarray):
                        continue
                    
                    # Calculate pixel coordinates in the mosaic
                    src_transform = row['profile']['transform']
                    src_bounds = row['bounds']
                    
                    # Calculate the window in the destination array
                    dst_window = rasterio.windows.from_bounds(
                        src_bounds[0], src_bounds[1],
                        src_bounds[2], src_bounds[3],
                        new_transform
                    )
                    
                    dst_window = dst_window.round_lengths().round_offsets()
                    
                    # Get the destination slice
                    dst_slice = (
                        slice(int(dst_window.row_off), int(dst_window.row_off + dst_window.height)),
                        slice(int(dst_window.col_off), int(dst_window.col_off + dst_window.width))
                    )
                    
                    # Resample the source image to match the destination window
                    resampled = np.zeros((int(dst_window.height), int(dst_window.width)), dtype=np.uint8)
                    rasterio.warp.reproject(
                        source=img,
                        destination=resampled,
                        src_transform=src_transform,
                        src_crs=template_profile['crs'],
                        dst_transform=new_transform,
                        dst_crs=template_profile['crs'],
                        resampling=rasterio.enums.Resampling.nearest
                    )
                    
                    # Update mosaic with prioritized values
                    current = mosaic[dst_slice]
                    new = resampled
                    
                    # Create masks for each priority class in the new data
                    mask_water = new == 1
                    mask_land = new == 0
                    mask_partial = new == 2
                    
                    # Apply updates in order of priority
                    # 1. Water (1) takes precedence over everything
                    mosaic[dst_slice][mask_water] = 1
                    
                    # 2. Land (0) takes precedence over partial water and other values
                    mask_land_update = mask_land & (current != 1)
                    mosaic[dst_slice][mask_land_update] = 0
                    
                    # 3. Partial water (2) takes precedence over non-priority values
                    mask_partial_update = mask_partial & (current != 1) & (current != 0)
                    mosaic[dst_slice][mask_partial_update] = 2
                    
                    # 4. For all other values, keep the existing value unless it's 255 (no data)
                    mask_other = ~(mask_water | mask_land | mask_partial) & (new != 255)
                    mask_other_update = mask_other & (current == 255)
                    mosaic[dst_slice][mask_other_update] = new[mask_other_update]
                
                monthly_images[img_key] = mosaic
            
            if monthly_images:
                record = {
                    'date': period.start_time,
                    **monthly_images,
                    'profile': mosaic_profile,
                    'bounds': overall_bounds,
                    'geometry': box(*overall_bounds)
                }
                monthly_records.append(record)
        
        if not monthly_records:
            print("Warning: No valid monthly records were created!")
            return None
            
        # Create GeoDataFrame
        result = gpd.GeoDataFrame(monthly_records, geometry='geometry')
        print(f"\nFinal mosaic shape: {result.shape}")
        
        # Save to pickle
        print(f"Saving monthly DSW mosaic to {pickle_path}")
        result.to_pickle(pickle_path)
        
        return result

    except Exception as e:
        print(f"\nError in create_monthly_dsw_mosaic: {str(e)}")
        raise

if __name__ == '__main__':
    area_name = 'milton'
    root_dir = '/d/surfaceWater/milton'
    
    # Define wkt_file and read geometry
    wkt_file = os.path.join(root_dir, 'map.wkt')
    with open(wkt_file, 'r') as file:
        polygon_wkt = file.read()
    geometry = wkt.loads(polygon_wkt)
    minlon, minlat, maxlon, maxlat = geometry.bounds
    
    # Get absolute paths
    mosaics_folder = os.path.abspath(os.path.join(root_dir, 'data'))
    land_cover_file = os.path.abspath(os.path.join(root_dir, 'resampled_landcover.tif'))

    # Plot the area of interest
    plot_area_of_interest(area_name, minlon, maxlon, minlat, maxlat, zoomLevel=8, title=f'Area of Interest - {area_name.title()}')

    print(f"Looking for mosaics in: {mosaics_folder}")
    print(f"Using land cover file: {land_cover_file}")

    # Now we can call extract_dsw with the geometry directly
    gdf = extract_dsw(geometry=geometry, mosaics_folder=mosaics_folder, land_cover_file=land_cover_file)
    
    # plot the first image
    idx=1
    plot_dsw(gdf, idx, image_type='image_wtr', zoomLevel=8)
    plot_dsw(gdf, idx, image_type='wtr', zoomLevel=8)
    plot_dsw_confidence(gdf, idx, zoomLevel=8)
    
    gdf_monthly = create_monthly_dsw_mosaic(gdf, root_dir)
    plot_dsw(gdf_monthly, 0, image_type='wtr', zoomLevel=6)

    confimg = gdf['image_conf'].iloc[idx]
    plt.imshow(confimg,vmin=0,vmax=10)
    plt.colorbar()
    plt.show()

    print(f"GeoDataFrame size: {len(gdf)}")
    if len(gdf) > 0:
        print(f"Available dates: {gdf.date.tolist()}")
        plot_dsw(gdf, 0, image_type='wtr')
    else:
        print("No data found for the specified area and time period")