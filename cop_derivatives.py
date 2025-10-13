#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from SARTS import makeMap


def create_water_frequency(gdf_dsw, image_type='image_wtr'):
    """
    Create a water frequency map from temporal DSW data.
    
    Each pixel's value (0-100) represents the percentage of time 
    water (both permanent and partial) was present.
    
    Parameters:
      gdf_dsw (GeoDataFrame): A GeoDataFrame where each row contains a water image.
      image_type (str): Column name that contains the water image.
    
    Returns:
      np.ndarray: 2D array with water frequency percentages.
    """
    # Stack all images into a 3D array of shape (time, rows, cols)
    images = np.stack([row[image_type] for _, row in gdf_dsw.iterrows()])
    
    # Identify water pixels (water detection: permanent = 1, partial water = 2)
    water_mask = (images == 1) | (images == 2)
    
    # Identify valid pixels (exclude no-data values coded as 255)
    valid_mask = images < 250

    # Count water and valid pixels along the temporal dimension
    water_count = np.sum(water_mask, axis=0)
    valid_count = np.sum(valid_mask, axis=0)
    
    # Compute frequency in percentage; if no valid data, then 0.
    frequency = np.where(valid_count > 0, (water_count / valid_count) * 100, 0)
    
    return frequency

def create_water_persistence(gdf_dsw, image_type='image_wtr', threshold=95):
    """
    Create a water persistence map indicating:
      0 -> No water,
      1 -> Sometimes water, and
      2 -> Always water.
    
    Uses a threshold such that if water is present in at least the given 
    percentage (default 99%) of valid observations, the pixel is treated as always water.
    
    Parameters:
      gdf_dsw (GeoDataFrame): Contains temporal DSW data.
      image_type (str): Column name for water imagery.
      threshold (float): Percentage threshold (0-100) to classify "always water".
      
    Returns:
      np.ndarray: 2D array with persistence values.
    """
    frequency = create_water_frequency(gdf_dsw, image_type)
    persistence = np.zeros(frequency.shape, dtype=np.uint8)
    # Some water: frequency is > 0 but below the threshold.
    persistence[(frequency > 0) & (frequency < threshold)] = 1  
    # Always water: frequency is at or above the threshold.
    persistence[frequency >= threshold] = 2                  
    return persistence

def create_water_change(gdf_dsw, image_type='image_wtr', n_start=3, n_end=3):
    """
    Create a water change map by comparing the start and end periods.
    
    The function computes the difference (last period minus first period) 
    in water frequency, considering both permanent (1) and partial water (2) as water.
    The output change ranges between -1 and 1.
    
    Parameters:
      gdf_dsw (GeoDataFrame): Contains temporal DSW data.
      image_type (str): Column name for water imagery.
      n_start (int): Number of images from the start period.
      n_end (int): Number of images from the end period.
      
    Returns:
      np.ndarray: 2D array with change values.
      
    Raises:
      ValueError: If there are not enough images in gdf_dsw.
    """
    if len(gdf_dsw) < n_start or len(gdf_dsw) < n_end:
        raise ValueError("Not enough temporal images in gdf_dsw to compute water change.")
        
    # Stack images for the start and end periods
    start_images = np.stack([row[image_type] for _, row in gdf_dsw.iloc[:n_start].iterrows()])
    end_images = np.stack([row[image_type] for _, row in gdf_dsw.iloc[-n_end:].iterrows()])
    
    # Consider both permanent (1) and partial water (2) as water
    start_water = ((start_images == 1) | (start_images == 2)).astype(np.float32)
    end_water = ((end_images == 1) | (end_images == 2)).astype(np.float32)
    
    # Frequency is defined as the mean of water presence in each period
    start_freq = np.mean(start_water, axis=0)
    end_freq = np.mean(end_water, axis=0)
    
    change = end_freq - start_freq
    return change

def compute_water_transitions(initial, final):
    """
    Compute transitions between initial and final water states.
    
    Transition codes:
      1: Permanent water remains
      2: New permanent water
      3: Lost permanent water
      4: Seasonal water remains
      5: New seasonal water
      6: Lost seasonal water
      7: Permanent to seasonal
      8: Seasonal to permanent
      9: Other valid changes
      nan: No data or invalid
    """
    WATER_CLASS = 1
    PARTIAL_WATER_CLASS = 2
    LAND_CLASS = 0
    NO_DATA = 255
    
    # Initialize transitions array with NaN (using float type)
    transitions = np.full(initial.shape, np.nan, dtype=np.float32)
    
    # Only process valid pixels (both initial and final not NO_DATA)
    valid = (initial != NO_DATA) & (final != NO_DATA)
    
    # Permanent water remains
    mask = valid & (initial == WATER_CLASS) & (final == WATER_CLASS)
    transitions[mask] = 0
    # New permanent water
    mask = valid & (initial == LAND_CLASS) & (final == WATER_CLASS)
    transitions[mask] = 1
    # Lost permanent water
    mask = valid & (initial == WATER_CLASS) & (final == LAND_CLASS)
    transitions[mask] =2
    # Seasonal (partial) water remains
    mask = valid & (initial == PARTIAL_WATER_CLASS) & (final == PARTIAL_WATER_CLASS)
    transitions[mask] = 3
    # New seasonal water
    mask = valid & (initial == LAND_CLASS) & (final == PARTIAL_WATER_CLASS)
    transitions[mask] = 4
    # Lost seasonal water
    mask = valid & (initial == PARTIAL_WATER_CLASS) & (final == LAND_CLASS)
    transitions[mask] = 5
    # Permanent to seasonal transition
    mask = valid & (initial == WATER_CLASS) & (final == PARTIAL_WATER_CLASS)
    transitions[mask] = 6
    # Seasonal to permanent transition
    mask = valid & (initial == PARTIAL_WATER_CLASS) & (final == WATER_CLASS)
    transitions[mask] = 7
    # Any other valid changes (e.g., land remains land: 0 -> 0)
    mask = valid & (np.isnan(transitions))
    transitions[mask] = np.nan
    
    return transitions

def combine_backup(img_primary, img_backup, threshold=250):
    """
    Combine two images (numpy arrays) using a backup strategy:
      - For each pixel, if the primary image's value is greater than 'threshold' 
        (indicating clouds/no data), then use the backup image's value.
      - If the primary is valid (<= threshold), it is used regardless of backup.
      - If both values exceed the threshold, the pixel is set to np.nan.
    
    Parameters:
      img_primary (np.ndarray): Primary image array.
      img_backup (np.ndarray): Backup image array.
      threshold (int): Value above which a pixel is considered invalid.
      
    Returns:
      np.ndarray: Combined image as a float32 array.
    """
    combined = np.empty_like(img_primary, dtype=np.float32)
    valid_primary = img_primary <= threshold
    valid_backup = img_backup <= threshold

    # Use primary values when they are valid.
    combined[valid_primary] = img_primary[valid_primary]
    # Where primary is invalid but backup is valid, use the backup.
    use_backup = (~valid_primary) & valid_backup
    combined[use_backup] = img_backup[use_backup]
    # Where neither is valid, assign np.nan.
    combined[(~valid_primary) & (~valid_backup)] = np.nan
    return combined

def combine_three(img1, img2, img3, threshold=250):
    """
    Combine three images (numpy arrays) in order of priority.
    For each pixel:
      - If img1's value is valid (≤ threshold), use it.
      - Otherwise, if img2 is valid, use that.
      - Otherwise, if img3 is valid, use that.
      - Else, assign np.nan.
      
    Returns a float32 array.
    """
    combined = np.empty_like(img1, dtype=np.float32)
    valid1 = img1 <= threshold
    valid2 = img2 <= threshold
    valid3 = img3 <= threshold

    # Use the first image if valid.
    combined[valid1] = img1[valid1]
    # For pixels where the first image is invalid, try the second.
    choose2 = ~valid1 & valid2
    combined[choose2] = img2[choose2]
    # For pixels where both first and second are invalid, try the third.
    choose3 = ~valid1 & ~valid2 & valid3
    combined[choose3] = img3[choose3]
    # For pixels where none are valid, assign np.nan.
    combined[~(valid1 | valid2 | valid3)] = np.nan

    return combined

def plot_panels(gdf_dsw,zl=12, output_path=None, image_type='image_wtr'):
    """
    Create a figure with 6 panels (a–f) to display different water analyses.
    
    Parameters:
      gdf_dsw (GeoDataFrame): Temporal DSW data containing a 'geometry' column.
      output_path (str, optional): If provided, saves the figure to this path.
      image_type (str): Column name with water imagery data.
    
    Raises:
        ValueError: If gdf_dsw is empty or missing geometry column.
    """
    if 'geometry' not in gdf_dsw.columns or gdf_dsw.empty:
        raise ValueError("GeoDataFrame must contain a 'geometry' column and not be empty")

    fig = plt.figure(figsize=(15, 12))  # Increased figure height to accommodate colorbars
    
    # Set up the basemap
    bg = 'World_Imagery'
    bg2 = 'World_Shaded_Relief'
    url = f'https://server.arcgisonline.com/ArcGIS/rest/services/{bg}/MapServer/tile/{{z}}/{{y}}/{{x}}.png'
    image = cimgt.GoogleTiles(url=url)
    url2 = f'https://server.arcgisonline.com/ArcGIS/rest/services/{bg2}/MapServer/tile/{{z}}/{{y}}/{{x}}.png'
    image2 = cimgt.GoogleTiles(url=url2)
    img_crs = image.crs
    map_crs = ccrs.PlateCarree()
    # Get the geometry bounds for all panels
    poly = gdf_dsw.unary_union
    bounds = poly.bounds
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # [xmin, xmax, ymin, ymax]

    # Panel a: Location map with Esri World Imagery basemap
    ax1 = fig.add_subplot(231, projection=map_crs)
    ax1.set_extent(extent, crs=map_crs)
    ax1.add_image(image, zl, zorder=1)
    ax1.coastlines(resolution='50m', color='gray')
    ax1.add_geometries([poly], crs=img_crs, facecolor='none',
                       edgecolor='red', linewidth=2)
    ax1.set_title('a: Location Map', loc='left')
    
    # Panel b: Water persistence
    ax2 = fig.add_subplot(232, projection=map_crs)
    ax2.add_image(image2, zl, zorder=1)
    persistence = create_water_persistence(gdf_dsw, image_type)
    colors = ['white', 'pink', 'blue']  # 0: no water, 1: sometimes, 2: always
    cmap_persistence = ListedColormap(colors)
    norm_persistence = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], len(colors))
    im2 = ax2.imshow(persistence, extent=extent, transform=map_crs,
                     cmap=cmap_persistence, norm=norm_persistence, zorder=2)
    ax2.set_title('b: Water Persistence', loc='left')
    
    # Panel c: Water Change (using robust composites from first 3 and last 3 dates)
    ax3 = fig.add_subplot(233, projection=map_crs)

    # First get the persistence classification
    persistence = create_water_persistence(gdf_dsw, image_type)
    
    # Create robust composites as before
    initial1 = gdf_dsw.iloc[0][image_type]
    initial2 = gdf_dsw.iloc[1][image_type]
    initial3 = gdf_dsw.iloc[2][image_type]
    robust_initial = combine_three(initial1, initial2, initial3, threshold=250)

    final1 = gdf_dsw.iloc[-1][image_type]
    final2 = gdf_dsw.iloc[-2][image_type]
    final3 = gdf_dsw.iloc[-3][image_type]
    robust_final = combine_three(final1, final2, final3, threshold=250)

    # Determine water presence
    initial_water = ((robust_initial == 1) | (robust_initial == 2)).astype(np.float32)
    final_water   = ((robust_final   == 1) | (robust_final   == 2)).astype(np.float32)

    # Compute water change
    water_change = final_water - initial_water

    # Mask out changes where:
    # 1. Pixels are always land (as before)
    # 2. Pixels are persistent water (persistence == 2)
    always_land_mask = (initial_water == 0) & (final_water == 0)
    persistent_water_mask = (persistence == 2)
    water_change[always_land_mask | persistent_water_mask] = np.nan

    colors_change = ['purple', 'black', 'green']
    cmap_change = ListedColormap(colors_change)
    cmap_change.set_bad(color='none')

    im3 = ax3.imshow(water_change, extent=extent, transform=map_crs,
                     cmap=cmap_change, vmin=-1, vmax=1)
    ax3.set_title('c: Water Change', loc='left')
    
    # Panel d: Water frequency
    ax4 = fig.add_subplot(234, projection=map_crs)
    ax4.add_image(image2, zl, zorder=1)
    frequency = create_water_frequency(gdf_dsw, image_type)
    # Set frequency to NaN for values less than 5%
    frequency[frequency < 5] = np.nan
    cmap_freq = plt.cm.RdYlBu.copy()
    cmap_freq.set_bad(color='none')  # Make non-values transparent
    im4 = ax4.imshow(frequency, extent=extent, transform=map_crs,
                     cmap=cmap_freq, vmin=5, vmax=100, zorder=2)
    ax4.set_title('d: Water Frequency', loc='left')
    
    # Panel e: Seasonal vs Permanent water distribution
    ax5 = fig.add_subplot(235, projection=map_crs)
    ax5.add_image(image2, zl, zorder=1)
    seasonal = np.where(frequency < 100, frequency, 0)
    permanent = np.where(frequency == 100, 1, 0)
    water_type = np.zeros_like(frequency, dtype=np.uint8)
    water_type[seasonal > 0] = 1  # Seasonal water
    water_type[permanent > 0] = 2  # Permanent water
    colors_type = ['white', 'lightblue', 'darkblue']
    cmap_type = ListedColormap(colors_type)
    norm_type = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], len(colors_type))
    im5 = ax5.imshow(water_type, extent=extent, transform=map_crs,
                     cmap=cmap_type, norm=norm_type, zorder=2)
    ax5.set_title('e: Seasonal vs Permanent', loc='left')
    
    # Panel f: Detailed Transitions using robust composites (first 3 and last 3 dates)
    ax6 = fig.add_subplot(236, projection=map_crs)
    
    # Get persistence classification (already calculated for panel c)
    persistence = create_water_persistence(gdf_dsw, image_type)
    
    # Create robust initial state using first three dates in priority order
    initial1 = gdf_dsw.iloc[0][image_type]
    initial2 = gdf_dsw.iloc[1][image_type]
    initial3 = gdf_dsw.iloc[2][image_type]
    robust_initial = combine_three(initial1, initial2, initial3, threshold=250)

    # Create robust final state using last three dates in priority order
    final1 = gdf_dsw.iloc[-1][image_type]
    final2 = gdf_dsw.iloc[-2][image_type]
    final3 = gdf_dsw.iloc[-3][image_type]
    robust_final = combine_three(final1, final2, final3, threshold=250)

    # Compute transitions using the robust composites
    transitions = compute_water_transitions(robust_initial, robust_final)
    
    # Mask out transitions where water is persistent
    persistent_water_mask = (persistence == 2)
    transitions[persistent_water_mask] = 1  # Set to "Permanent water remains" for persistent water
    
    colors_trans = ['blue',      # Permanent water remains
                    'green',     # New permanent water
                    'red',       # Lost permanent water
                    'lightblue', # Seasonal water remains
                    'lime',      # New seasonal water
                    'orange',    # Lost seasonal water
                    'yellow',    # Permanent to seasonal
                    'gray',      # Seasonal to permanent
                    'pink']      # Ephemeral transitions
    
    cmap_trans = ListedColormap(colors_trans)
    norm_trans = BoundaryNorm(np.arange(0.5, 9.5, 1), len(colors_trans))
    im6 = ax6.imshow(transitions, extent=extent, transform=map_crs,
                     cmap=cmap_trans, norm=norm_trans)
    ax6.set_title('f: Detailed Transitions', loc='left')

    # Adjust subplot spacing to make room for colorbars
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
  
    # Add colorbars below each panel
    for ax, im, label in [
        (ax2, im2, 'Sometimes water → Always water'),
        (ax3, im3, 'Decrease → Increase'),
        (ax4, im4, 'Water Frequency (%)'),
        (ax5, im5, 'Seasonal → Permanent'),
        (ax6, im6, 'Transition Codes')
    ]:
        # Get the position of the subplot
        pos = ax.get_position()
        # Calculate colorbar position (centered, 70% width)
        cbar_width = pos.width * 0.7
        cbar_left = pos.x0 + pos.width * 0.15  # Centers the colorbar
        cbar_bottom = pos.y0 - 0.08  # Place below the subplot
        cbar_height = 0.02  # Height of colorbar
        
        # Create new axes for colorbar
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        fig.colorbar(im, cax=cax, orientation='horizontal', label=label)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    fig = plt.figure(figsize=(15, 12))  # Increased figure height to accommodate colorbars
    ax6 = fig.add_subplot(111, projection=map_crs)
    ax6.add_image(image, zl, zorder=1)
    ax6.imshow(transitions, extent=extent, transform=map_crs,
                     cmap=cmap_trans, norm=norm_trans,zorder=2)
    ax6.set_title('f: Detailed Transitions', loc='left')



    plt.show()

def generate_dummy_gdf_dsw(num_images=6, rows=50, cols=50, image_type='image_wtr'):
    """
    Generate a dummy GeoDataFrame with random DSW data for testing.
    
    Each row contains a simulated image (np.ndarray) of water classes:
     - 0: Land
     - 1: Permanent water
     - 2: Seasonal (partial) water
     - 255: No data
    """
    # Define probability distribution for each class
    classes = [0, 1, 2, 255]
    probs = [0.7, 0.15, 0.1, 0.05]
    
    data = []
    for i in range(num_images):
        image = np.random.choice(classes, size=(rows, cols), p=probs)
        entry = {image_type: image, 'date': f"2020-01-{i+1:02d}"}
        data.append(entry)
    
    gdf = gpd.GeoDataFrame(data)
    # Create a sample geometry (could be any valid polygon)
    geom = box(-100, 30, -90, 40)  # Example area in central US
    gdf['geometry'] = geom
    return gdf

def plot_interactive_timeseries(gdf_dsw, image_type='image_wtr'):
    """
    Plot an interactive map from the given GeoDataFrame. When the user clicks on the map,
    a new plot pops up showing the time series of water classification values at the clicked pixel.
    
    If any value is > 200, it is set to NaN (and not shown on the time series plot).
    
    Parameters:
      gdf_dsw (GeoDataFrame): Temporal DSW data with a 'geometry' column and an image column.
      image_type (str): Column in gdf_dsw containing the water imagery data (e.g., water classification).
    """
    # Get union geometry to define extent (bounds: minx, miny, maxx, maxy)
    poly = gdf_dsw.unary_union
    bounds = poly.bounds  
    # Rearranged extent expected by imshow: [minx, maxx, miny, maxy]
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    
    # Get image dimensions from the first image
    sample_img = gdf_dsw.iloc[0][image_type]
    rows, cols = sample_img.shape

    # Create the base map figure.
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the sample image with the proper extent.
    im = ax.imshow(sample_img, extent=extent, origin='upper')
    ax.set_title("Interactive Map - Click for Time Series")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    def on_click(event):
        # Only act if we click inside the axis and valid xdata/ydata are provided.
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            # Convert map coordinates to pixel indices.
            minx, maxx, miny, maxy = extent
            col = int((x - minx) / (maxx - minx) * cols)
            row = int((maxy - y) / (maxy - miny) * rows)
            
            if row < 0 or row >= rows or col < 0 or col >= cols:
                print("Clicked outside image area.")
                return
            
            # Gather the time series for the clicked pixel from each image.
            time_series = []
            times = []
            for idx, row_data in gdf_dsw.iterrows():
                img = row_data[image_type]
                value = img[row, col]
                # If the value is greater than 200, set it to np.nan so it won't be shown.
                if value > 200:
                    value = np.nan
                time_series.append(value)
                # Attempt to use a 'date' field; if not available, default to the index.
                times.append(row_data.get('date', idx))
            
            # Plot the time series in a new figure.
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            ax_ts.plot(times, time_series, marker='o', linestyle='-')
            ax_ts.set_title(f"Time Series at Pixel (row={row}, col={col})")
            ax_ts.set_xlabel("Time")
            ax_ts.set_ylabel("Water Classification")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Connect the click event to our handler.
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.show()

if __name__ == '__main__':
    # Option 1: Using dummy data for testing.
    # gdf_dsw = generate_dummy_gdf_dsw()
    # plot_panels(gdf_dsw, output_path='water_analysis.png')
    
    # Option 2: Using your own data.
    # import geopandas as gpd
    # gdf_dsw = gpd.read_file('path/to/your/data.gpkg')
    # plot_panels(gdf_dsw, output_path='water_analysis.png')
    os.chdir('/d/surfaceWater/napa_marshes')
    pickle_path = 'data/DSW/monthly_dsw_mosaic.pkl'

    # Check if pickle file exists
    if os.path.exists(pickle_path):
        print(f"Loading monthly DSW mosaic from {pickle_path}")
        gdf_dsw = pd.read_pickle(pickle_path)
    else:
        print(f"Pickle file {pickle_path} does not exist")

    plot_panels(gdf_dsw, output_path='water_analysis.png')

    # Generate or load your gdf_dsw (here we use a dummy generator for demonstration)
    plot_interactive_timeseries(gdf_dsw)