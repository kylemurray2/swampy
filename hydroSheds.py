import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.mask import mask
import contextily as ctx

# Define San Francisco Bay Area bounding box (approximate coordinates)
sf_bbox = {
    'minx': -123.0,
    'miny': 37.0,
    'maxx': -121.0,
    'maxy': 38.5
}

# Load and crop vector data
def load_and_crop_vector(filepath, bbox):
    gdf = gpd.read_file(filepath)
    return gdf.cx[bbox['minx']:bbox['maxx'], bbox['miny']:bbox['maxy']]

# Load vector datasets
lakes = load_and_crop_vector('/Volumes/NAS_NC/haw/Documents/research/surfaceWater/HydroShedsData/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp', sf_bbox)
rivers = load_and_crop_vector('/Volumes/NAS_NC/haw/Documents/research/surfaceWater/HydroShedsData/HydroRIVERS_v10_na_shp/HydroRIVERS_v10_na.shp', sf_bbox)
land_fn = '/Volumes/NAS_NC/haw/Documents/research/surfaceWater/HydroShedsData/hydrosheds_land_mask/na_msk_3s.tif'
# Load and crop raster land mask
with rasterio.open(land_fn) as src:
    # Create geometry for cropping
    bbox_geometry = gpd.GeoDataFrame({
        'geometry': [box(sf_bbox['minx'], sf_bbox['miny'], 
                        sf_bbox['maxx'], sf_bbox['maxy'])],
        'id': [1]}, crs='EPSG:4326')
    
    # Crop raster
    land_mask, transform = mask(src, bbox_geometry.geometry, crop=True)

# Create three separate maps
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Map 1: Lakes
lakes.plot(ax=ax1, color='blue', alpha=0.6)
ctx.add_basemap(ax1, crs=lakes.crs)
ax1.set_title('HydroLAKES')

# Map 2: Rivers
rivers.plot(ax=ax2, color='blue', linewidth=0.5)
ctx.add_basemap(ax2, crs=rivers.crs)
ax2.set_title('HydroRIVERS')

# Map 3: Land Mask
im = ax3.imshow(land_mask[0], extent=[sf_bbox['minx'], sf_bbox['maxx'], 
                                     sf_bbox['miny'], sf_bbox['maxy']])
plt.colorbar(im, ax=ax3)
ax3.set_title('Land Mask')

# Adjust layout and display
plt.tight_layout()
plt.show()
