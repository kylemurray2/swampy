# Import necessary libraries
import ee
import geemap

# Initialize the Google Earth Engine API
ee.Initialize()

# Define the area of interest (AOI) for the Island of Hawaii
aoi = ee.Geometry.BBox(-156.0, 18.5, -154.5, 20.5)  # Bounding box around the Island of Hawaii

# Define the dataset - using MODIS Land Surface Temperature (LST) product
dataset = ee.ImageCollection('MODIS/006/MOD11A1') \
    .filterBounds(aoi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .select('LST_Day_1km')  # Selecting daytime land surface temperature

# Convert temperature from Kelvin to Celsius
dataset_celsius = dataset.map(lambda image: image.multiply(0.02).subtract(273.15).copyProperties(image, image.propertyNames()))

# Calculate the mean temperature over the year
mean_temperature = dataset_celsius.mean().clip(aoi)

# Set up visualization parameters
vis_params = {
    'min': 20,
    'max': 40,
    'palette': ['blue', 'green', 'yellow', 'red']
}

# Create a map to visualize the data
Map = geemap.Map()
Map.centerObject(aoi, 8)
Map.addLayer(mean_temperature, vis_params, 'Mean Land Surface Temperature (Celsius)')

# Export the data to Google Drive
export_task = ee.batch.Export.image.toDrive(
    image=mean_temperature,
    description='Hawaii_Heat_Data',
    folder='EarthEngineExports',
    fileNamePrefix='hawaii_heat_data',
    scale=1000,
    region=aoi,
    fileFormat='GeoTIFF'
)
export_task.start()

print("Export task started: Hawaii heat data")
