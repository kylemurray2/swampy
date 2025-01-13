import earthaccess
from pystac_client import Client

# Authenticate with NASA Earthdata
earthaccess.login()

# Define search parameters
temporal_range = ("2023-08-18", "2023-08-23")
# For STAC
bbox = [-160.0, 18.5, -154.5, 22.5]
# For earthaccess (west, south, east, north)
west, south, east, north = bbox

print("\n=== Using earthaccess ===")
# Search using earthaccess
ea_results = earthaccess.search_data(
    short_name="OPERA_L3_DSWX-HLS_V1",
    bounding_box=(west, south, east, north),
    temporal=(temporal_range[0], temporal_range[1]),
    count=100
)
print(f"earthaccess results: Found {len(ea_results)} items")
for result in ea_results[:3]:  # Show first 3 results
    print(f"\nGranule ID: {result.id}")
    print(f"Name: {result.name}")
    print(f"Time Start: {result.time_start}")
    print(f"Time End: {result.time_end}")
    print(f"Download URL: {result.data_links()[0] if result.data_links() else 'No direct download link'}")

print("\n=== Using STAC ===")
# Initialize STAC client for PO.DAAC catalog
stac_url = "https://cmr.earthdata.nasa.gov/cloudstac/POCLOUD"
client = Client.open(stac_url)

# Search for data using STAC with correct collection ID
search = client.search(
    collections=["OPERA_L3_DSWX-HLS_V1_1.0"],  # Updated collection ID
    bbox=bbox,
    datetime=f"{temporal_range[0]}/{temporal_range[1]}"
)

# Get the STAC results
stac_results = list(search.get_items())
print(f"STAC results: Found {len(stac_results)} items")
for item in stac_results[:3]:  # Show first 3 results
    print(f"\nTitle: {item.id}")
    print(f"Datetime: {item.datetime}")
    print(f"Assets: {list(item.assets.keys())}")
