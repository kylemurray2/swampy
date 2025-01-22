import urllib.request
import os

def download_water_data(
    destination_folder="./water_data",
    dataset_name="occurrence",
    bounds=None  # Format: [min_lon, min_lat, max_lon, max_lat]
):
    """
    Download water data for a specific geographic bounding box.
    
    Args:
        destination_folder (str): Where to save the downloaded files
        dataset_name (str): Name of the dataset to download
        bounds (list): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
                      Values should be in degrees (-180 to 180 for longitude, -90 to 90 for latitude)
    """
    # Set default bounds if none provided (whole world)
    if bounds is None:
        bounds = [-180, -50, 180, 90]
    
    # Validate bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and
            -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Invalid bounds. Longitude: -180 to 180, Latitude: -90 to 90")

    # Ensure destination folder ends with slash and exists
    if not destination_folder.endswith("/"):
        destination_folder += "/"
    if not os.path.exists(destination_folder):
        print(f"Creating folder {destination_folder}")
        os.makedirs(destination_folder)

    # Generate longitude and latitude ranges based on bounds
    # Round to nearest 10 degrees (tile size)
    min_lon_tile = (min_lon // 10) * 10
    max_lon_tile = ((max_lon + 9) // 10) * 10
    min_lat_tile = (min_lat // 10) * 10
    max_lat_tile = ((max_lat + 9) // 10) * 10

    # Generate longitude range
    longs = []
    for lon in range(int(min_lon_tile), int(max_lon_tile + 10), 10):
        if lon < 0:
            longs.append(f"{abs(lon)}W")
        else:
            longs.append(f"{lon}E")

    # Generate latitude range
    lats = []
    for lat in range(int(min_lat_tile), int(max_lat_tile + 10), 10):
        if lat < 0:
            lats.append(f"{abs(lat)}S")
        else:
            lats.append(f"{lat}N")

    file_count = len(longs) * len(lats)
    counter = 1

    print(f"Downloading {file_count} tiles for area: {bounds}")

    for lng in longs:
        for lat in lats:
            filename = f"{dataset_name}_{lng}_{lat}v1_4_2021.tif"
            file_path = destination_folder + filename
            
            if os.path.exists(file_path):
                print(f"{file_path} already exists - skipping")
            else:
                url = f"http://storage.googleapis.com/global-surface-water/downloads2021/{dataset_name}/{filename}"
                try:
                    code = urllib.request.urlopen(url).getcode()
                    if code == 200:
                        print(f"Downloading {url} ({counter}/{file_count})")
                        urllib.request.urlretrieve(url, file_path)
                    else:
                        print(f"{url} returned status code {code}")
                except urllib.error.HTTPError as e:
                    print(f"{url} not found (Error {e.code})")
                except Exception as e:
                    print(f"Error downloading {url}: {str(e)}")
            counter += 1

if __name__ == "__main__":
    # Example usage:
    
    # Download data for the continental United States (roughly)
    bounds_usa = [-125, 25, -65, 50]  # [min_lon, min_lat, max_lon, max_lat]
    download_water_data(
        destination_folder="./usa_water_data",
        dataset_name="occurrence",
        bounds=bounds_usa
    )
    
    # Or use default bounds (whole world)
    # download_water_data()
    
    # Or download data for a specific region (e.g., Western Europe)
    # bounds_europe = [-10, 35, 30, 60]
    # download_water_data(
    #     destination_folder="./europe_water_data",
    #     dataset_name="occurrence",
    #     bounds=bounds_europe
    # )