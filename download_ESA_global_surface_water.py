import urllib.request
import os

def download_water_data(
    destination_folder="./water_data",
    dataset_name="occurrence",
):
    """
    Download water data for the SF Bay Area (130W_30N tiles).
    
    Args:
        destination_folder (str): Where to save the downloaded files
        dataset_name (str): Name of the dataset to download
    """
    # Fixed coordinates for SF Bay Area
    lng = "130W"
    lat = "30N"
    
    years = range(1984, 2022)  # 1984 to 2021
    total_files = len(years)
    counter = 1
    
    print(f"Downloading {total_files} files for tile {lng}_{lat}")

    for year in years:
        filename = f"{dataset_name}_{lng}_{lat}v1_4_{year}.tif"
        file_path = os.path.join(destination_folder, filename)
        
        if os.path.exists(file_path):
            print(f"{file_path} already exists - skipping")
        else:
            url = f"http://storage.googleapis.com/global-surface-water/downloads{year}/{dataset_name}/{filename}"
            try:
                code = urllib.request.urlopen(url).getcode()
                if code == 200:
                    print(f"Downloading {url} ({counter}/{total_files})")
                    urllib.request.urlretrieve(url, file_path)
                else:
                    print(f"{url} returned status code {code}")
            except urllib.error.HTTPError as e:
                print(f"{url} not found (Error {e.code})")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
        counter += 1

if __name__ == "__main__":
    download_water_data(
        destination_folder="/d/surfaceWater/sf_water_data",
        dataset_name="occurrence"
    )
    
    # Example usage:
    
    # Download data for the continental United States (roughly)
    # bounds_usa = [-125, 25, -65, 50]  # [min_lon, min_lat, max_lon, max_lat]
    # download_water_data(
    #     destination_folder="./usa_water_data",
    #     dataset_name="occurrence",
    #     bounds=bounds_usa
    # )
    
    # Or use default bounds (whole world)
    # download_water_data()
    
    # Or download data for a specific region (e.g., Western Europe)
    # bounds_europe = [-10, 35, 30, 60]
    # download_water_data(
    #     destination_folder="./europe_water_data",
    #     dataset_name="occurrence",
    #     bounds=bounds_europe
    # )