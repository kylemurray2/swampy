import earthaccess, os, zipfile
from shapely.geometry import box


def search_data(short_name, start_time, end_time, min_lat, max_lat, min_lon, max_lon):
    """
    Function to search SWOT data from NASA's Earthdata system
    
    Parameters:
    short_name (str): The short name of the dataset
    start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format
    end_time (str): End time in 'YYYY-MM-DD HH:MM:SS' format
    min_lat (float): Minimum latitude of the bounding box
    max_lat (float): Maximum latitude of the bounding box
    min_lon (float): Minimum longitude of the bounding box
    max_lon (float): Maximum longitude of the bounding box

    Returns:
    list: List of search results
    
    Info:
        Check out the different data products
        https://podaac.jpl.nasa.gov/SWOT?tab=mission-objectives&sections=about%2Bdata
    
    
    
    """
    # Define bounding box using Shapely
    bounding_box = box(min_lon, min_lat, max_lon, max_lat).bounds
    
    # Perform search using earthaccess
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(start_time, end_time),
        bounding_box=bounding_box
    )
    
    return results

def download_data(results, download_path):
    """
    Function to download data based on search results
    
    Parameters:
    results (list): Search results from search_data function
    download_path (str): Directory to download the data
    
    Returns:
    None
    """
    if not results:
        print("No results found to download.")
        return
    
    # Download data using earthaccess
    earthaccess.download(results, download_path)
    print(f"Data downloaded to {download_path}")

def list_available_swot_products():
    """
    Function to list all available SWOT data products

    Returns:
    dict: Dictionary of available SWOT data products with descriptions
    """
    swot_products = {
        "SWOT_L2_HR_Raster_2.0": "Rasterized water surface elevation and inundation extent in fixed tiles at 100 m and 250 m resolution.",
        "SWOT_L2_LR_SSH_2.0": "Sea surface heights.",
        "SWOT_L2_HR_PIXC_2.0": "Water Mask Pixel Cloud NetCDF.",
        "SWOT_L2_HR_PIXCVec_2.0": "Water Mask Pixel Cloud Vector Attribute NetCDF.",
        "SWOT_L2_HR_RiverSP_2.0": "River Vector Shapefile.",
        "SWOT_L2_HR_RiverAvg_2.0": "SWOT Level 2 River Cycle-Averaged Data Product, Version C",
        "SWOT_L2_HR_LakeSP_2.0": "Lake Vector Shapefile.",
        "SWOT_L1B_LR_INTF_2.0": "SWOT Level 1B Low-Rate Interferogram Data Product, Version C."
    }
    return swot_products

def unzip(folder):
    
    for item in os.listdir(folder):  # loop through items in dir
        if item.endswith(".zip"):  # check for ".zip" extension
            # Extract the date from the file name
            date = item.split('_')[8][0:8]
            date_folder = os.path.join(folder, date)
            
            # Create the date folder if it doesn't exist
            if not os.path.exists(date_folder):
                os.mkdir(date_folder)
            
            # Create zipfile object
            zip_path = os.path.join(folder, item)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files into the date folder
                zip_ref.extractall(date_folder)
            
            # Optionally, remove the zip file after extraction
            os.remove(zip_path)  # Uncomment this if you want to delete the .zip file after extraction


# Example usage
if __name__ == "__main__":
    
    # Define parameters for search_____________________________________________
    folder     = "./swot/RasterData"
    dataset    = 'SWOT_L2_HR_Raster_2.0'
    dl         = True
    start_time = '2023-06-03 00:00:00'
    end_time   = '2029-04-02 23:59:59'
    # min_lat, max_lat, min_lon, max_lon = 34.54, 34.61, -119.82, -120  # cachuma
    min_lat, max_lat, min_lon, max_lon = 32, 49, -125, -114  # west US
    #__________________________________________________________________________
    
    # Authenticate to Earthdata
    auth = earthaccess.login()
    if not os.path.isdir(folder):
        os.mkdir(folder)

    results = search_data(dataset, start_time, end_time, min_lat, max_lat, min_lon, max_lon)
    # Check results
    if results:
        print(f"Found {len(results)} results.")
    else:
        print("No data found for the given parameters.")
    # Download data
    if dl:
        download_data(results, folder)
    else:
        print("skipping download")
              
    # List all available SWOT products
    swot_products = list_available_swot_products()
    print("Available SWOT Data Products:")
    for key, value in swot_products.items():
        print(f"{key}: {value}")

    unzip(folder)



