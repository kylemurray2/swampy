
import planetary_computer
from pystac_client import Client
from datetime import datetime
import os
import asyncio
import aiohttp
import concurrent.futures

async def download_item(session, href, output_path):
    """Download a single STAC item."""
    signed_href = planetary_computer.sign(href)
    async with session.get(signed_href) as response:
        if response.status == 200:
            with open(output_path, 'wb') as f:
                f.write(await response.read())
            print(f"Downloaded: {output_path}")
        else:
            print(f"Failed to download {href}: {response.status}")

async def download_items(items, output_dir):
    """Download multiple STAC items concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in items:
            href = item.assets["water_body"].href
            filename = f"water_body_{item.id}.tif"
            output_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(output_path):  # Skip if file exists
                task = download_item(session, href, output_path)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)

def download_water_bodies_data(
    bbox: list,  # [min_lon, min_lat, max_lon, max_lat]
    start_date: str,  # Format: 'YYYY-MM-DD'
    end_date: str,    # Format: 'YYYY-MM-DD'
    output_dir: str
):
    """
    Download Copernicus Water Bodies data using STAC API.
    
    Args:
        bbox (list): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory to save downloaded files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the STAC client for Microsoft Planetary Computer
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # Search for water bodies data
    search = catalog.search(
        collections=["io-water-body"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}"
    )

    # Get all items
    items = list(search.get_items())
    print(f"Found {len(items)} items")

    # Download items using asyncio
    asyncio.run(download_items(items, output_dir))

if __name__ == "__main__":
    # Example usage
    bbox = [-122.5, 37.5, -122.0, 38.0]  # Example: San Francisco Bay Area
    start_date = "1970-01-01"
    end_date = "2023-12-31"
    output_dir = "water_bodies_data"

    download_water_bodies_data(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
