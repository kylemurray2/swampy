#!/usr/bin/env python3
"""
USGS Dynamic Surface Water Extent (DSWE) Downloader

This script downloads DSWE products from USGS using the M2M API.
It specifically targets Landsat Collection 2 Level-3 DSWE products.

Example DSWE URL format:
https://landsatlook.usgs.gov/level-3/collection02/DSWE/2024/CU/002/009/LC08_CU_002009_20241108_20241117_02_DSWE/
"""

import requests
import json
import os
import time
import csv
from datetime import datetime
from pathlib import Path
import sys
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

class DSWEDownloader:
    def __init__(self, username, token):
        self.username = username
        self.token = token
        self.base_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
        self.api_key = None
        # Known DSWE dataset names (from API)
        # Note: tile-based datasets may not filter by date properly
        self.dswe_dataset_names = [
            "landsat_dswe_tile_c2",           # Collection 2 tiles (newest data)
            "landsat_dswe_tile_files_c2",     # Collection 2 tile files
            "landsat_c1_l3_dswe_sample",      # Collection 1 samples
        ]
    
    def authenticate(self):
        """Authenticate with USGS M2M API using token"""
        url = f"{self.base_url}login-token"
        payload = {
            "username": self.username,
            "token": self.token
        }
        
        try:
            print(f"Authenticating with username: {self.username}")
            response = requests.post(url, json=payload)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response: {e}")
                print(f"Full response: {response.text}")
                return False
            
            if response.status_code != 200 or ("errorCode" in data and data["errorCode"]):
                print(f"Authentication failed: {data.get('errorMessage', 'Unknown error')}")
                return False
            
            self.api_key = data["data"]
            print("Authentication successful")
            return True
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False
    
    def logout(self):
        """Logout from USGS M2M API"""
        if not self.api_key:
            return True
        
        url = f"{self.base_url}logout"
        headers = {"X-Auth-Token": self.api_key}
        
        try:
            response = requests.post(url, headers=headers)
            self.api_key = None
            print("Logged out successfully")
            return True
        except Exception as e:
            print(f"Logout error: {str(e)}")
            return False
    
    def find_dswe_dataset(self):
        """Find the correct DSWE dataset"""
        if not self.api_key:
            print("Not authenticated. Please call authenticate() first.")
            return None
        
        print("\nTrying known DSWE datasets...")
        # First try direct access to known DSWE datasets
        for dataset_name in self.dswe_dataset_names:
            try:
                url = f"{self.base_url}dataset"
                headers = {"X-Auth-Token": self.api_key}
                
                payload = {
                    "datasetName": dataset_name
                }
                
                response = requests.post(url, json=payload, headers=headers)
                data = response.json()
                
                if response.status_code == 200 and "data" in data:
                    print(f"✓ Successfully accessed DSWE dataset: {dataset_name}")
                    return dataset_name
                else:
                    print(f"✗ Dataset {dataset_name} not found or not accessible")
            except Exception as e:
                print(f"✗ Error accessing {dataset_name}: {str(e)}")
                continue
        
        # If direct access fails, search for DSWE datasets
        print("\nSearching for DSWE datasets...")
        try:
            url = f"{self.base_url}dataset-search"
            headers = {"X-Auth-Token": self.api_key}
            
            # Try multiple search patterns
            search_patterns = [
                "landsat dswe",
                "dynamic surface water",
                "landsat collection 2 level-3"
            ]
            
            for pattern in search_patterns:
                payload = {
                    "datasetName": pattern
                }
                
                response = requests.post(url, json=payload, headers=headers)
                data = response.json()
                
                if response.status_code == 200 and "data" in data:
                    datasets = data["data"]
                    
                    if datasets and len(datasets) > 0:
                        print(f"\nFound {len(datasets)} datasets matching '{pattern}':")
                        
                        # Filter for DSWE datasets
                        dswe_datasets = []
                        for i, dataset in enumerate(datasets[:20]):  # Show first 20
                            # Handle different API response formats
                            dataset_name = dataset.get('collectionName') or dataset.get('datasetAlias') or 'Unknown'
                            dataset_id = dataset.get('datasetAlias') or dataset.get('collectionName') or 'Unknown'
                            
                            if 'dswe' in dataset_name.lower() or 'dswe' in dataset_id.lower():
                                print(f"  {len(dswe_datasets)+1}. {dataset_id}")
                                dswe_datasets.append(dataset_id)
                        
                        if dswe_datasets:
                            print(f"\nUsing first DSWE dataset: {dswe_datasets[0]}")
                            return dswe_datasets[0]
            
        except Exception as e:
            print(f"Dataset search error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n⚠ No DSWE dataset found. Available datasets may require different search terms.")
        print("Try checking https://m2m.cr.usgs.gov/ for current dataset names.")
        return None
    
    def list_all_datasets(self, max_results=100, show_all=False):
        """List all available datasets in the API"""
        if not self.api_key:
            print("Not authenticated. Please call authenticate() first.")
            return None
        
        print("\nListing available datasets...")
        try:
            url = f"{self.base_url}dataset-search"
            headers = {"X-Auth-Token": self.api_key}
            
            payload = {
                "datasetName": ""
            }
            
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()
            
            if response.status_code == 200 and "data" in data:
                datasets = data["data"]
                print(f"\nFound {len(datasets)} total datasets")
                
                # Show DSWE/water-related datasets first
                print("\nDSWE and water-related datasets:")
                dswe_found = False
                water_related = []
                
                for dataset in datasets:
                    dataset_name = dataset.get('collectionName') or dataset.get('datasetAlias') or 'Unknown'
                    dataset_id = dataset.get('datasetAlias') or dataset.get('collectionName') or 'Unknown'
                    abstract = dataset.get('abstract', '').lower()
                    
                    # Check for DSWE, water, or level-3 in name or abstract
                    if ('dswe' in dataset_name.lower() or 
                        'dswe' in dataset_id.lower() or 
                        'dynamic surface water' in dataset_name.lower() or
                        'dynamic surface water' in abstract or
                        ('water' in dataset_name.lower() and 'landsat' in dataset_name.lower()) or
                        ('level-3' in dataset_name.lower() and 'landsat' in dataset_name.lower()) or
                        ('level 3' in dataset_name.lower() and 'landsat' in dataset_name.lower())):
                        water_related.append((dataset_id, dataset_name))
                        dswe_found = True
                
                if water_related:
                    for dataset_id, dataset_name in water_related[:max_results]:
                        print(f"  - {dataset_id}")
                        if show_all:
                            print(f"    Name: {dataset_name}")
                else:
                    print("  (No DSWE/water datasets found)")
                    print("\n  Showing first 20 Landsat datasets for reference:")
                    landsat_count = 0
                    for dataset in datasets:
                        dataset_name = dataset.get('collectionName') or dataset.get('datasetAlias') or 'Unknown'
                        dataset_id = dataset.get('datasetAlias') or dataset.get('collectionName') or 'Unknown'
                        
                        if 'landsat' in dataset_name.lower() or 'landsat' in dataset_id.lower():
                            print(f"    - {dataset_id}")
                            landsat_count += 1
                            if landsat_count >= 20:
                                break
                
                return datasets
            else:
                print(f"Failed to list datasets: {data.get('errorMessage', 'Unknown error')}")
                return None
            
        except Exception as e:
            print(f"Error listing datasets: {str(e)}")
            import traceback
            traceback.print_exc()
        return None
    
    def search_dswe_scenes(self, dataset, bbox, start_date, end_date):
        """Search for DSWE scenes in a specific area and time range"""
        if not self.api_key:
            print("Not authenticated. Please call authenticate() first.")
            return None
        
        url = f"{self.base_url}scene-search"
        headers = {"X-Auth-Token": self.api_key}
        
        # Create spatial filter
        min_lon, min_lat, max_lon, max_lat = bbox
        
        payload = {
            "datasetName": dataset,
            "maxResults": 100,  # Increase if needed
            "startingNumber": 1,
            "sceneFilter": {
                "spatialFilter": {
                    "filterType": "mbr",
                    "lowerLeft": {
                        "latitude": min_lat,
                        "longitude": min_lon
                    },
                    "upperRight": {
                        "latitude": max_lat,
                        "longitude": max_lon
                    }
                },
                "temporalFilter": {
                    "startDate": start_date,
                    "endDate": end_date
                }
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()
            
            if response.status_code != 200 or "errorCode" in data and data["errorCode"]:
                print(f"Scene search failed: {data.get('errorMessage', 'Unknown error')}")
                return None
            
            results = data["data"]["results"]
            print(f"Found {len(results)} DSWE scenes")
            
            # Print first few results
            for i, scene in enumerate(results[:3]):
                print(f"  {i+1}. ID: {scene.get('entityId', 'Unknown')}")
                
                # Try to extract acquisition date and path/row
                metadata = scene.get('metadata', [])
                acquisition_date = "Unknown"
                path_row = "Unknown/Unknown"
                
                for field in metadata:
                    if field.get('fieldName') == 'Acquisition Date':
                        acquisition_date = field.get('value', 'Unknown')
                    elif field.get('fieldName') == 'WRS Path':
                        path = field.get('value', 'Unknown')
                    elif field.get('fieldName') == 'WRS Row':
                        row = field.get('value', 'Unknown')
                        path_row = f"{path}/{row}"
                
                print(f"     Acquisition Date: {acquisition_date}")
                print(f"     Path/Row: {path_row}")
            
            return results
        except Exception as e:
            print(f"Scene search error: {str(e)}")
            return None
    
    def get_dswe_metadata(self, scene_id, dataset):
        """Get detailed metadata for a DSWE scene"""
        if not self.api_key:
            print("Not authenticated. Please call authenticate() first.")
            return None
        
        url = f"{self.base_url}scene-metadata"
        headers = {"X-Auth-Token": self.api_key}
        
        payload = {
            "datasetName": dataset,
            "entityId": scene_id
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()
            
            if response.status_code != 200 or "errorCode" in data and data["errorCode"]:
                print(f"Metadata request failed: {data.get('errorMessage', 'Unknown error')}")
                return None
            
            return data["data"]
        except Exception as e:
            print(f"Metadata error: {str(e)}")
            return None
    
    def construct_dswe_url(self, scene_metadata):
        """Construct direct DSWE download URL based on metadata"""
        # Extract necessary components from metadata
        product_id = None
        path = None
        row = None
        acquisition_date = None
        processing_date = None
        
        for item in scene_metadata:
            if item.get('fieldName') == 'Landsat Product Identifier L3':
                product_id = item.get('value')
            elif item.get('fieldName') == 'WRS Path':
                path = item.get('value')
            elif item.get('fieldName') == 'WRS Row':
                row = item.get('value')
            elif item.get('fieldName') == 'Date L1 Generated':
                processing_date = item.get('value')
            elif item.get('fieldName') == 'Acquisition Date':
                acquisition_date = item.get('value')
        
        if not product_id:
            # Try to find any product ID that ends with DSWE
            for item in scene_metadata:
                if item.get('value') and '_DSWE' in item.get('value'):
                    product_id = item.get('value')
                    break
        
        if not product_id:
            print("Could not find DSWE product ID in metadata")
            return None
        
        # Parse product ID to extract components if available
        # Example: LC08_CU_002009_20241108_20241117_02_DSWE
        if not path or not row:
            match = re.search(r'_CU_(\d{3})(\d{3})_', product_id)
            if match:
                path = match.group(1)
                row = match.group(2)
        
        if not acquisition_date or not processing_date:
            match = re.search(r'_(\d{8})_(\d{8})_', product_id)
            if match:
                acquisition_date = match.group(1)
                processing_date = match.group(2)
        
        # Format dates if needed
        if acquisition_date and len(acquisition_date) == 10:  # YYYY-MM-DD format
            acquisition_date = acquisition_date.replace('-', '')
        if processing_date and len(processing_date) == 10:  # YYYY-MM-DD format
            processing_date = processing_date.replace('-', '')
        
        # Construct URL based on example format
        # landsatlook.usgs.gov/level-3/collection02/DSWE/2024/CU/002/009/LC08_CU_002009_20241108_20241117_02_DSWE/
        if path and row and acquisition_date and processing_date:
            year = acquisition_date[:4]
            base_url = f"https://landsatlook.usgs.gov/level-3/collection02/DSWE/{year}/CU/{path}/{row}/{product_id}"
            return base_url
        else:
            print("Missing components for DSWE URL construction")
            print(f"Product ID: {product_id}")
            print(f"Path: {path}, Row: {row}")
            print(f"Acquisition Date: {acquisition_date}, Processing Date: {processing_date}")
            return None
    
    def get_download_options(self, scene_id, dataset):
        """Get available download options for a scene"""
        if not self.api_key:
            print("Not authenticated. Please call authenticate() first.")
            return None
        
        url = f"{self.base_url}download-options"
        headers = {"X-Auth-Token": self.api_key}
        
        payload = {
            "datasetName": dataset,
            "entityIds": [scene_id]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()
            
            if response.status_code != 200 or "errorCode" in data and data["errorCode"]:
                print(f"Download options request failed: {data.get('errorMessage', 'Unknown error')}")
                return None
            
            return data["data"]
        except Exception as e:
            print(f"Error getting download options: {str(e)}")
            return None
    
    def request_download(self, scene_id, dataset, product_id):
        """Request download URL for a scene"""
        if not self.api_key:
            print("Not authenticated. Please call authenticate() first.")
            return None
        
        url = f"{self.base_url}download-request"
        headers = {"X-Auth-Token": self.api_key}
        
        payload = {
            "downloads": [
                {
                    "entityId": scene_id,
                    "productId": product_id
                }
            ],
            "datasetName": dataset
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()
            
            if response.status_code != 200 or "errorCode" in data and data["errorCode"]:
                print(f"Download request failed: {data.get('errorMessage', 'Unknown error')}")
                return None
            
            return data["data"]
        except Exception as e:
            print(f"Error requesting download: {str(e)}")
            return None

    def request_download_batch(self, dataset, downloads):
        """Request download URLs for multiple scenes at once."""
        if not self.api_key:
            print("Not authenticated. Please call authenticate() first.")
            return None

        if not downloads:
            return None

        url = f"{self.base_url}download-request"
        headers = {"X-Auth-Token": self.api_key}

        payload = {
            "downloads": downloads,
            "datasetName": dataset
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            data = response.json()

            if response.status_code != 200 or ("errorCode" in data and data["errorCode"]):
                print(f"Batch download request failed: {data.get('errorMessage', 'Unknown error')}")
                return None

            return data["data"]
        except Exception as e:
            print(f"Error requesting batch download: {str(e)}")
            return None
    
    def download_dswe_files(self, scene_id, dataset, output_dir):
        """Download DSWE files using M2M API (legacy per-scene method)."""
        prepared, _, _ = self.prepare_downloads(
            scene_ids=[scene_id],
            dataset=dataset,
            output_root=os.path.dirname(output_dir) or output_dir,
            skip_existing=False,
            display_lookup={scene_id: os.path.basename(output_dir) or scene_id},
        )

        results = self.process_downloads(
            prepared,
            dataset=dataset,
            max_workers=1,
        )

        downloaded_files = [
            res.get("output_file")
            for res in results
            if res.get("status") == "downloaded" and res.get("output_file")
        ]
        return downloaded_files

    def scene_already_downloaded(self, scene_dir):
        if not os.path.isdir(scene_dir):
            return False
        for name in os.listdir(scene_dir):
            lname = name.lower()
            if (
                lname.endswith(('.tif', '.json', '.xml', '.tar', '.zip'))
                or 'gen-bundle' in lname
            ):
                return True
        return False

    def prepare_downloads(self, scene_ids, dataset, output_root, skip_existing=True, display_lookup=None):
        prepared = {}
        skipped_existing = []
        failures = []

        for scene_id in scene_ids:
            if not scene_id:
                continue

            dir_name = display_lookup.get(scene_id) if display_lookup else None
            if not dir_name:
                dir_name = scene_id

            scene_dir = os.path.join(output_root, dir_name)

            if skip_existing and self.scene_already_downloaded(scene_dir):
                print(f"- Skipping {scene_id}; existing files found in {scene_dir}")
                skipped_existing.append(scene_id)
                continue

            os.makedirs(scene_dir, exist_ok=True)

            download_options = self.get_download_options(scene_id, dataset)
            if not download_options:
                failures.append((scene_id, "No download options available"))
                continue

            best_option = None
            for option in download_options:
                if option.get('available'):
                    product_name = option.get('productName', '').lower()
                    if 'dswe' in product_name or 'level-3' in product_name:
                        best_option = option
                        break

            if not best_option:
                for option in download_options:
                    if option.get('available'):
                        best_option = option
                        break

            if not best_option:
                failures.append((scene_id, "No available download options"))
                continue

            prepared[scene_id] = {
                "entityId": scene_id,
                "productId": best_option.get('id'),
                "productName": best_option.get('productName', 'Unknown'),
                "output_dir": scene_dir,
            }

        return prepared, skipped_existing, failures

    def process_downloads(self, prepared_tasks, dataset, max_workers=4):
        if not prepared_tasks:
            return []

        downloads_payload = [
            {"entityId": info["entityId"], "productId": info["productId"]}
            for info in prepared_tasks.values()
        ]

        print(f"Requesting download URLs for {len(downloads_payload)} scenes in batch...")
        download_data = self.request_download_batch(dataset=dataset, downloads=downloads_payload)

        if not download_data:
            print("Batch download request failed; no downloads will be processed.")
            return []

        url_tasks = []
        for download in download_data.get('availableDownloads', []):
            download_url = download.get('url')
            if not download_url:
                continue

            entity_id = download.get('entityId') or download.get('sceneId')
            if not entity_id or entity_id not in prepared_tasks:
                continue

            filename = download_url.split('/')[-1]
            if '?' in filename:
                filename = filename.split('?')[0]
            if not filename:
                filename = f"{entity_id}.tar"

            output_dir = prepared_tasks[entity_id]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, filename)

            url_tasks.append({
                "entity_id": entity_id,
                "url": download_url,
                "output_file": output_file,
                "product_name": prepared_tasks[entity_id]["productName"],
            })

        if download_data.get('preparingDownloads'):
            print("\nSome downloads are being prepared by USGS (not yet available):")
            for download in download_data['preparingDownloads']:
                print(f"  - {download.get('downloadId')}: check status later")

        if not url_tasks:
            print("No immediate download URLs were returned by the API.")
            return []

        print(f"\nStarting downloads for {len(url_tasks)} scenes with up to {max_workers} workers...")

        def download_task(task):
            url = task["url"]
            output_file = task["output_file"]
            entity_id = task["entity_id"]

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return {
                    "entity_id": entity_id,
                    "output_file": output_file,
                    "status": "skipped",
                    "message": "File already exists",
                }

            try:
                response = requests.get(url, stream=True, timeout=300)
                if response.status_code != 200:
                    return {
                        "entity_id": entity_id,
                        "output_file": output_file,
                        "status": "error",
                        "message": f"HTTP {response.status_code}",
                    }

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                if total_size and downloaded < total_size:
                    return {
                        "entity_id": entity_id,
                        "output_file": output_file,
                        "status": "error",
                        "message": f"Incomplete download ({downloaded}/{total_size})",
                    }

                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                return {
                    "entity_id": entity_id,
                    "output_file": output_file,
                    "status": "downloaded",
                    "message": f"{size_mb:.2f} MB",
                }
            except Exception as exc:
                return {
                    "entity_id": entity_id,
                    "output_file": output_file,
                    "status": "error",
                    "message": str(exc),
                }

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(download_task, task): task for task in url_tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "entity_id": task['entity_id'],
                        "output_file": task['output_file'],
                        "status": "error",
                        "message": f"Unhandled exception: {exc}",
                    }

                status = result.get("status")
                message = result.get("message", "")
                entity_id = result.get("entity_id")

                if status == "downloaded":
                    print(f"✓ Downloaded {entity_id} -> {os.path.basename(result['output_file'])} ({message})")
                elif status == "skipped":
                    print(f"- Skipped {entity_id}: {message}")
                else:
                    print(f"✗ {entity_id}: {message}")

                results.append(result)

        return results
    
    def extract_bundle(self, bundle_file):
        """Extract tar bundle file if it's a compressed archive"""
        if not os.path.exists(bundle_file):
            print(f"Bundle file not found: {bundle_file}")
            return []
        
        # Check if it's a tar file
        try:
            output_dir = os.path.dirname(bundle_file)
            extracted_files = []
            
            # Try to open as tar file
            with tarfile.open(bundle_file, 'r') as tar:
                print(f"Extracting {os.path.basename(bundle_file)}...")
                tar.extractall(path=output_dir)
                
                # Get list of extracted files
                for member in tar.getmembers():
                    extracted_path = os.path.join(output_dir, member.name)
                    if os.path.isfile(extracted_path):
                        extracted_files.append(extracted_path)
                        print(f"  Extracted: {member.name}")
            
            # Remove the original bundle file after successful extraction
            os.remove(bundle_file)
            print(f"✓ Extracted {len(extracted_files)} files from bundle")
            
            return extracted_files
        except (tarfile.TarError, OSError) as e:
            print(f"Could not extract bundle (may not be a tar file): {str(e)}")
            return []
    
    def download_dswe_files_direct(self, base_url, output_dir):
        """Download DSWE files from direct URL (fallback method)"""
        os.makedirs(output_dir, exist_ok=True)
        
        file_extensions = [
            ".tif",
            ".json",
            ".xml",
            ".jpg",
            "_thumb_large.jpg",
            "_thumb_small.jpg"
        ]
        
        product_id = base_url.split('/')[-1]
        downloaded_files = []
        
        for ext in file_extensions:
            file_url = f"{base_url}/{product_id}{ext}"
            output_file = os.path.join(output_dir, f"{product_id}{ext}")
            
            try:
                response = requests.get(file_url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    file_size = os.path.getsize(output_file) / (1024 * 1024)
                    print(f"✓ Downloaded {product_id}{ext} ({file_size:.2f} MB)")
                    downloaded_files.append(output_file)
            except Exception as e:
                pass  # Silently skip missing files
        
        return downloaded_files
    
    def export_scene_list(self, scenes, output_file):
        """Export scene list to CSV file"""
        if not scenes:
            print("No scenes to export")
            return
        
        try:
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = ['entityId', 'displayId', 'acquisitionDate', 'path', 'row']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for scene in scenes:
                    # Extract metadata
                    display_id = scene.get('displayId', 'Unknown')
                    acquisition_date = 'Unknown'
                    path = 'Unknown'
                    row = 'Unknown'
                    
                    for field in scene.get('metadata', []):
                        if field.get('fieldName') == 'Acquisition Date':
                            acquisition_date = field.get('value', 'Unknown')
                        elif field.get('fieldName') == 'WRS Path':
                            path = field.get('value', 'Unknown')
                        elif field.get('fieldName') == 'WRS Row':
                            row = field.get('value', 'Unknown')
                    
                    writer.writerow({
                        'entityId': scene.get('entityId', 'Unknown'),
                        'displayId': display_id,
                        'acquisitionDate': acquisition_date,
                        'path': path,
                        'row': row
                    })
                
                print(f"Exported {len(scenes)} scenes to {output_file}")
        except Exception as e:
            print(f"Error exporting scene list: {str(e)}")

def read_credentials():
    """Read USGS credentials from file"""
    home_dir = str(Path.home())
    cred_file = os.path.join(home_dir, '.earthexplorer')
    
    if not os.path.exists(cred_file):
        print(f"Credentials file not found: {cred_file}")
        return None, None
    
    try:
        with open(cred_file, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 2:
            print("Credentials file should contain username on first line and token on second line")
            return None, None
        
        username = lines[0].strip()
        token = lines[1].strip()
        
        print(f"Read username: {username}")
        token_preview = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        print(f"Token length: {len(token)}, preview: {token_preview}")
        
        return username, token
    except Exception as e:
        print(f"Error reading credentials file: {str(e)}")
        return None, None

def main():
    """Main function to download DSWE data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download USGS DSWE data')
    parser.add_argument('--list-datasets', action='store_true', 
                        help='List all available datasets')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to an EarthExplorer CSV export to drive downloads')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset name to use')
    parser.add_argument('--bbox', type=float, nargs=4, 
                        default=[-122.6, 37.1, -121.8, 38.0],
                        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                        help='Bounding box (default: Bay Area)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2020-12-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-downloads', type=int, default=5,
                        help='Maximum number of scenes to download')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of scenes to request per M2M batch (default: 10)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of concurrent download workers (default: 4)')
    parser.add_argument('--force', action='store_true',
                        help='Redownload scenes even if files already exist')
    parser.add_argument('--output-dir', type=str, default='usgs_dswe_data',
                        help='Output directory for downloaded files')
    
    args = parser.parse_args()
    
    # Read credentials from ~/.earthexplorer file
    username, token = read_credentials()
    
    if not username or not token:
        print("Failed to read credentials. Exiting.")
        return
    
    # Initialize downloader
    downloader = DSWEDownloader(username, token)
    
    try:
        # Authenticate
        if not downloader.authenticate():
            print("Authentication failed. Exiting.")
            return
        
        # List datasets if requested
        if args.list_datasets:
            downloader.list_all_datasets()
            return
        
        # Find DSWE dataset
        print("\nFinding DSWE dataset...")
        if args.dataset:
            dswe_dataset = args.dataset
            print(f"Using specified dataset: {dswe_dataset}")
        else:
            dswe_dataset = downloader.find_dswe_dataset()
        
        if not dswe_dataset:
            print("\n⚠ Could not find DSWE dataset.")
            print("Try running with --list-datasets to see available datasets.")
            return
        
        print(f"\n✓ Using DSWE dataset: {dswe_dataset}")
        
        # Determine scenes to download
        scenes = []

        if args.csv:
            print(f"\nLoading scenes from CSV file: {args.csv}")
            try:
                # EarthExplorer CSV exports may use Windows-1252 encoding when
                # degree symbols are present in the coordinate fields. Try UTF-8
                # first, and fall back to a more permissive codec if necessary.
                encodings_to_try = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']

                csv_read_success = False
                last_error = None

                for enc in encodings_to_try:
                    try:
                        with open(args.csv, 'r', newline='', encoding=enc) as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                entity_id = row.get('Entity ID') or row.get('DSWE Tile Identifier')
                                display_id = row.get('Display ID') or entity_id

                                if entity_id:
                                    scenes.append({
                                        'entityId': entity_id,
                                        'displayId': display_id
                                    })

                        csv_read_success = True
                        break
                    except UnicodeDecodeError as decode_err:
                        scenes.clear()
                        last_error = decode_err
                        continue

                if not csv_read_success:
                    raise UnicodeDecodeError(
                        "CSV", b"", 0, 0,
                        f"Unable to decode CSV file with tried encodings: {encodings_to_try}. Last error: {last_error}"
                    )

                if not scenes:
                    print("\n⚠ No valid Entity IDs found in the CSV file. Nothing to download.")
                    return

                print(f"  Found {len(scenes)} scenes listed in CSV")
            except FileNotFoundError:
                print(f"\n✗ CSV file not found: {args.csv}")
                return
            except Exception as exc:
                print(f"\n✗ Error reading CSV file: {exc}")
                return
        else:
            # Search for DSWE scenes via API
            print(f"\nSearching for DSWE scenes via API...")
            print(f"  Area: {args.bbox}")
            print(f"  Date range: {args.start_date} to {args.end_date}")

            scenes = downloader.search_dswe_scenes(
                dswe_dataset,
                tuple(args.bbox),
                args.start_date,
                args.end_date
            )

            if not scenes or len(scenes) == 0:
                print("\n⚠ No DSWE scenes found for the specified criteria.")
                print("Try adjusting the bounding box or date range.")
                return

            print(f"\n✓ Found {len(scenes)} DSWE scenes")

            # Export scene list to CSV for reference
            csv_file = "bay_area_dswe_scenes_2020.csv"
            downloader.export_scene_list(scenes, csv_file)
        
        # Prepare for batch downloads
        selected_scenes = scenes[:args.max_downloads]
        scene_ids = []
        display_lookup = {}
        missing_ids = 0
        for scene in selected_scenes:
            entity_id = scene.get('entityId')
            if not entity_id:
                missing_ids += 1
                continue
            scene_ids.append(entity_id)
            display_lookup[entity_id] = scene.get('displayId', entity_id)

        if missing_ids:
            print(f"\n⚠ {missing_ids} scene entries were missing entity IDs and will be skipped.")

        if not scene_ids:
            print("\nNo valid scenes to download after filtering.")
            return

        os.makedirs(args.output_dir, exist_ok=True)
        total_results = []
        total_skipped_existing = []
        total_failures = []

        batch_size = max(1, args.batch_size)
        total_batches = (len(scene_ids) + batch_size - 1) // batch_size

        print(f"\nDownloading up to {len(scene_ids)} scenes in {total_batches} batch(es)...")
        print("=" * 60)

        for batch_index in range(0, len(scene_ids), batch_size):
            batch_ids = scene_ids[batch_index:batch_index + batch_size]
            batch_number = batch_index // batch_size + 1
            print(f"\nBatch {batch_number}/{total_batches}: {len(batch_ids)} scene(s)")
            print("-" * 60)

            prepared, skipped_existing, failures = downloader.prepare_downloads(
                scene_ids=batch_ids,
                dataset=dswe_dataset,
                output_root=args.output_dir,
                skip_existing=not args.force,
                display_lookup=display_lookup,
            )

            total_skipped_existing.extend(skipped_existing)
            total_failures.extend(failures)

            if not prepared:
                if skipped_existing:
                    print("All scenes in this batch already exist locally.")
                else:
                    print("No scenes prepared for download in this batch.")
                continue
            
            batch_results = downloader.process_downloads(
                prepared_tasks=prepared,
                dataset=dswe_dataset,
                max_workers=max(1, args.workers),
            )

            for result in batch_results:
                status = result.get('status')
                output_file = result.get('output_file')
                if status == "downloaded" and output_file:
                    extracted = downloader.extract_bundle(output_file)
                    if extracted:
                        print(f"  Extracted {len(extracted)} file(s)")

            total_results.extend(batch_results)

            # Respect API etiquette between batches
            time.sleep(1)

        downloaded_count = sum(1 for r in total_results if r.get('status') == 'downloaded')
        skipped_count = len(total_skipped_existing) + sum(1 for r in total_results if r.get('status') == 'skipped')
        error_count = len(total_failures) + sum(1 for r in total_results if r.get('status') == 'error')

        print("\n" + "=" * 60)
        print(f"Download summary:")
        print(f"  Downloaded: {downloaded_count}")
        print(f"  Skipped (existing): {skipped_count}")
        print(f"  Errors: {error_count}")

        if total_failures:
            print("\nScenes that could not be prepared:")
            for scene_id, reason in total_failures:
                print(f"  - {scene_id}: {reason}")

        if total_skipped_existing and args.force:
            print("\nNote: --force was specified but some scenes were still skipped due to existing files.")

        print(f"\nFiles saved to: {args.output_dir}")
    
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Always logout
        downloader.logout()

if __name__ == "__main__":
    main() 