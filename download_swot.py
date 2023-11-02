'''
Download SWOT data from podaac

Each dataset has it’s own unique collection ID. For the 
SWOT_SIMULATED_NA_CONTINENT_L2_HR_RIVERSP_V1 dataset, we find the collection ID
here: https://podaac.jpl.nasa.gov/dataset/SWOT_SIMULATED_NA_CONTINENT_L2_HR_RIVERSP_V1


'''

import requests,json,glob,os,zipfile 
import geopandas as gpd
from pathlib import Path
import pandas as pd
from urllib.request import urlretrieve
from json import dumps

import earthaccess
from earthaccess import Auth, DataCollections, DataGranules, Store

auth = earthaccess.login(strategy="interactive", persist=True)

#earthaccess data search
results = earthaccess.search_data(concept_id="C2263384307-POCLOUD", bounding_box=(-124.848974,24.396308,-66.885444,49.384358))


#add desired links to a list
#if the link has "Reach" instead of "Node" in the name, we want to download it for the swath use case
downloads = []
for g in results:
    for l in earthaccess.results.DataGranule.data_links(g):
        if 'https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/' in l:
            if 'Reach' in l:
                downloads.append(l)
            
print(len(downloads))

#Create folder to house downloaded data 
folder = Path("SWOT_sample_files")
#newpath = r'SWOT_sample_files' 
if not os.path.exists(folder):
    os.makedirs(folder)
    
#download data
earthaccess.download(downloads, "./SWOT_sample_files")

for item in os.listdir(folder): # loop through items in dir
    if item.endswith(".zip"): # check for ".zip" extension
        zip_ref = zipfile.ZipFile(f"{folder}/{item}") # create zipfile object
        zip_ref.extractall(folder) # extract file to dir
        zip_ref.close() # close file
        
os.listdir(folder)
