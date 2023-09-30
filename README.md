# SWAMPy  
Surface Water Anyalysis Modules in Python   
version 0.1  

Install:  

    create conda environment using swampy/docs/requirements.yaml  
    conda env create -f requirements.yaml  
    Add swampy to your PATH and PYTHONPATH environment variables 

Workflow:  

setup_swampy.py  
   > Copies the params.py template to current directory.  
    Modify this file with your required parameters  

download_dsw.py  
    Search an area and time range.   
    Download and put all of the output tifs in a directory.  
    Move files to respective date directories  

stitch_dsw.py   
    Stitches files together within each date if there is more than one file  
    
water_elevation.py  
    For each date, estimate the elevation of the water based on the surface   
    water extent and where the edges intersect a DEM.  
    plots a time series of estimated surface elevations  