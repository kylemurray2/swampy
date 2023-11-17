# SWAMPy  
Surface Water Anyalysis Modules in Python   
version 0.1    

** Alpha version/under development/experimental **  
** please contribute if you are interested in helping develop these tools **

Experimental tools designed to work with JPL OPERA Dynamic Surface Water eXtent (DSWx) products
https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite

SWOT tools will also be added.

Install dependencies:  

    conda env create -f swampy/docs/requirements.yaml  
> Add swampy to your PATH and PYTHONPATH environment variables 

Workflow:  

setup_swampy.py  
   > Copies the params.py template to current directory.  
    Modify this file with your required parameters  

download_dsw.py  
   > Search an area and time range.   
    Download and put all of the output tifs in a directory.  
    Move files to respective date directories  

stitch_dsw.py   
   > Stitches files together within each date if there is more than one file  
    
water_elevation.py  
   > For each date, estimate the elevation of the water based on the surface   
    water extent and where the edges intersect a DEM.  
    plots a time series of estimated surface elevations
> 
![image](https://github.com/kylemurray2/swampy/assets/8814635/f33bf673-b87d-4d4a-849f-89ae18db8261)
