#!/usr/bin/env python3
import os,shutil
from swampy import setup_swampy

swampyPath = setup_swampy.__file__
swampyPath = os.path.dirname(swampyPath)

# If you want to ensure the output always ends with a '/'
paramPath = os.path.join(swampyPath, 'docs', 'params_template.yaml')

print('Copying from ' + paramPath)

shutil.copy(paramPath, './params.yaml')

# Define the path to your file
file_path = os.path.join(swampyPath, 'docs', 'swampy.txt')

# Open the file and print its contents
with open(file_path, 'r') as file:
    contents = file.read()
    print(contents)

print('\n')
print('Copying the params.yaml template.')
print('Next, edit the params.yaml file in this directory. Then you can download data with download_dsw.py')
print('\n')
