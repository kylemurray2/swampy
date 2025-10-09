#!/usr/bin/env python

'''Downloader for the Global Surface Water Data of the Copernicus Programme:
https://global-surface-water.appspot.com/download
https://global-surface-water.appspot.com/map
Based on the original downloadWaterData.py'''


import sys
import signal
import os
import argparse
if sys.version_info.major == 2:
    # Python 2
    from urllib import urlretrieve
    from urllib2 import HTTPError
else:
    # Python 3
    from urllib.request import urlretrieve
    from urllib.error import HTTPError


KNOWN_DATASETS = 'occurrence', 'change', 'seasonality', 'recurrence', 'transitions', 'extent'
REVISIONS = '1_0', '1_1', '1_1_2019', '1_3_2020', '1_4_2021'
_GLOBALS = {}


def templates(revision):
    '''Configure URL and file templates'''

    v10, v11, v11_2019, v13_2020, v14_2021 = REVISIONS
    url_tmpl  = 'http://storage.googleapis.com/global-surface-water/downloads'
    file_tmpl = '{ds}_{lon}_{lat}'
    if revision == v10:
        padding   = 15
    elif revision == v11:
        url_tmpl  += '2'
        file_tmpl += '_v' + v11
        padding   = 20
    elif revision == v11_2019:
        url_tmpl  += '2019v2'
        file_tmpl += 'v' + v11_2019
        padding   = 24
    elif revision == v13_2020:
        url_tmpl  += '2020'
        file_tmpl += 'v' + v13_2020
        padding   = 24
    elif revision == v14_2021:
        url_tmpl  += '2021'
        file_tmpl += 'v' + v14_2021
        padding   = 24
    url_tmpl  += '/{ds}/{file}'
    file_tmpl += '.tif'
    return (url_tmpl, file_tmpl, padding)


def sigint_handler(signum, frame):
    '''Handler for interruption signal (Ctrl+C).'''

    print('\ninterrupted by user')
    part_file = _GLOBALS['part_file']
    if os.path.exists(part_file):
        os.remove(part_file)
    sys.exit(0)


def main():
    '''The main function.'''
    
    # Define parameters directly instead of parsing command line arguments
    params = {
        'datasets': ['occurrence', 'change', 'seasonality', 'recurrence', 'transitions', 'extent'],
        'directory': os.path.join('/d/surfaceWater', 'esa_surface_water_data'),  # Output directory
        'force': True,  # Rewrite existing files
    }

    # Specify the tiles and years you want to download
    tiles_to_download = {
        '1_0': ['130W_40N'],      # 1984-2015 data
        '1_1': ['130W_40N'],      # 2016-2017 data
        '1_1_2019': ['130W_40N'], # 2018-2019 data
        '1_3_2020': ['130W_40N'], # 2020 data
        '1_4_2021': ['130W_40N']  # 2021 data
    }

    # register sigint handler
    signal.signal(signal.SIGINT, sigint_handler)

    # check output dir
    params['directory'] = os.path.normpath(params['directory'])
    if not os.path.isdir(params['directory']):
        print('Creating destination directory "{}"'.format(params['directory']))
        os.makedirs(params['directory'])
    else:
        print('Using destination directory "{}"'.format(params['directory']))

    # Download for each specified year and tile
    for revision, tiles in tiles_to_download.items():
        print(f'\nProcessing revision {revision}')
        
        # configure templates
        url_tmpl, file_tmpl, filename_padding = templates(revision)
        print(f'URL template: {url_tmpl}')

        # downloading datasets
        skip = not params['force']
        for ds_name in params['datasets']:
            print(f'\nProcessing dataset: {ds_name}')
            ds_dir = os.path.join(params['directory'], f'{ds_name}_{revision}')
            if not os.path.isdir(ds_dir):
                os.makedirs(ds_dir)
            
            for tile in tiles:
                lon, lat = tile.split('_')
                filename = file_tmpl.format(ds=ds_name, lon=lon, lat=lat)
                filepath = os.path.join(ds_dir, filename)
                url = url_tmpl.format(ds=ds_name, file=filename)
                
                print(f'\nAttempting to download:')
                print(f'URL: {url}')
                print(f'To: {filepath}')
                
                if skip and os.path.exists(filepath):
                    print('File already exists, skipping')
                else:
                    try:
                        part_file = filepath + '.part'
                        _GLOBALS['part_file'] = part_file
                        urlretrieve(url, part_file)
                        os.rename(part_file, filepath)
                        print('Download successful')
                    except HTTPError as err:
                        print(f'HTTP Error: {err.code} - {err.reason}')
                        print(f'Failed URL: {url}')
                    except Exception as e:
                        print(f'Unexpected error: {str(e)}')
    
    print('\nProcess finished')


if __name__ == "__main__":
    main()