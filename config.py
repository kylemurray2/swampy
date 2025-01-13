#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:46:44 2023

@author: km
"""

import numpy as np
import os,sys
import yaml
import argparse



def load_yaml_to_namespace(yaml_file):
    # Load the YAML file into a dictionary
    with open(yaml_file, 'r') as yaml_in:
        yaml_dict = yaml.safe_load(yaml_in)

    # Create a namespace from the dictionary
    namespace = argparse.Namespace(**yaml_dict)
    
    return namespace

def getPS(directory='.'):
    # Load the params from the yaml file
    yaml_file = os.path.join(directory,'params.yaml')
    
    if os.path.isfile(yaml_file):
        print('Parsing yaml file and updating ps namespace...')
        params = load_yaml_to_namespace(yaml_file)
        
        # Load the ps namespace with error checking
        ps_path = os.path.join(directory, 'ps.npy')
        if os.path.isfile(ps_path):
            try:
                ps = np.load(ps_path, allow_pickle=True).item()
                # Verify ps is a namespace object
                if not isinstance(ps, argparse.Namespace):
                    print('Warning: ps.npy did not contain a valid namespace. Creating new one.')
                    ps = params
            except:
                print('Warning: Could not load ps.npy properly. Creating new namespace.')
                ps = params
        else:
            ps = params
        
        # Update ps with any changes to params
        for attr in dir(params):
            if not attr.startswith('_'):
                setattr(ps, attr, getattr(params, attr))
    
    else:
        print('no params.yaml file found')
        sys.exit(1)
            
    # Save the updated ps namespace 
    np.save(os.path.join(directory, 'ps.npy'), ps)
    
    return ps