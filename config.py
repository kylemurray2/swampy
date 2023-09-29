#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:46:44 2023

@author: km
"""

import numpy as np
import os,sys,yaml,argparse



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
        # Load the ps namespace
        if os.path.isfile('./ps.npy'):
            ps = np.load('./ps.npy',allow_pickle=True).all()
        else:
            ps = params
        
        # Update ps with any changes to params
        for attr in dir(params):
            if not attr.startswith('_'):
                # print(attr)
                setattr(ps, attr, getattr(params, attr))
        
        
    else:
        print('no params.yaml file found')
        sys.exit(1)
            
    # Save the updated ps namespace 
    np.save(os.path.join(directory, 'ps.npy'),ps)
    
    return ps