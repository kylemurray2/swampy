#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:08:52 2023

extra functions for swampy

@author: km
"""

import numpy as np
from matplotlib import pyplot as plt
import os,re


def update_yaml_key(file_path, key, new_value):
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            # Try to match a YAML key-value pair line
            match = re.match(rf"({key}\s*:\s*)(\S+)", line)
            if match:
                # Replace the value while preserving the key and any surrounding whitespace
                line = f"{match.group(1)}{new_value}\n"
            f.write(line)