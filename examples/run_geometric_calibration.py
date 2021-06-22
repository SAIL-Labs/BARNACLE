# -*- coding: utf-8 -*-
"""
Example on how to use the library ``geometric_calibration.py``.

It gives the shape of the channels.
"""
import os
import numpy as np
from barnacle.calibration.geometric_calibration import *

''' Settings '''
save = True
plotting = True

print("Getting the shape (position and width) of all tracks")
''' Inputs '''
datafolder = 'data202006/20200602/turbulence/'
#    root = "/mnt/96980F95980F72D3/glint/"
root = "//tintagel.physics.usyd.edu.au/snert/"
#    data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
data_path = '//tintagel.physics.usyd.edu.au/snert/GLINTData/'+datafolder
output_path = root+'GLINTprocessed/'+datafolder
data_list = [data_path +
             f for f in os.listdir(data_path) if 'dark' not in f][:1000]

spectral_calibration_path = output_path
wl_to_px_coeff = np.load(spectral_calibration_path+'20200601_wl_to_px.npy')
px_to_wl_coeff = np.load(spectral_calibration_path+'20200601_px_to_wl.npy')

geo_calib = do_geometric_calibration(data_list, output_path, px_to_wl_coeff,
                                     save, plotting)
