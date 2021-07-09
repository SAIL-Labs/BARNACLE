# -*- coding: utf-8 -*-
"""
Example on how to use the library ``zeta_coeff_lib.py``.

It gives the shape of the channels.
"""
import numpy as np
from barnacle.calibration.zeta_coeff_lib import *

# Settings
nb_img = (None, None)
save = False
mode_flux = 'raw'
spectral_binning = False
wl_bin_min, wl_bin_max = 1525, 1575  # In nm
bandwidth_binning = 50  # In nm

# I/O
datafolder = 'data202105/20210503/zeta/'
# root = "/mnt/96980F95980F72D3/glint/"
root = "//tintagel.physics.usyd.edu.au/snert/"
output_path = root+'GLINTprocessed/'+datafolder
dark_path = output_path
spectral_calibration_path = output_path
geometric_calibration_path = output_path
# data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
data_path = '//tintagel.physics.usyd.edu.au/snert/GLINTData/'+datafolder

wl_to_px_coeff = np.load(
    spectral_calibration_path+'20210429_wl_to_px.npy')
px_to_wl_coeff = np.load(
    spectral_calibration_path+'20210429_px_to_wl.npy')

zeta_coeff = do_zeta_coeff(data_path, dark_path, output_path,
                           geometric_calibration_path,
                           wl_to_px_coeff, px_to_wl_coeff, mode_flux,
                           spectral_binning, wl_bin_min, wl_bin_max,
                           bandwidth_binning, nb_img,
                           save, plotting=True)
