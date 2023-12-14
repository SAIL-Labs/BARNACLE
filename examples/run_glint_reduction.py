#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:13:55 2021

@author: mam
"""
from barnacle.science.glint_reduction import *

''' Inputs '''
datafolder = '20231213_simu/'
#    root = "C:/Users/marc-antoine/glint/"
root = "/mnt/96980F95980F72D3/"
# root = "//tintagel.physics.usyd.edu.au/snert/"
output_path = root+'GLINTprocessed/'+datafolder
spectral_calibration_path = output_path
geometric_calibration_path = output_path
# data_path = '//tintagel.physics.usyd.edu.au/snert/GLINTData/'+datafolder
# data_path = 'C:/Users/marc-antoine/glint//GLINTData/'+datafolder
data_path = '/mnt/96980F95980F72D3/GLINTData/'+datafolder

''' Settings '''
nb_img = (0, None)
debug = False
save = True
nb_files = (0, None)
bin_frames = False
nb_frames_to_bin = 50
spectral_binning = False
wl_bin_min, wl_bin_max = 1525, 1575  # In nm
bandwidth_binning = 50  # In nm
mode_flux = 'raw'
activate_estimate_spectrum = False
nb_files_spectrum = (5000, 10000)
wavelength_bounds = (1400, 1700)
suffix = 'datacube'
#    ron = 0
plot_name = datafolder.split('/')[-2]


wl_bin_bounds = (wl_bin_min, wl_bin_max)
spectral_calibration_files = (spectral_calibration_path +
                              '20200601_wl_to_px.npy',
                              spectral_calibration_path +
                              '20200601_px_to_wl.npy')
monitor_amplitude, monitor_null, monitor_photo, wl_scale = \
    reduce_data(data_path, plot_name, output_path, suffix, nb_files, nb_img,
                nb_frames_to_bin,
                geometric_calibration_path, spectral_calibration_files,
                save, bin_frames, debug, spectral_binning, wl_bin_bounds,
                bandwidth_binning,
                activate_estimate_spectrum,
                nb_files_spectrum, mode_flux, wavelength_bounds)
