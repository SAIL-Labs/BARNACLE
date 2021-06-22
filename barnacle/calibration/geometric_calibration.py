#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script determines the position and width of the outputs per\
spectral channel, assuming a Gaussian profile.

This script requires the library :doc:`glint_classes` to work.

The inputs are **averaged dark** and **datacube with no fringe**.
To get them, either you can try to be out of the coherent envelop for all
baselines or having a large time-varying phase.
In the last case, the average frame of the datacube blurs the fringe.

The product is:
        * the coefficients of the Gaussian profile as an array of shape: \
            (spectral channels, output, coefficient).

The saved coefficients are in this order: the amplitude, the location,
the sigma and the offset of the Gaussian.
While only the location and the sigma are really useful for the DRS,
the others are kept for deep diagnotics.
The product is saved into numpy-format file (.npy).

This script is used in 3 steps.

First step: specify the following settings:
    * :save: *(boolean)*, ``True`` for saving products and monitoring data,\
            ``False`` otherwise
    * :monitoring: *(boolean)*, ``True`` for displaying the results of\
                    the model fitting and the residuals for both location and\
                        width for all outputs

Second step: specify the paths to load data and save outputs:
    * :datafolder: folder containing the datacube to use.
    * :root: path to **datafolder**.
    * :data_list: list of files in **datafolder** to open.
    * :output_path: path to the folder where the products are saved.
    * :spectral_calibration_path: path to the files of the spectral\
                                    calibration


Third step: call the function ``do_geometric_calibration``.

Example available in ``examples/run_geometric_calibration.py``.
"""
import numpy as np
import matplotlib.pyplot as plt
import barnacle.glint_classes as glint_classes
from scipy.optimize import curve_fit
from barnacle.glint_functions import gaussian_with_offset, \
    get_channel_positions

LABELS = ['P4', 'N3', 'P3', 'N2', 'AN4', 'N5', 'N4', 'AN5',
          'N6', 'AN1', 'AN6', 'N1', 'AN2', 'P2', 'AN3', 'P1']
NB_TRACKS = 16


def do_geometric_calibration(data_list, output_path, px_to_wl_coeff,
                             save, plotting):
    """
    Wrapper to get and save the geometric calibration for each channel at once.

    :param data_list: list of files of no-fringe data.
    :type data_list: list
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: if ``True``, save the outputs of the function.
    :type save: boolean
    :param plotting: Plot figures.
                    If ``save`` is True, the figures are saved in
                    ``output_path``.
    :raises IndexError: check the list of data is not empty.
    :return: Coefficients of the Gaussian profil of each spectral channel\
            for each channel
    :rtype: array

    """
    if len(data_list) == 0:
        raise IndexError('Data list is empty')

    ''' Remove dark from the frames and average them to increase SNR '''
    dark = np.load(output_path+'superdark.npy')
    dark_per_channel = np.load(output_path+'superdarkchannel.npy')

    ''' Define bounds of each track '''
    channel_pos, sep = get_channel_positions(NB_TRACKS)

    slices = average_frames(data_list, output_path, dark, dark_per_channel,
                            channel_pos, sep, save, plotting)

    geo_calib, slices_axes, residuals = \
        get_geometric_calibration(slices, dark, channel_pos, sep, output_path,
                                  save)

    if save:
        np.save(output_path+'pattern_coeff', geo_calib)

    if plotting:
        plot(slices, slices_axes, geo_calib, residuals, px_to_wl_coeff, dark,
             output_path, save)

    return geo_calib


def average_frames(data_list, output_path, dark, dark_per_channel, channel_pos,
                   sep, save, plotting):
    """
    Average the frames to get no-fringe data.

    It is assumed the phase diversity is high enough to blur the fringes on
    the averaged frame.

    :param data_list: list of files of no-fringe data.
    :type data_list: list
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param dark: average dark.
    :type dark: 2D-array
    :param dark_per_channel: average dark per channel.
    :type dark_per_channel: 3D-array
    :param channel_pos: positions of the channels.
    :type channel_pos: list or array
    :param sep: spatial gap between two successive channels, in pixel.
    :type sep: float
    :param save: if ``True``, save the outputs of the function.
    :type save: boolean
    :param plotting: Plot figures.
                    If ``save`` is True, the figures are saved in
                    ``output_path``.
    :type plotting: boolean
    :return: array containing the averaged slices individually.
    :rtype: 3D-array

    """
    super_img = np.zeros(dark.shape)
    super_nb_img = 0.

    spatial_axis = np.arange(dark.shape[0])
    slices = np.zeros_like(dark_per_channel)

    print('Averaging frames')
    for f in data_list[:]:
        img = glint_classes.Null(f)
        img.cosmeticsFrames(np.zeros(dark.shape))
        img.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
        super_img = super_img + img.data.sum(axis=0)
        slices = slices + np.sum(img.slices, axis=0)
        super_nb_img = super_nb_img + img.nbimg
        if data_list.index(f) % 1000 == 0:
            print(data_list.index(f))

    slices = slices / super_nb_img
    super_img = super_img / super_nb_img

    if plotting:
        plt.figure(0, figsize=(19.2, 10.8))
        plt.clf()
        plt.imshow(super_img-dark, interpolation='none')
        plt.colorbar()
        if save:
            plt.savefig(output_path+'geo_calib_avg_frame.png')

    return slices


def get_geometric_calibration(slices, dark, channel_pos, sep, output_path,
                              save, plotting):
    """
    Get the geometric calibration (i.e. the shapes of the channels).

    :param slices: array containing the averaged slices individually.
    :type slices: 3D-array
    :param dark: average dark.
    :type dark: 2D-array
    :param channel_pos: positions of the channels.
    :type channel_pos: list or array
    :param sep: spatial gap between two successive channels, in pixel.
    :type sep: float
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: if ``True``, save some outputs of the function.
    :type save: boolean
    :param plotting: Plot figures.
                    If ``save`` is True, the figures are saved in
                    ``output_path``.
    :type plotting: boolean
    :return: Coefficients of the Gaussian profil of each spectral channel\
            for each channel, axes of the channels for ``plot`` function,\
                residuals of the fit.
    :rtype: 3-tuple

    """
    # Fit a gaussian on every track and spectral channel
    # to get their positions, widths and amplitude
    print('Determine position and width of the outputs')
    spatial_axis = np.arange(dark.shape[0])

    img = glint_classes.Null(data=None, nbimg=(0, 1))
    img.cosmeticsFrames(np.zeros(dark.shape))
    img.getChannels(channel_pos, sep, spatial_axis)

    img.slices = slices
    slices_axes = img.slices_axes
    params = []
    cov = []
    residuals = []

    pattern = []
    for i in range(slices.shape[0]):  # Loop over columns of pixel
        for j in range(slices.shape[1]):  # Loop over tracks
            p_init2 = np.array([100, channel_pos[j], 1., 0])
            try:
                popt, pcov = curve_fit(
                    gaussian_with_offset, slices_axes[j], slices[i, j],
                    p0=p_init2)
                params.append(popt)
                cov.append(np.diag(pcov))
                residuals.append(
                    slices[i, j] - gaussian_with_offset(slices_axes[j], *popt))
            except RuntimeError:
                popt = np.zeros(p_init2.shape)
                pcov = np.zeros(p_init2.shape)
                params.append(popt)
                cov.append(pcov)
                residuals.append(np.zeros(slices_axes[j].shape))
                print("Error fit at spectral channel " +
                      str(i)+" of track "+str(j))
            reconstruction = gaussian_with_offset(slices_axes[j], *popt)
            pattern.append(reconstruction)
            if i == 40 or i == 60 or i == 85:
                if plotting:
                    plt.figure(1, figsize=(19.2, 10.8))
                    plt.subplot(4, 4, j+1)
                    plt.plot(slices_axes[j], slices[i, j], '.')
                    plt.plot(slices_axes[j], gaussian_with_offset(
                        slices_axes[j], *popt))
                    plt.grid()
                    plt.title(LABELS[j])
                    plt.tight_layout()
                    if save:
                        plt.savefig(output_path+'geo_calib_fitting_output.png')

    params = np.array(params).reshape((dark.shape[1], NB_TRACKS, -1))
    cov = np.array(cov).reshape((dark.shape[1], NB_TRACKS, -1))
    residuals = np.array(residuals).reshape((dark.shape[1], NB_TRACKS, -1))
    pattern = np.array(pattern).reshape((dark.shape[1], NB_TRACKS, -1))

    # convert negative widths into positive ones
    params[:, :, 2] = abs(params[:, :, 2])

    return params, slices_axes, residuals


def plot(slices, slices_axes, geo_calib, residuals, px_to_wl_coeff, dark,
         output_path, save):
    """
    Plot the data and the model fitting.

    :param slices: array containing the averaged slices individually.
    :type slices: 3D-array
    :param slices_axes: axes of the channels
    :type slices_axes: array
    :param geo_calib: fitted coefficients of the Gaussian profile of each\
        spectral channel for each channel.
    :type geo_calib: array
    :param residuals: residuals of the fit of a Gaussian profile on each\
        spectral channel for each channel.
    :type residuals: array
    :param px_to_wl_coeff: polynomial coefficients of the spectral calibration\
                            converting pixel positions into wavelength.
    :type px_to_wl_coeff: array
    :param dark: average dark.
    :type dark: 2D-array
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: if ``True``, save some outputs of the function.
    :type save: boolean
    :return: Nothing
    :rtype: Vacuum

    """
    spectral_axis = np.arange(dark.shape[1])
    plt.figure(2, figsize=(19.2, 10.8))
    plt.clf()
    for k in range(16):
        spectral_axis_in_nm = np.poly1d(px_to_wl_coeff[k])(spectral_axis)
        step = np.mean(np.diff(spectral_axis_in_nm))
        plt.subplot(4, 4, k+1)
        plt.plot(spectral_axis_in_nm[20:], geo_calib[20:, k, 1])
        plt.imshow(np.log10(slices[:, k, :].T), interpolation='none',
                   extent=[spectral_axis_in_nm[0]-step/2,
                           spectral_axis_in_nm[-1]+step/2,
                           slices_axes[k, -1]-0.5,
                           slices_axes[k, 0]+0.5], aspect='auto')
        plt.colorbar()
        plt.title(LABELS[k])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Vertical position')
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'geo_calib_avg_slices.png')

    plt.figure(3, figsize=(19.2, 10.8))
    plt.clf()
    for k in range(16):
        spectral_axis_in_nm = np.poly1d(px_to_wl_coeff[k])(spectral_axis)
        step = np.mean(np.diff(spectral_axis_in_nm))
        plt.subplot(4, 4, k+1)
        plt.imshow(slices[:, k, :].T, interpolation='none', aspect='auto')
        plt.colorbar()
        plt.title(LABELS[k])
        plt.xlabel('Wavelength (px)')
        plt.ylabel('Vertical position (px)')
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'geo_calib_avg_slices_no_unit.png')

    plt.figure(4, figsize=(19.2, 10.8))
    plt.clf()
    for k in range(16):
        spectral_axis_in_nm = np.poly1d(px_to_wl_coeff[k])(spectral_axis)
        step = np.mean(np.diff(spectral_axis_in_nm))
        plt.subplot(4, 4, k+1)
        plt.imshow(residuals[:, k].T, interpolation='none', aspect='auto')
        plt.colorbar()
        plt.title(LABELS[k])
        plt.xlabel('Wavelength (px)')
        plt.ylabel('Vertical position')
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'geo_calib_residuals.png')
