#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes the spectral calibration of the 16 outputs.
This script requires the library :doc:`glint_classes` to work.

The inputs are **averaged dark**, **datacubes with spectral bands** and the
**data from the geometric calibration**
(cf :doc:`glint_geometric_calibration`).
It is assumed one data file contains only one spectral band with
its wavelength in the name.
The script successively loads the data files related to one wavelength and
extracts the 16 outputs.
For each of them, we assume the spectral band is shaped as a Gaussian.
A model fitting with this shape is performed to get the position and width
(in pixel).
Once all files of all wavelength are processed, a polynomial fit
is performed to map the wavelength to the column of pixels
for each output.

The outputs products are:
        * The polynomial coefficients mapping the wavelength respect \
            to the column of pixels
        * The polynomial coefficients mapping he column of pixels respect to\
            the wavelengths
        * The spectral psf, giving the spectral resolution.

The outputs are saved into numpy-format file (.npy).

This script is used in 3 steps.

First step: specify the following settings:
    * :save: *(boolean)* -- ``True`` for saving products and monitoring data,
            ``False`` otherwise
    * :wavelength: *(list)* -- list of wavelengths used to acquire the data

Second step: specify the paths to load data and save outputs:
    * :datafolder: *(string)* -- folder containing the datacube to use.
    * :data_list: *(string)* -- list of files in **datafolder** to open.
    * :output_path: *(string)* -- path to the folder where
                    the products are saved.
    * :calibration_path: *(string)* -- path to the calibration files used to
                        process the file
                        (location, width of the outputs, etc.).

Third step: call the function ``do_sectral_calibration``.

Example available in ``examples/run_spectral_calibration.py``.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import barnacle.glint_classes as glint_classes
from scipy.optimize import curve_fit
from barnacle.glint_functions import gaussian_with_offset,\
    get_channel_positions

NB_TRACKS = 16


def do_sectral_calibration(data_path, wavelength, save, output_path,
                           prompt_spectral_resolution=True, plotting=True):
    """
    Wrapper to get and save the spectral calibration for each channel at once.

    :param data_path: path to the data to load.
    :type data_list0: string
    :param wavelength: list of the wavelengths to use for
                        the spectral calibration.
    :type wavelength: list-like
    :param save: if ``True``, save some outputs of the function.
    :type save: boolean
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param prompt_spectral_resolution: display the spectral resolutions,
                                        defaults to True.
    :type prompt_spectral_resolution: boolean, optional
    :param plotting: Plot figures, defaults to True.
                    If ``save`` is True, the figures are saved in
                    ``output_path``.
    :type plotting: boolean, optional
    :return: coefficients of the polynoms linking wavelength to pixels
            positions and vice versa, the positions and sigma of Gaussian
            profiles for the different wavelength-cases.
    :rtype: 3-tuple

    """
    # Output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_list0 = [[data_path+f for f in os.listdir(data_path)
                   if str(wl)+'_' in f] for wl in wavelength]

    # Define bounds of each channel
    channel_pos, sep = get_channel_positions(NB_TRACKS)

    calib_pos = extract_wl_pos(data_list0, wavelength,
                               channel_pos, sep, save, output_path, plotting)

    coeff_poly_wl_to_px, coeff_poly_px_to_wl, spectral_psf = \
        get_spectral_calibration(wavelength, calib_pos)

    if prompt_spectral_resolution:
        display_spectral_resolution(wavelength, calib_pos, coeff_poly_px_to_wl,
                                    output_path, save)
    if plotting:
        plot_spectral_laws(wavelength, calib_pos, coeff_poly_wl_to_px,
                           output_path, save)

    if save:
        np.save(output_path+'wl_to_px', coeff_poly_wl_to_px)
        np.save(output_path+'px_to_wl', coeff_poly_px_to_wl)
        np.save(output_path+'spectral_psf', spectral_psf)

    return coeff_poly_wl_to_px, coeff_poly_px_to_wl, spectral_psf


def extract_wl_pos(data_list0, wavelength,
                   channel_pos, sep, save, output_path, plotting):
    """
    Extract the position of the fluxes for a given wavelength.

    :param data_list0: contains sub-lists. Each sub-list contains the data
                        acquired at one wavelength.
    :type data_list0: list
    :param wavelength: list of wavelength to set the spectral calibration.
    :type wavelength: list
    :param channel_pos: positions of the channels.
    :type channel_pos: list or array
    :param sep: spatial gap between two successive channels, in pixel.
    :type sep: float
    :param save: if ``True``, save some outputs of the function.
    :type save: boolean
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param plotting: Plot figures.
                    If ``save`` is True, the figures are saved in
                    ``output_path``.
    :type plotting: boolean
    :return: x-coordinates of each wavelength for each channel.
    :rtype: array

    """
    calib_pos = []
    dark = np.load(output_path+'superdark.npy')
    dark_per_channel = np.load(output_path+'superdarkchannel.npy')
    super_img = np.zeros(dark.shape)
    super_nb_img = 0.
    slices = np.zeros_like(dark_per_channel)
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])

    for data_list in data_list0:
        print('Processing wavelength %s' %
              (wavelength[data_list0.index(data_list)]))
        print('Averaging frames')
        for f in data_list[:]:
            img = glint_classes.Null(f)
            img.cosmeticsFrames(np.zeros(dark.shape))
            img.getChannels(channel_pos, sep, spatial_axis,
                            dark=dark_per_channel)
            super_img = super_img + img.data.sum(axis=0)
            slices = slices + np.sum(img.slices, axis=0)
            super_nb_img = super_nb_img + img.nbimg

        slices = slices / super_nb_img
        super_img = super_img / super_nb_img

        labels = ['P4', 'N3', 'P3', 'N2', 'AN4', 'N5', 'N4', 'AN5',
                  'N6', 'AN1', 'AN6', 'N1', 'AN2', 'P2', 'AN3', 'P1']

        # Fit a gaussian on every track and spectral channel
        # to get their positions, widths and amplitude
        print('Determine position and width of the outputs')
        img = glint_classes.Null(data=None, nbimg=(0, 1))
        img.cosmeticsFrames(np.zeros(dark.shape))
        img.getChannels(channel_pos, sep, spatial_axis)

        img.slices = slices
        # Average along spatial axis
        tracks = img.slices[:, :, 10-4:10+5].mean(axis=2)
        tracks = np.transpose(tracks)

        wl_pos = []
        for i in range(NB_TRACKS):
            popt = curve_fit(gaussian_with_offset, spectral_axis,
                             tracks[i],
                             p0=[tracks[i].max(),
                                 spectral_axis[np.argmax(tracks[i])],
                                 1., 0])[0]
            wl_pos.append(popt[1:-1])

            if plotting:
                plt.figure(10+data_list0.index(data_list),
                           figsize=(19.2, 10.8))
                plt.subplot(4, 4, i+1)
                plt.plot(spectral_axis, tracks[i], '.')
                plt.plot(spectral_axis, gaussian_with_offset(
                    spectral_axis, *popt))
                plt.grid('on')
                plt.title(labels[i])
                plt.suptitle(str(wavelength[data_list0.index(data_list)]))
                plt.tight_layout()
                if save:
                    plt.savefig(output_path+'fitting_%s' %
                                (wavelength[data_list0.index(data_list)]))
        calib_pos.append(wl_pos)

    calib_pos = np.array(calib_pos)
    return calib_pos


def get_spectral_calibration(wavelength, calib_pos):
    """
    Perform the spectral calibration.

    :param wavelength: list of wavelength to set the spectral calibration.
    :type wavelength: list
    :param calib_pos: x-coordinates of each wavelength for each channel.
    :type calib_pos: array
    :return: coefficients of the polynoms linking wavelength to pixels
            positions and vice versa.
    :rtype: tuple

    """
    # detector resolution is around 5 nm/px
    coeff_poly_wl_to_px = np.array([np.polyfit(wavelength, calib_pos[:, i, 0],
                                               deg=1)
                                    for i in range(NB_TRACKS)])
    coeff_poly_px_to_wl = np.array(
        [np.polyfit(calib_pos[:, i, 0], wavelength,
                    deg=1) for i in range(NB_TRACKS)])
    poly_px = [np.poly1d(coeff_poly_px_to_wl[i]) for i in range(NB_TRACKS)]

    spectral_psf_pos = np.array(
        [poly_px[i](calib_pos[:, i, 0]) for i in range(16)]).T
    spectral_psf_sig = calib_pos[:, :, 1] * \
        abs(coeff_poly_px_to_wl[None, :, 0])
    spectral_psf = np.stack([spectral_psf_pos, spectral_psf_sig], axis=2)

    return coeff_poly_wl_to_px, coeff_poly_px_to_wl, spectral_psf


def display_spectral_resolution(wavelength, calib_pos, coeff_poly_px_to_wl,
                                output_path, save):
    """
    Print the spectral resolution for each tested wavelength.

    As the calibration source may be slightly resolved by the
    GLINT spectrograph, an deconvolved version is also given.
    The print can be saved in a text file.

    :param wavelength: list of wavelength to set the spectral calibration.
    :type wavelength: list
    :param calib_pos: x-coordinates of each wavelength for each channel.
    :type calib_pos: array
    :param coeff_poly_px_to_wl: coefficient of the polynom converting
                                positions in pixel to wavelength.
    :type coeff_poly_px_to_wl: array
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: if ``True``, save some outputs of the function.
    :type save: boolean
    :return: Nothing
    :rtype: Vacuum

    """
    fwhm = 2 * np.sqrt(2*np.log(2)) * \
        calib_pos[:, :, 1] * abs(coeff_poly_px_to_wl[None, :, 0])
    print('Spectral resolution for')
    for wl in wavelength:
        print(str(wl)+' nm -> ' +
              str(wl/fwhm.mean(axis=1)[wavelength.index(wl)]))

    print('Deconvolution from CHARIS tunable laser')
    fwhm2 = fwhm.mean(axis=1)
    measured_sigma = fwhm2 / (2 * np.sqrt(2*np.log(2)))
    # measured_sigma2 = calib_pos[:, :, 1] * \
    #                     abs(coeff_poly_px_to_wl[None, :, 0])

    # Check the CHARIS source cal documentation for updating x/x2 and y/y2
    # x = np.array([400, 1000])
    # y = np.array([1, 2])
    x2 = np.array([1000, 2300])
    y2 = np.array([2, 5])
    # coeff = np.polyfit(x, y, 1)
    coeff2 = np.polyfit(x2, y2, 1)
    # p = np.poly1d(coeff)
    p2 = np.poly1d(coeff2)
    laser_sig = p2(wavelength) / (2 * np.sqrt(2*np.log(2)))
    deconv_sig = (measured_sigma**2 - laser_sig**2)**0.5
    # deconv_sig2 = (measured_sigma2**2 - laser_sig[:, None]**2)**0.5
    deconv_fwhm = deconv_sig * 2 * np.sqrt(2*np.log(2))
    # deconv_fwhm2 = deconv_sig2 * 2 * np.sqrt(2*np.log(2))
    for wl in wavelength:
        print(str(wl)+' nm -> '+str(wl/deconv_fwhm[wavelength.index(wl)]))

    if save:
        with open(output_path+'spectral_resolution.txt', 'a') as sr:
            sr.write('Spectral resolution for:\n')
            for wl in wavelength:
                sr.write(str(wl)+' nm -> \t' +
                         str(wl/fwhm.mean(axis=1)[wavelength.index(wl)])+'\n')
            sr.write('\n')
            sr.write('Spectral resolution after deconvolution for:\n')
            for wl in wavelength:
                sr.write(str(wl)+' nm -> \t' +
                         str(wl/deconv_fwhm[wavelength.index(wl)])+'\n')
            sr.write('\n')


def plot_spectral_laws(wavelength, calib_pos, coeff_poly_wl_to_px,
                       output_path, save):
    """
    Display the plots of the positions in pixel in function of the wavelength.

    These plots monitor the goodness of fit of the spectral law (linear).

    :param wavelength: list of wavelength to set the spectral calibration.
    :type wavelength: list
    :param calib_pos: x-coordinates of each wavelength for each channel.
    :type calib_pos: array
    :param coeff_poly_wl_to_px: coefficient of the polynom converting
                                wavelength to positions in pixel.
    :type coeff_poly_wl_to_px: array
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: if ``True``, save some outputs of the function.
    :type save: boolean
    :return: Nothing
    :rtype: Vacuum

    """
    poly_wl = [np.poly1d(coeff_poly_wl_to_px[i]) for i in range(NB_TRACKS)]

    plt.figure(figsize=(19.2, 10.8))
    for i in range(NB_TRACKS):
        plt.subplot(4, 4, i+1)
        plt.plot(wavelength, calib_pos[:, i, 0], 'o')
        plt.plot(wavelength, poly_wl[i](wavelength))
        plt.grid()
        plt.title('Track %s' % (i+1))
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'wl2px.png', format='png', dpi=150)
