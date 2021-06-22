# -*- coding: utf-8 -*-
"""
This script measures the intensity in every output, per spectral channel and
computes the null depths.
It relies on the library :doc:`glint_classes` to work.

The inputs are the datacubes (target or dark).
The datacube of **dark** needs to be processed as it gives the distribution of
dark currents in the different outputs used in the model fitting later.
The datacube of **target** gives the **null depth** and the intensities are
used to monitor any suspicious behavior if needed.

The products are HDF5 files structured as a dictionary
(see glint_classes documentation).
One HDF5 produces per datacube.

It contains:
    * 1d-arrays for the null depth, named **nullX** (with X=1..6).
        The elements of the array are the null depths per spectral channel.
    * 1d-array for the photometry, named **pX** (with X=1..4).
        It has the same structure as ``nullX``.
    * 1d-array for the intensity in the null outputs, named **IminusX**
        (with X=1..6).
    * 1d-array for the intensity in the anti-null outputs, named **IplusX**
        (with X=1..6).
    * 1d-array containing the common spectral channels for all the outputs
        (as each output is slightly shifted from the others)

Some monitoring data can be created (but not saved):
    * Histograms of intensities of the photometries
    * Optimal parameters from a Gaussian fitting of the intensity profile in
        one output for one spectral channel
    * Evolution of the null depths along the frame axis for 10 spectral
        channels
    * Evolution of the measured intensities along the frame axis of every
        outputs according to different estimators

The monitoring is activated by setting the boolean variable **debug = True**.

In that case, it is strongly advised deactivate the save of the results and
to process one frame of one datacube to avoid extremely long data processing.


In order to increase the SNR in the extraction of the flux in the photometric
output, ones can create the spectra in them by averaging the frames.
The spectra are then normalized so that their integral in the bandwidth
is equal to 1.
Therefore, the extraction of the photometries on a frame basis first estimates
the total flux in bandwidth then the spectral flux is given by the product of
this total flux with the spectra.
However, the gain of SNR is barely significant so this mode should not be used.


This script is used in 3 steps.

First step: simply change the value of the variables in the **Settings**
section:
    * **save**: boolean, ``True`` for saving products and monitoring data,
        ``False`` otherwise
    * **nb_files**: 2-tuple of int, set the bounds between which the data files
        are selected. ``None`` is equivalent to 0 if it is the lower bound or
            -1 included or it is the upper one.
    * **nb_img**: 2-tuple of int, set the bounds between which the frame are
        selected, into a data file.
    * **nulls_to_invert**: list of null outputs to invert. Fill with ``nullX``
        (X=1..6) or leave empty if no null is to invert (deprecated)
    * **bin_frames**: boolean, set True to bin frames
    * **nb_frames_to_bin**: number of frames to bin (average) together.
        If ``None``, the whole stack is average into one frame.
        If the total number of frames is not a multiple of the binning value,
        the remaining frames are lost.
    * **spectral_binning**: bool, set to ``True`` to spectrally bins the
        outputs
    * **wl_bin_min**: scalar, lower bounds (in nm) of the bandwidth to bin,
        possibly in several chunks
    * **wl_bin_max**: scalar, upper bounds (in nm) of the bandwidth to bin,
        possibly in several chunks
    * **bandwidth_binning**: scalar, width of the chunks of spectrum to bin
        between the lower and upper bounds
    * **mode_flux**: string, choose the method to estimate the spectral flux
        in the outputs among:
        * ``amplitude`` uses patterns determined in the script
            ``glint_geometric_calibration`` and a linear least square is
            performed to get the amplitude of the pattern
        * ``model`` proceeds like ``amplitude`` but the integral of the flux
            is returned
        * ``windowed`` returns a weighted mean as flux of the spectral channel.
            The weights is the same pattern as the other modes above
        * ``raw`` returns the mean of the flux along the spatial axis over
            the whole width of the output
    * **activate_estimate_spectrum**, boolean, if ``True``, the spectrum of
        the source in the photometric output is created.
    * **nb_files_spectrum**: tuple, range of files to read to get the spectra.
    * **wavelength_bounds**: tuple, bounds of the bandwidth one wants to keep
        after the extraction. Used in the method ``getIntensities``.
            It works independantly of **wl_bin_min** and **wl_bin_max**.
    * **suffix**: str, suffix to distinguish plots respect to data present in
        the datafolder (e.g. dark, baselines, stars...)

Second step: change the value of the variables in the **Inputs** and
**Outputs** sections:
    * **datafolder**: folder containing the datacube to use.
    * **root**: path to **datafolder**.
    * **data_list**: list of files in **datafolder** to open.
    * **spectral_calibration_path**: path to the spectral calibration files
        used to process the file
    * **geometric_calibration_path**: path to the geometric calibration files
        used to process the file (location and width of the outputs per
                                  spectral channel)

Third step: call the function ``reduce_data'' and let the script be in charge.

Example available in ``examples/run_glint_reduction.py``.

**NB**: monitoring data can be voluminous if the data set to process is
several thousands of files or higher.
The program will getting slower and a ``MemoryError'' may rise.
The solution could be to chunk the data set by changing ``nb_files'',
keep in mind that the plots
can be erased by the next batch if :plot_name: and :suffix: are the same.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import barnacle.glint_classes as glint_classes
from timeit import default_timer as time
from scipy.optimize import curve_fit
from barnacle.glint_functions import gaussian_curve, get_channel_positions
from barnacle.tests.check_tools import check_datalist_not_empty, \
    check_mode_flux_exists

NB_TRACKS = 16  # Number of tracks


def reduce_data(data_path, plot_name, output_path, suffix, nb_files, nb_img,
                nb_frames_to_bin,
                geometric_calibration_path, spectral_calibration_files,
                save, bin_frames, debug, spectral_binning, wl_bin_bounds,
                bandwidth_binning,
                activate_estimate_spectrum,
                nb_files_spectrum, mode_flux, wavelength_bounds):
    """
    Wrapper to reduce the data.

    :param data_path: path to data
    :type data_path: string
    :param plot_name: name used to save figures
    :type plot_name: string
    :param output_path: path where to save or load the intermediate products
    :type output_path: string
    :param suffix: ID-like of the data to process
    :type suffix: string
    :param nb_files: lower and upper border of the list of files to process
    :type nb_files: 2-tuple-like
    :param nb_img: lower and upper border of the list of frames inside\
        a datacube to process
    :type nb_img: 2-tuple-like
    :param nb_frames_to_bin: number of frames to bin inside a file. If bigger\
            than the number of frames inside a datacube, the whole datacube is\
            binned
    :type nb_frames_to_bin: int
    :param geometric_calibration_path: path to the geometrical calibration file
    :type geometric_calibration_path: string
    :param spectral_calibration_files: path to the spectral calibration file
    :type spectral_calibration_files: string
    :param save: save intermediate products and reduced data if ``True''
    :type save: boolean
    :param bin_frames: bin frames if ``True''
    :type bin_frames: boolean
    :param debug: activate *debug* mode if ``True''
    :type debug: boolean
    :param spectral_binning: activate the binning of some spectral channels,\
        if ``True''
    :type spectral_binning: boolean
    :param wl_bin_bounds: lower and upper bound of the bandwidth to bin
    :type wl_bin_bounds: 2-tuple-like of int
    :param bandwidth_binning: bandwidth to bin, in nm. If equal or higher than\
        the difference between ``wl_bin_bounds'' values, the whole bandwidth
        is binned. Otherwise, the bandwidth is chunked into pieces of
        ``bandwidth_binning'' wide which are binned.
    :type bandwidth_binning: int
    :param activate_estimate_spectrum: determine the spectrum in the\
        photometric outputs before reducing the data.
    :type activate_estimate_spectrum: boolean
    :param nb_files_spectrum: lower and upper bound of files to process\
        to determine the spectra of the photometric outputs
    :type nb_files_spectrum: 2-tuple of int
    :param mode_flux: mode of measurement of the flux in spectral channels\
        for all outputs. 2 modes are currently set: *raw* and *fit*.
    :type mode_flux: string
    :param wavelength_bounds: lower and upper bound of the bandwidth between\
        which the data are reduced
    :type wavelength_bounds: 2-tuple of int
    :return: miscelleanous monitor products
    :rtype: 4-tuple

    """
    data_list = sorted(
        [data_path+f for f in os.listdir(data_path) if suffix in f])

    check_datalist_not_empty(data_list)
    check_mode_flux_exists(mode_flux)

    if activate_estimate_spectrum:
        get_spectrum(data_list, output_path,
                     geometric_calibration_path, spectral_calibration_files,
                     nb_files_spectrum, nb_img, mode_flux, wavelength_bounds)

    monitor_amplitude, monitor_null, monitor_photo, wl_scale = \
        extract_data(data_list, nb_files, nb_img, nb_frames_to_bin,
                     output_path, geometric_calibration_path,
                     spectral_calibration_files, mode_flux, wavelength_bounds,
                     activate_estimate_spectrum, wl_bin_bounds, save,
                     bin_frames, debug, spectral_binning, bandwidth_binning)

    plots(monitor_amplitude, monitor_null, monitor_photo, wl_scale,
          output_path, suffix, plot_name, save, debug)

    return monitor_amplitude, monitor_null, monitor_photo, wl_scale


def get_spectrum(data_list, output_path,
                 geometric_calibration_path, spectral_calibration_files,
                 nb_files_spectrum, nb_img, mode_flux, wavelength_bounds
                 ):
    """
    Extract the spectrum for each photometric output.

    :param data_list: contains path to the data to load
    :type data_list: list
    :param output_path: path where to save or load the intermediate products
    :type output_path: string
    :param geometric_calibration_path: path to the geometrical calibration file
    :type geometric_calibration_path: string
    :param spectral_calibration_files: path to the spectral calibration file
    :type spectral_calibration_files: string
    :param nb_files_spectrum: lower and upper bound of files to process\
        to determine the spectra of the photometric outputs
    :type nb_files_spectrum: 2-tuple of int
    :param nb_img: lower and upper border of the list of frames inside\
        a datacube to process
    :type nb_img: 2-tuple-like
    :param mode_flux: mode of measurement of the flux in spectral channels\
        for all outputs. 2 modes are currently set: *raw* and *fit*.
    :type mode_flux: string
    :param wavelength_bounds: lower and upper bound of the bandwidth between\
        which the data are reduced
    :type wavelength_bounds: 2-tuple of int

    """
    # Get the spectrum of photometric channels
    nb_frames = 0
    dark, dark_per_channel = _load_dark(output_path)
    slices_spectrum = np.zeros_like(dark_per_channel)
    channel_pos, sep = get_channel_positions(NB_TRACKS)
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])
    position_outputs, width_outputs = \
        _load_geometric_calibration(geometric_calibration_path)
    wl_to_px_coeff, px_to_wl_coeff = \
        _load_spectral_calibration(*spectral_calibration_files)

    print('Determining spectrum\n')
    for f in data_list[nb_files_spectrum[0]:nb_files_spectrum[1]]:
        start = time()
        print("Process of : %s (%d / %d)"
              % (f, data_list.index(f)+1,
                 len(data_list[nb_files_spectrum[0]:nb_files_spectrum[1]]))
              )
        img_spectrum = _measure_flux(
            f, nb_img, 0, output_path,
            geometric_calibration_path, spectral_calibration_files,
            mode_flux, wavelength_bounds, False, False)

        img_spectrum = glint_classes.Null(f, nbimg=nb_img)

        slices_spectrum = slices_spectrum + \
            np.sum(img_spectrum.slices, axis=0)
        nb_frames = nb_frames + img_spectrum.nbimg
        stop = time()
        print('Spectrum time:', stop-start)

    slices_spectrum = slices_spectrum / nb_frames
    spectrum = glint_classes.Null(data=None, nbimg=(0, 1))
    spectrum.cosmeticsFrames(np.zeros(dark.shape))
    spectrum.getChannels(channel_pos, sep, spatial_axis)
    spectrum.slices = np.reshape(
        slices_spectrum, (1, slices_spectrum.shape[0],
                          slices_spectrum.shape[1],
                          slices_spectrum.shape[2]))
    spectrum.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
    spectrum.getSpectralFlux(
        spectral_axis, position_outputs, width_outputs, mode_flux)

    spectrum.getIntensities(mode=mode_flux, wl_bounds=wavelength_bounds)
    spectrum.p1 = spectrum.p1[0] / spectrum.p1[0].sum()
    spectrum.p2 = spectrum.p2[0] / spectrum.p2[0].sum()
    spectrum.p3 = spectrum.p3[0] / spectrum.p3[0].sum()
    spectrum.p4 = spectrum.p4[0] / spectrum.p4[0].sum()
    spectra = np.array(
        [spectrum.p1, spectrum.p2, spectrum.p3, spectrum.p4])
    del spectrum, img_spectrum
    np.save(output_path+'spectra', spectra)


def extract_data(data_list, nb_files, nb_img, nb_frames_to_bin, output_path,
                 geometric_calibration_path, spectral_calibration_files,
                 mode_flux, wavelength_bounds,
                 activate_estimate_spectrum, wl_bin_bounds,
                 save, bin_frames, debug, spectral_binning, bandwidth_binning):
    """
    Extract the flux of the spectral channels for each outputs.

    :param data_list: contains path to the data to load
    :type data_list: list
    :param nb_files: lower and upper border of the list of files to process
    :type nb_files: 2-tuple-like
    :param nb_img: lower and upper border of the list of frames inside\
        a datacube to process
    :type nb_img: 2-tuple-like
    :param nb_frames_to_bin: number of frames to bin inside a file. If bigger\
            than the number of frames inside a datacube, the whole datacube is\
            binned
    :type nb_frames_to_bin: int
    :param output_path: path where to save or load the intermediate products
    :type output_path: string
    :param geometric_calibration_path: path to the geometrical calibration file
    :type geometric_calibration_path: string
    :param spectral_calibration_files: path to the spectral calibration file
    :type spectral_calibration_files: string
    :param mode_flux: mode of measurement of the flux in spectral channels\
        for all outputs. 2 modes are currently set: *raw* and *fit*.
    :type mode_flux: string
    :param wavelength_bounds: lower and upper bound of the bandwidth between\
        which the data are reduced
    :type wavelength_bounds: 2-tuple of int
    :param activate_estimate_spectrum: determine the spectrum in the\
        photometric outputs before reducing the data.
    :type activate_estimate_spectrum: boolean
    :param wl_bin_bounds: lower and upper bound of the bandwidth to bin
    :type wl_bin_bounds: 2-tuple-like of int
    :param save: save intermediate products and reduced data if ``True''
    :type save: boolean
    :param bin_frames: bin frames if ``True''
    :type bin_frames: boolean
    :param debug: activate *debug* mode if ``True''
    :type debug: boolean
    :param spectral_binning: activate the binning of some spectral channels,\
        if ``True''
    :type spectral_binning: boolean
    :param bandwidth_binning: bandwidth to bin, in nm. If equal or higher than\
        the difference between ``wl_bin_bounds'' values, the whole bandwidth
        is binned. Otherwise, the bandwidth is chunked into pieces of
        ``bandwidth_binning'' wide which are binned.
    :type bandwidth_binning: int
    :return: miscelleanous monitor products
    :rtype: 4-tuple

    """
    ''' Output lists for different stages of the processing.
    Include data from all processed files '''
    amplitude = []
    amplitude_fit = []
    integ_raw = []
    integ_raw_err = []
    integ_model = []
    integ_windowed = []
    residuals_reg = []
    residuals_fit = []
    fluxes = np.zeros((1, 4))
    null = []
    null_err = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []

    dark, dark_per_channel = _load_dark(output_path)
    channel_pos, sep = get_channel_positions(NB_TRACKS)
    position_outputs, width_outputs = \
        _load_geometric_calibration(geometric_calibration_path)
    wl_to_px_coeff, px_to_wl_coeff = \
        _load_spectral_calibration(*spectral_calibration_files)

    if os.path.exists(output_path+'spectra.npy') and \
            activate_estimate_spectrum:
        spectra = np.load(output_path+'spectra.npy')

    ''' Start the data processing '''
    nb_frames = 0
    for f in data_list[nb_files[0]:nb_files[1]]:
        start = time()
        print("Process of : %s (%d / %d)" %
              (f, data_list.index(f)+1,
               len(data_list[nb_files[0]:nb_files[1]])))
        img = _measure_flux(
            f, nb_img, nb_frames_to_bin, output_path,
            geometric_calibration_path, spectral_calibration_files,
            mode_flux, wavelength_bounds, bin_frames, debug)

        if activate_estimate_spectrum:
            integ = np.array([np.sum(img.p1, axis=1), np.sum(
                img.p2, axis=1), np.sum(img.p3, axis=1),
                np.sum(img.p4, axis=1)])
            new_photo = integ[:, :, None] * spectra[:, None, :]

            img.p1, img.p2, img.p3, img.p4 = new_photo

        if spectral_binning:
            wl_bin_min, wl_bin_max = wl_bin_bounds
            img.spectralBinning(wl_bin_min, wl_bin_max,
                                bandwidth_binning, wl_to_px_coeff)

        p1.append(img.p1)
        p2.append(img.p2)
        p3.append(img.p3)
        p4.append(img.p4)

        ''' Compute null depth '''
        print('Computing null depths')
        img.computeNullDepth()
        null_depths = np.array(
            [img.null1, img.null2, img.null3, img.null4, img.null5, img.null6])
        null_depths_err = np.array(
            [img.null1_err, img.null2_err, img.null3_err, img.null4_err,
             img.null5_err, img.null6_err])

        ''' Output file'''
        if save:
            img.save(output_path+os.path.basename(f)
                     [:-4]+'.hdf5', '2019-04-30')
            print('Saved')

        null.append(np.transpose(null_depths, axes=(1, 0, 2)))
        null_err.append(np.transpose(null_depths_err, axes=(1, 0, 2)))
        if debug:
            amplitude.append(img.amplitude)
            residuals_reg.append(img.amplitude_error)
            integ_raw.append(img.raw)
            integ_raw_err.append(img.raw_err)

            try:  # if debug mode TRUE in getSpectralFlux method
                residuals_fit.append(img.residuals_fit)
                amplitude_fit.append(img.amplitude_fit)
            except AttributeError:
                pass

        ''' For following the evolution of flux in every tracks '''
        print('Getting total flux')
        img.getTotalFlux()
        fluxes = np.vstack((fluxes, img.fluxes.T))
        nb_frames += img.nbimg
        stop = time()
        print('Last: %.3f' % (stop-start))

    '''
    Store quantities for monitoring purpose
    '''
    amplitude = np.array([selt for elt in amplitude for selt in elt])
    amplitude = np.array([[elt[i][img.px_scale[i]]
                           for i in range(16)] for elt in amplitude])
    amplitude_fit = np.array([selt for elt in amplitude_fit for selt in elt])
    amplitude_fit = np.array([[elt[i][img.px_scale[i]]
                               for i in range(16)] for elt in amplitude_fit])
    integ_raw = np.array([selt for elt in integ_raw for selt in elt])
    integ_raw = np.array([[elt[i][img.px_scale[i]]
                           for i in range(16)] for elt in integ_raw])
    integ_raw_err = np.array([selt for elt in integ_raw_err for selt in elt])
    integ_raw_err = np.array([[elt[i][img.px_scale[i]]
                               for i in range(16)] for elt in integ_raw_err])
    integ_model = np.array([selt for elt in integ_model for selt in elt])
    integ_model = np.array([[elt[i][img.px_scale[i]]
                             for i in range(16)] for elt in integ_model])
    integ_windowed = np.array([selt for elt in integ_windowed for selt in elt])
    integ_windowed = np.array([[elt[i][img.px_scale[i]]
                                for i in range(16)] for elt in integ_windowed])
    residuals_reg = np.array([selt for elt in residuals_reg for selt in elt])
    residuals_reg = np.array([[elt[i][img.px_scale[i]]
                               for i in range(16)] for elt in residuals_reg])
    residuals_fit = np.array([selt for elt in residuals_fit for selt in elt])
    residuals_fit = np.array([[elt[i][img.px_scale[i]]
                               for i in range(16)] for elt in residuals_fit])
    null = np.array([selt for elt in null for selt in elt])
    null_err = np.array([selt for elt in null_err for selt in elt])
    p1 = np.array([selt for elt in p1 for selt in elt])
    p2 = np.array([selt for elt in p2 for selt in elt])
    p3 = np.array([selt for elt in p3 for selt in elt])
    p4 = np.array([selt for elt in p4 for selt in elt])
    fluxes = fluxes[1:]

    monitor_amplitude = (amplitude, amplitude_fit, integ_raw, integ_raw_err,
                         integ_model, integ_windowed, residuals_reg,
                         residuals_fit)
    monitor_null = (null, null_err)
    monitor_photo = (p1, p2, p3, p4)

    return monitor_amplitude, monitor_null, monitor_photo, img.wl_scale


def plots(monitor_amplitude, monitor_null, monitor_photo, wl_scale,
          output_path, suffix, plot_name, save, debug):
    """
    Plot figures of the monitored data.

    Monitored data are some parameters of the function.

    :param monitor_amplitude: contains the flux of the spectral channels\
        of all the outputs measured from different methods
    :type monitor_amplitude: list-like
    :param monitor_null: contains the null depths of the spectral channels\
        of 6 baselines deduced from the flux measurements methods
    :type monitor_null: list-like
    :param monitor_photo: contains the flux of the photometric outputs
    :type monitor_photo: list-like
    :param wl_scale: wavelength scale
    :type wl_scale: list-like
    :param output_path: path where to save or load the intermediate products
    :type output_path: string
    :param suffix: ID-like of the data to process
    :type suffix: string
    :param plot_name: name used to save figures
    :type plot_name: string
    :param save: save intermediate products and reduced data if ``True''
    :type save: boolean
    :param debug: activate *debug* mode if ``True''
    :type debug: boolean

    """
    amplitude, amplitude_fit, integ_raw, integ_raw_err,\
        integ_model, integ_windowed, residuals_reg,\
        residuals_fit = monitor_amplitude

    p1, p2, p3, p4 = monitor_photo

    null, null_err = monitor_null

    photometries = [p1.mean(axis=1), p2.mean(
        axis=1), p3.mean(axis=1), p4.mean(axis=1)]

    for k in range(len(photometries)):
        photo = photometries[k]
        histo, bin_edges = np.histogram(photo, int(photo.size**0.5))
        binning = bin_edges[:-1] + np.diff(bin_edges)/2
        histo = histo / np.sum(histo)
        popt, pcov = curve_fit(gaussian_curve, binning, histo, p0=[
            max(histo), photo.mean(), photo.std()])
        y = gaussian_curve(binning, *popt)

        fig = plt.figure(figsize=(19.20, 10.80))
        ax = fig.add_subplot(111)
        plt.plot(binning, histo/histo.max(), 'o-', label='P%s' % (k+1))
        plt.plot(binning, y/y.max(), '--', lw=4, label='Fit of P%s' % (k+1))
        plt.grid()
        plt.legend(loc='best', fontsize=36)
        plt.xticks(size=36)
        plt.yticks(size=36)
        plt.xlabel('Bins', size=40)
        plt.ylabel('Counts (normalised)', size=40)
        txt = r'$\mu_{p%s} = %.3f$' % (
            k+1, popt[1]) + '\n' + r'$\sigma_{p%s} = %.3f$' % (k+1, popt[2])
        plt.text(0.05, 0.3, txt, va='center', fontsize=30,
                 transform=ax.transAxes,
                 bbox=dict(boxstyle="square", facecolor='white'))
        if save:
            plt.savefig(output_path+plot_name+'_'+suffix +
                        '_plot_histo_p%s.png' % (k+1))

    for k in range(len(photometries)):
        plt.figure(figsize=(19.20, 10.80))
        plt.plot(np.arange(photometries[k].size)[
                 ::100], photometries[k][::100])
        plt.grid()
        plt.xlabel('Frame/100', size=30)
        plt.ylabel('Fitted amplitude', size=30)
        plt.xticks(size=30)
        plt.yticks(size=30)

    if debug:
        for k in range(1):
            plt.figure()
            plt.suptitle('Amplitude respect to different methods')
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.title('Track '+str(i+1))
                plt.plot(amplitude[k, i, :], '^')
                plt.plot(integ_raw[k, i, :], 'o')
                plt.plot(integ_windowed[k, i, :], 'd')
                plt.plot(integ_model[k, i, :], '+')
                plt.grid()

        plt.figure()
        plt.suptitle('Amplitude')
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.title('Track '+str(i+1))
            plt.plot(wl_scale[i], amplitude[0, i, :], 'o')
            plt.ylim(-50, 800)
            plt.grid()

    for i in range(5):
        plt.figure()
        plt.suptitle('Null')
        plt.subplot(321)
        plt.plot(wl_scale[0], null[i][0], '.')
        plt.grid()
        plt.subplot(322)
        plt.plot(wl_scale[0], null[i][1], '.')
        plt.grid()
        plt.subplot(323)
        plt.plot(wl_scale[0], null[i][2], '.')
        plt.grid()
        plt.subplot(324)
        plt.plot(wl_scale[0], null[i][3], '.')
        plt.grid()
        plt.subplot(325)
        plt.plot(wl_scale[0], null[i][4], '.')
        plt.grid()
        plt.subplot(326)
        plt.plot(wl_scale[0], null[i][5], '.')
        plt.grid()


def _load_dark(output_path):
    """
    Load the dark files.

    :param output_path: path where to save or load the intermediate products
    :type output_path: string
    :return: average dark frame and average dark per channel
    :rtype: 2-tuple of arrays

    """
    dark = np.load(output_path+'superdark.npy')
    dark_per_channel = np.load(output_path+'superdarkchannel.npy')
    return dark, dark_per_channel


def _load_geometric_calibration(geometric_calibration_path):
    """
    Load geometric calibration file.

    :param geometric_calibration_path: path to the geometric calibration file
    :type geometric_calibration_path: string
    :return: 2 lists: vertical positions of the outputs, their widths
    :rtype: 2-tuple of lists

    """
    pattern_coeff = np.load(geometric_calibration_path+'pattern_coeff.npy')
    position_outputs = pattern_coeff[:, :, 1].T
    width_outputs = pattern_coeff[:, :, 2].T

    return position_outputs, width_outputs


def _load_spectral_calibration(spectral_calibration_path_wl_px,
                               spectral_calibration_path_px_wl):
    """
    Load spectral calibration files.

    :param spectral_calibration_path_wl_px: path to the spectral calibration\
        file converting the wavelength into pixel column for each output
    :type spectral_calibration_path_wl_px: string
    :param spectral_calibration_path_px_wl: path to the spectral calibration\
        file converting the pixel column into wavelength for each output
    :type spectral_calibration_path_px_wl: string
    :return: DESCRIPTION
    :rtype: TYPE

    """
    wl_to_px_coeff = np.load(spectral_calibration_path_wl_px)
    px_to_wl_coeff = np.load(spectral_calibration_path_px_wl)

    return wl_to_px_coeff, px_to_wl_coeff


def _measure_flux(datacube, nb_img, nb_frames_to_bin, output_path,
                  geometric_calibration_path, spectral_calibration_files,
                  mode_flux, wavelength_bounds, bin_frames, debug):
    """
    Measure the flux of the spectral channels for each outputs.

    :param datacube: path to the data file to process
    :type datacube: string
    :param nb_img: lower and upper border of the list of frames inside\
        a datacube to process
    :type nb_img: 2-tuple-like
    :param nb_frames_to_bin: number of frames to bin inside a file. If bigger\
            than the number of frames inside a datacube, the whole datacube is\
            binned
    :type nb_frames_to_bin: int
    :param output_path: path where to save or load the intermediate products
    :type output_path: string
    :param geometric_calibration_path: path to the geometrical calibration file
    :type geometric_calibration_path: string
    :param spectral_calibration_files: path to the spectral calibration file
    :type spectral_calibration_files: string
    :param mode_flux: mode of measurement of the flux in spectral channels\
        for all outputs. 2 modes are currently set: *raw* and *fit*.
    :type mode_flux: string
    :param wavelength_bounds: lower and upper bound of the bandwidth between\
        which the data are reduced
    :type wavelength_bounds: 2-tuple of int
    :param bin_frames: bin frames if ``True''
    :type bin_frames: boolean
    :param debug: activate *debug* mode if ``True''
    :type debug: boolean
    :return: object containing the reduced data of the :datacube:
    :rtype: class-object

    """
    dark, dark_per_channel = _load_dark(output_path)
    channel_pos, sep = get_channel_positions(NB_TRACKS)
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])
    position_outputs, width_outputs = \
        _load_geometric_calibration(geometric_calibration_path)
    wl_to_px_coeff, px_to_wl_coeff = \
        _load_spectral_calibration(*spectral_calibration_files)

    img = glint_classes.Null(datacube, nbimg=nb_img)

    ''' Process frames '''
    if bin_frames:
        img.data = img.binning(
            img.data, nb_frames_to_bin, axis=0, avg=True)
        img.nbimg = img.data.shape[0]
    img.cosmeticsFrames(np.zeros(dark.shape))

    ''' Insulating each track '''
    print('Getting channels')
    img.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
#        img.slices = img.slices + np.random.normal(0, ron, img.slices.shape)

    ''' Map the spectral channels between every chosen tracks before computing
    the null depth'''
    img.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)

    ''' Measurement of flux per frame, per spectral channel, per track '''
    img.getSpectralFlux(spectral_axis, position_outputs,
                        width_outputs, mode_flux, debug=debug)

    ''' Reconstruct flux in photometric channels '''
    img.getIntensities(mode=mode_flux, wl_bounds=wavelength_bounds)

    return img
