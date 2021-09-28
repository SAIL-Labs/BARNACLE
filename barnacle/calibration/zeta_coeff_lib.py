#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script gives the conversion coefficient between the null/antinull outputs
and the photometric ones, for every spectral channels.
It also estimate the splitting and coupling coefficients for characterization
purpose.
The data to process consists in files acquired when each beam is active one
after the other.
The name of the file must contain the keyword ``pX`` where X is the id of the
beam (1 to 4).

The script concatenate all the files of a given active beam into one frame
then extract the intensities into everey output.
The zeta coefficient (conversion factor) are computed by doing the ratio of
the interferometric output over the photometric one.

The wrapper ``do_zeta_coeff`` does all the operation at once.

**Nota Bene**: one spectral channel consists in one column of pixel.
The whole width of the frame is used, including the part with no signal.
Consequently, some coefficients (zeta, splitting or coupling) are absurd.

The inputs are set in the ``I/O`` section:
    * :datafolder: *(string)* -- folder containing the files to process \
                    for all beams
    * :root: *(string)* -- path to the root folder containing all\
            the subfolders containing all the products needed
    * :calibration_path: *(string)* -- folder containing the calibration data\
                        (spectral and geometric calibrations)
    * :output_path: *(string)* -- path to the folder where the products
                    of the script are saved

The settings are in the ``Settings`` section:
    * :nb_img: tuple, bounds between frames are selected. Leave ``None``\
                to start from the first frame or to finish to the\
                    last one (included).
    * :save: *(bool)* -- set to ``True`` to save the zeta coefficient.

The outputs are:
    * Some plots for characterization and monitoring purpose,\
        they are not automatically saved.
    * An HDF5 file containing the zeta coefficient.

Example available in ``examples/run_geometric_calibration.py``.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import barnacle.glint_classes as glint_classes
import h5py
from barnacle.glint_functions import get_channel_positions

NB_TRACKS = 16  # Number of tracks


def do_zeta_coeff(data_path, output_path,
                  geometric_calibration_path,
                  wl_to_px_coeff, px_to_wl_coeff, mode_flux,
                  spectral_binning=False, wl_bin_min=1525, wl_bin_max=1575,
                  bandwidth_binning=50, nb_img=(None, None),
                  save=True, plotting=True):
    """
    Wrapper to get, save and plot the zeta coefficient and intermediate\
        quantities.

    The spectral binning crop the channels to the wavelength of interest
    (``spectral_binning``, ``wl_bin_min``, ``wl_bin_max``)
    then bin the spectral channels inside the croped channels to the
    specified bigger spectral bandwidth (``bandwidth_binning``).

    :param data_path: path to the data folder
    :type data_path: string
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param geometric_calibration_path: path to the geometric calibration file.
    :type geometric_calibration_path: string
    :param wl_to_px_coeff: polynomial coefficients of the table converting
                    wavelength (in nm) into pixel position
    :type wl_to_px_coeff: array
    :param px_to_wl_coeff: polynomial coefficients of the table converting
                    pixel position into wavelength (in nm)
    :type px_to_wl_coeff: array
    :param mode_flux: mode of measurement of the spectral fluxes among\
                    'amplitude', 'model', 'windowed', 'raw'. 'raw' is\
                        preferred. See documentation of\
                            ``glint_classes.getSpectralFlux`` for more details.
    :type mode_flux: string
    :param spectral_binning: activate the spectral binning of the channels,\
                            defaults to False
    :type spectral_binning: boolean, optional
    :param wl_bin_min: lower wavelength above which the spectral binning is\
                        done, defaults to 1525
    :type wl_bin_min: int, optional
    :param wl_bin_max: upper wavelength below which the spectral binning is\
                        done, defaults to 1575
    :type wl_bin_max: int, optional
    :param bandwidth_binning: banwdith of the spectral binning, defaults to 50
    :type bandwidth_binning: int, optional
    :param nb_img: lower and upper bounds of the frames in one datacube,\
                    defaults to (None, None)
    :type nb_img: tuple, optional
    :param save: save zeta coefficients or plots if ``True``, defaults to True
    :type save: boolean, optional
    :param plotting: Display and save the plots if ``True``, defaults to True
    :type plotting: boolean, optional
    :return: zeta coefficients for each pair of input/output waveguides.
    :rtype: dict

    """
    zeta_coeff, img2, Imp, photometries = \
        get_zeta_coeff(data_path, output_path, geometric_calibration_path,
                       wl_to_px_coeff, px_to_wl_coeff, nb_img, mode_flux,
                       spectral_binning, wl_bin_min, wl_bin_max,
                       bandwidth_binning, save)

    Iminus, Iplus = Imp
    P1, P2, P3, P4 = photometries
    _save_zeta_coeff(zeta_coeff, output_path, mode_flux,
                     img2.wl_scale.mean(axis=0))

    if plotting:
        plot_zeta(zeta_coeff, img2.wl_scale[0], output_path, save)
        _plot_coupling_ratios(img2.wl_scale[0], Iminus, Iplus,
                              output_path, save)
        _plot_splitting_ratios(img2.wl_scale[0], Iminus, Iplus,
                               P1, P2, P3, P4, output_path, save)
        plot_for_publication(img2.wl_scale[0], Iminus, Iplus, P1,
                             zeta_coeff, output_path, save)

    return zeta_coeff


def _load_frames(data_list, nb_img, dark, beam, plotting=True):
    """
    Load the frames and average them.

    :param data_list: contains path to the data to load
    :type data_list: list
    :param nb_img: lower and upper bounds of the frames in one datacube,\
                    defaults to (None, None)
    :type nb_img: tuple
    :param dark: average dark frame
    :type dark: 2D-array
    :param beam: processed beam (1..4)
    :type beam: int
    :param plotting: display the spectral flux when beam X is active,\
                    defaults to True
    :type plotting: boolean, optional
    :return: averaged frame
    :rtype: 2D-array

    """
    super_data = np.zeros((344, 96))
    super_nb_img = 0
    for f in data_list:
        print("Process of : %s (%d / %d)" %
              (f, data_list.index(f)+1, len(data_list)))
        img = glint_classes.ChipProperties(f, nbimg=nb_img)
        print(img.data[:, :, 10].mean())

        super_data = super_data + img.data.sum(axis=0)
        super_nb_img += img.nbimg

    super_data = super_data / super_nb_img

    if plotting:
        plt.figure()
        plt.imshow(super_data-dark, interpolation='none')
        plt.colorbar()
        plt.title('P'+str(beam)+' on')

    return super_data


def _plot_flux(img2, output_path, beam, save):
    """
    Plot the fluxes in the outputs of GLINT when one beam is active.

    It is useful to check any cross-talk.

    :param img2: contains the extracted fluxes of the processed beam
    :type img2: class object
    :param output_path: DESCRIPTION
    :type output_path: string
    :param beam: processed beam (1..4)
    :type beam: int
    :param save: save zeta coefficients or plots if ``True``
    :type save: boolean
    :return: Nothing
    :rtype: Emptiness

    """
    plt.figure(figsize=(19.20, 10.80))
#    plt.suptitle('P%s on'%beam)
    plt.subplot(3, 4, 1)
    plt.title('P1')
    plt.plot(img2.wl_scale[0], img2.p1[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.p1[0]) > 5000:
        plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.p1[0])) > 1500: plt.ylim(-1, 1500)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 2)
    plt.title('P2')
    plt.plot(img2.wl_scale[0], img2.p2[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.p2[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 3)
    plt.title('P3')
    plt.plot(img2.wl_scale[0], img2.p3[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.p3[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 4)
    plt.title('P4')
    plt.plot(img2.wl_scale[0], img2.p4[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.p4[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 5)
    plt.title('N1 and N7 (12)')
    plt.plot(img2.wl_scale[0], img2.Iminus1[0])
    plt.plot(img2.wl_scale[0], img2.Iplus1[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.Iminus1[0]) > 5000 or np.max(img2.Iplus1[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 6)
    plt.title('N2 and N8 (23)')
    plt.plot(img2.wl_scale[0], img2.Iminus2[0])
    plt.plot(img2.wl_scale[0], img2.Iplus2[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.Iminus2[0]) > 5000 or np.max(img2.Iplus2[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 7)
    plt.title('N3 and N9 (14)')
    plt.plot(img2.wl_scale[0], img2.Iminus3[0])
    plt.plot(img2.wl_scale[0], img2.Iplus3[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.Iminus3[0]) > 5000 or np.max(img2.Iplus3[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 8)
    plt.title('N4 and N10 (34)')
    plt.plot(img2.wl_scale[0], img2.Iminus4[0])
    plt.plot(img2.wl_scale[0], img2.Iplus4[0])
    plt.grid()
    plt.ylim(-1)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 9)
    plt.title('N5 and N11 (13)')
    plt.plot(img2.wl_scale[0], img2.Iminus5[0])
    plt.plot(img2.wl_scale[0], img2.Iplus5[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.Iminus5[0]) > 5000 or np.max(img2.Iplus5[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.subplot(3, 4, 10)
    plt.title('N6 and N12 (24)')
    plt.plot(img2.wl_scale[0], img2.Iminus6[0])
    plt.plot(img2.wl_scale[0], img2.Iplus6[0])
    plt.grid()
    plt.ylim(-1)
    if np.max(img2.Iminus6[0]) > 5000 or np.max(img2.Iplus6[0]) > 5000:
        plt.ylim(ymax=5000)
    plt.ylabel('Intensity (AU)')
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'fluxes_p%s' % (beam)+'.png')


def get_zeta_coeff(data_path, output_path, geometric_calibration_path,
                   wl_to_px_coeff, px_to_wl_coeff, nb_img, mode_flux,
                   spectral_binning, wl_bin_min, wl_bin_max,
                   bandwidth_binning, save):
    """Get the zeta coefficients.

    This function is the core of the measurement of the zeta coefficients.

    :param data_path: path to the data folder
    :type data_path: string
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param geometric_calibration_path: path to the geometric calibration file.
    :type geometric_calibration_path: string
    :param wl_to_px_coeff: polynomial coefficients of the table converting
                    wavelength (in nm) into pixel position
    :type wl_to_px_coeff: array
    :param px_to_wl_coeff: polynomial coefficients of the table converting
                    pixel position into wavelength (in nm)
    :type px_to_wl_coeff: array
    :param nb_img: lower and upper bounds of the frames in one datacube,\
                    defaults to (None, None)
    :type nb_img: tuple
    :param mode_flux: mode of measurement of the spectral fluxes among\
                    'amplitude', 'model', 'windowed', 'raw'. 'raw' is\
                        preferred. See documentation of\
                            ``glint_classes.getSpectralFlux`` for more details.
    :type mode_flux: string
    :param spectral_binning: activate the spectral binning of the channels,\
                            defaults to False
    :type spectral_binning: boolean, optional
    :param wl_bin_min: lower wavelength above which the spectral binning is\
                        done, defaults to 1525
    :type wl_bin_min: int, optional
    :param wl_bin_max: upper wavelength below which the spectral binning is\
                        done, defaults to 1575
    :type wl_bin_max: int, optional
    :param bandwidth_binning: banwdith of the spectral binning, defaults to 50
    :type bandwidth_binning: int, optional
    :param save: save zeta coefficients or plots if ``True``
    :type save: boolean
    :return: zeta coefficients *(dict)*,
            extracted flux of current beam *(class object)*,
            Fluxes in the null and antinull outputs for all beams
            *(2-tuple of arrays)*,
            Fluxes in the photometric outputs for all beams
            *(4-tuple of arrays)*

    :rtype: tuple

    """
    # Init the outputs of the function
    Iminus = []
    Iplus = []
    P1, P2, P3, P4 = 0, 0, 0, 0
    zeta_coeff = {}

    # Check if the output folder exists and create it if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load calibration data
    dark = np.load(output_path+'superdark.npy')
    dark_per_channel = np.load(output_path+'superdarkchannel.npy')

    for beam in range(1, 5):
        data_list = [data_path +
                     f for f in os.listdir(data_path) if 'p'+str(beam) in f]

        # Set processing configuration and load instrumental calibration data
        pattern_coeff = np.load(geometric_calibration_path+'pattern_coeff.npy')
        position_outputs = pattern_coeff[:, :, 1].T
        width_outputs = pattern_coeff[:, :, 2].T

        spatial_axis = np.arange(dark.shape[0])
        spectral_axis = np.arange(dark.shape[1])

        # Define bounds of each channel
        channel_pos, sep = get_channel_positions(NB_TRACKS)

        # Do the data processing
        super_data = _load_frames(data_list, nb_img, dark, beam)

        img2 = glint_classes.ChipProperties(nbimg=(0, 1))
        img2.data = np.reshape(
            super_data, (1, super_data.shape[0], super_data.shape[1]))

        img2.cosmeticsFrames(np.zeros(dark.shape))

        # Insulating each channel
        print('Getting channels')
        img2.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)

        # Map the spectral channels between every chosen tracks before
        # computing the null depth
        img2.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)

        # Measurement of flux per frame, per spectral channel, per track
        img2.getSpectralFlux(spectral_axis, position_outputs,
                             width_outputs, mode_flux)
        img2.getIntensities(mode_flux)

        if spectral_binning:
            img2.spectralBinning(wl_bin_min, wl_bin_max,
                                 bandwidth_binning, wl_to_px_coeff)

        # Get split and coupler coefficient, biased with transmission coeff
        # between nulling-chip and detector
        img2.getRatioCoeff(beam, zeta_coeff)

        Iminus.append([img2.Iminus1[0], img2.Iminus2[0], img2.Iminus3[0],
                       img2.Iminus4[0], img2.Iminus5[0], img2.Iminus6[0]])
        Iplus.append([img2.Iplus1[0], img2.Iplus2[0], img2.Iplus3[0],
                      img2.Iplus4[0], img2.Iplus5[0], img2.Iplus6[0]])

        if beam == 1:
            P1 = img2.p1[0]
        if beam == 2:
            P2 = img2.p2[0]
        if beam == 3:
            P3 = img2.p3[0]
        if beam == 4:
            P4 = img2.p4[0]

        _plot_flux(img2, output_path, beam, save)

    Iplus = np.array(Iplus)
    Iminus = np.array(Iminus)

    return zeta_coeff, img2, (Iminus, Iplus), (P1, P2, P3, P4)


def _save_zeta_coeff(zeta_coeff, output_path, mode_flux, wl_scale):
    """
    Save the zeta coefficient in HDF5 format.

    It is a dictionary without tree structure.

    :param zeta_coeff: zeta coefficients
    :type zeta_coeff: dict
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param mode_flux: mode of measurement of the spectral fluxes among\
                    'amplitude', 'model', 'windowed', 'raw'. 'raw' is\
                        preferred. See documentation of\
                            ``glint_classes.getSpectralFlux`` for more details.
    :type mode_flux: string
    :param wl_scale: wavelengths scale in nm
    :type wl_scale: array
    :return: Nothing
    :rtype: Emptiness

    """
    with h5py.File(output_path+'/zeta_coeff_'+mode_flux+'.hdf5', 'w')\
            as f:
        f.create_dataset('wl_scale', data=wl_scale)
        f['wl_scale'].attrs['comment'] = 'wl in nm'
        for key in zeta_coeff.keys():
            f.create_dataset(key, data=zeta_coeff[key][0])


def _format_zeta_keys(zeta_coeff):
    """
    Create a list of the keys of the 'zeta_coeff' dictionary to label plots.

    :param zeta_coeff: zeta coefficients
    :type zeta_coeff: dict
    :return: array of keys of the zeta coefficients dictionary,\
            array of labels for plots based on these keys.
    :rtype: 2-tuple

    """
    keys = np.array(list(zeta_coeff.keys()))
    keys_title = np.array([elt[0].upper()+'eam '+elt[1]+' to ' +
                           elt[2:6].capitalize()+' '+elt[6:] for elt in keys])
    keys_title = keys_title.reshape(4, 6)
    keys = keys.reshape(4, 6)
    return keys, keys_title


def plot_zeta(zeta_coeff, wl_scale, output_path,  save):
    """
    Plot the zeta coefficients.

    The figure can be saved.

    :param zeta_coeff: zeta coefficients
    :type zeta_coeff: dict
    :param wl_scale: wavelengths scale in nm
    :type wl_scale: array
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: save zeta coefficients or plots if ``True``
    :type save: boolean
    :return: Nothing
    :rtype: Emptiness

    """
    keys, keys_title = _format_zeta_keys(zeta_coeff)
    fig = plt.figure(figsize=(19.20, 10.80))
    grid = plt.GridSpec(4, 6, wspace=0.2, hspace=0.4,
                        left=0.03, bottom=0.05, right=0.98, top=0.92)
    plt.suptitle('Zeta coefficients')
    for i in range(4):
        for j in range(6):
            fig.add_subplot(grid[i, j])
            plt.plot(wl_scale, zeta_coeff[keys[i, j]][0])
            plt.grid()
            plt.title(keys_title[i, j])
            if i == 3:
                plt.xlabel('Wavelength (nm)')
            if j == 0:
                plt.ylabel(r'$\zeta$ coeff')
            plt.ylim(-0.2, 5)
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'zeta_coeff.png')


def _plot_coupling_ratios(wl_scale, Iminus, Iplus, output_path, save):
    """
    Plot the coupling ratios for all couplers.

    The figure can be saved.

    :param wl_scale: wavelengths scale in nm
    :type wl_scale: array
    :param Iminus: fluxes in the null outputs for all beams
    :type Iminus: array
    :param Iplus: fluxes in the antinull outputs for all beams
    :type Iplus: array
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: save zeta coefficients or plots if ``True``
    :type save: boolean
    :return: Nothing
    :rtype: Emptiness

    """
    plt.figure(figsize=(19.20, 10.80))
    plt.subplot(2, 3, 1)
    plt.title('Coupling ratio for N1 (12)')
    plt.plot(wl_scale, Iminus[0, 0] /
             (Iminus[0, 0]+Iplus[0, 0]), label='Beam 1 to N1')
    plt.plot(wl_scale, Iplus[0, 0] /
             (Iminus[0, 0]+Iplus[0, 0]), label='Beam 1 to AN1')
    plt.plot(wl_scale, Iminus[1, 0] /
             (Iminus[1, 0]+Iplus[1, 0]), label='Beam 2 to N1')
    plt.plot(wl_scale, Iplus[1, 0] /
             (Iminus[1, 0]+Iplus[1, 0]), label='Beam 2 to AN1')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2, 3, 2)
    plt.title('Coupling ratio for N2 (23)')
    plt.plot(wl_scale, Iminus[1, 1] /
             (Iminus[1, 1]+Iplus[1, 1]), label='Beam 2 to N2')
    plt.plot(wl_scale, Iplus[1, 1] /
             (Iminus[1, 1]+Iplus[1, 1]), label='Beam 2 to AN2')
    plt.plot(wl_scale, Iminus[2, 1] /
             (Iminus[2, 1]+Iplus[2, 1]), label='Beam 3 to N2')
    plt.plot(wl_scale, Iplus[2, 1] /
             (Iminus[2, 1]+Iplus[2, 1]), label='Beam 3 to AN2')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2, 3, 3)
    plt.title('Coupling ratio for N3 (14)')
    plt.plot(wl_scale, Iminus[0, 2] /
             (Iminus[0, 2]+Iplus[0, 2]), label='Beam 1 to N3')
    plt.plot(wl_scale, Iplus[0, 2] /
             (Iminus[0, 2]+Iplus[0, 2]), label='Beam 1 to AN3')
    plt.plot(wl_scale, Iminus[3, 2] /
             (Iminus[3, 2]+Iplus[3, 2]), label='Beam 4 to N3')
    plt.plot(wl_scale, Iplus[3, 2] /
             (Iminus[3, 2]+Iplus[3, 2]), label='Beam 4 to AN3')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2, 3, 4)
    plt.title('Coupling ratio for N4 (34)')
    plt.plot(wl_scale, Iminus[2, 3] /
             (Iminus[2, 3]+Iplus[2, 3]), label='Beam 3 to N4')
    plt.plot(wl_scale, Iplus[2, 3] /
             (Iminus[2, 3]+Iplus[2, 3]), label='Beam 3 to AN4')
    plt.plot(wl_scale, Iminus[3, 3] /
             (Iminus[3, 3]+Iplus[3, 3]), label='Beam 4 to N4')
    plt.plot(wl_scale, Iplus[3, 3] /
             (Iminus[3, 3]+Iplus[3, 3]), label='Beam 4 to AN4')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2, 3, 5)
    plt.title('Coupling ratio for N5 (13)')
    plt.plot(wl_scale, Iminus[0, 4] /
             (Iminus[0, 4]+Iplus[0, 4]), label='Beam 1 to N5')
    plt.plot(wl_scale, Iplus[0, 4] /
             (Iminus[0, 4]+Iplus[0, 4]), label='Beam 1 to AN5')
    plt.plot(wl_scale, Iminus[2, 4] /
             (Iminus[2, 4]+Iplus[2, 4]), label='Beam 3 to N5')
    plt.plot(wl_scale, Iplus[2, 4] /
             (Iminus[2, 4]+Iplus[2, 4]), label='Beam 3 to AN5')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2, 3, 6)
    plt.title('Coupling ratio for N6 (24)')
    plt.plot(wl_scale, Iminus[1, 5] /
             (Iminus[1, 5]+Iplus[1, 5]), label='Beam 2 to N6')
    plt.plot(wl_scale, Iplus[1, 5] /
             (Iminus[1, 5]+Iplus[1, 5]), label='Beam 2 to AN6')
    plt.plot(wl_scale, Iminus[3, 5] /
             (Iminus[3, 5]+Iplus[3, 5]), label='Beam 4 to N6')
    plt.plot(wl_scale, Iplus[3, 5] /
             (Iminus[3, 5]+Iplus[3, 5]), label='Beam 4 to AN6')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'coupling_ratios.png')


def _plot_splitting_ratios(wl_scale, Iminus, Iplus,
                           P1, P2, P3, P4, output_path, save):
    """
    Plot the splitting ratios of all splitters.

    The figure can be saved.

    :param wl_scale: wavelengths scale in nm
    :type wl_scale: array
    :param Iminus: fluxes in the null outputs for all beams
    :type Iminus: array
    :param Iplus: fluxes in the antinull outputs for all beams
    :type Iplus: array
    :param P1: fluxes in the photometric output of beam 1
    :type P1: array
    :param P2: fluxes in the photometric output of beam 2
    :type P2: array
    :param P3: fluxes in the photometric output of beam 3
    :type P3: array
    :param P4: fluxes in the photometric output of beam 4
    :type P4: array
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: save zeta coefficients or plots if ``True``
    :type save: boolean
    :return: Nothing
    :rtype: Emptiness

    """
    plt.figure(figsize=(19.20, 10.80))
    plt.subplot(2, 2, 1)
    plt.title('Splitting ratio for Beam 1')
    plt.plot(wl_scale, P1/(P1 + Iminus[0, 0] + Iplus[0, 0] + Iminus[0, 2] +
                           Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4]),
             label='Photometry')
    plt.plot(wl_scale, Iminus[0, 0]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                                     Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4]
                                     + Iplus[0, 4]), label='N1')
    plt.plot(wl_scale, Iplus[0, 0]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                                    Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4]
                                    + Iplus[0, 4]), label='AN1')
    plt.plot(wl_scale, Iminus[0, 2]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                                     Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4]
                                     + Iplus[0, 4]), label='N3')
    plt.plot(wl_scale, Iplus[0, 2]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                                    Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4]
                                    + Iplus[0, 4]), label='AN3')
    plt.plot(wl_scale, Iminus[0, 4]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                                     Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4]
                                     + Iplus[0, 4]), label='N5')
    plt.plot(wl_scale, Iplus[0, 4]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                                    Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4]
                                    + Iplus[0, 4]), label='AN5')
    plt.grid()
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.subplot(2, 2, 2)
    plt.title('Splitting ratio for Beam 2')
    plt.plot(wl_scale, P2/(P2 + Iminus[1, 0] + Iplus[1, 0] + Iminus[1, 1] +
                           Iplus[1, 1] + Iminus[1, 5] + Iplus[1, 5]),
             label='Photometry')
    plt.plot(wl_scale, Iminus[1, 0]/(P2 + Iminus[1, 0] + Iplus[1, 0] +
                                     Iminus[1, 1] + Iplus[1, 1] + Iminus[1, 5]
                                     + Iplus[1, 5]), label='N1')
    plt.plot(wl_scale, Iplus[1, 0]/(P2 + Iminus[1, 0] + Iplus[1, 0] +
                                    Iminus[1, 1] + Iplus[1, 1] + Iminus[1, 5]
                                    + Iplus[1, 5]), label='AN1')
    plt.plot(wl_scale, Iminus[1, 1]/(P2 + Iminus[1, 0] + Iplus[1, 0] +
                                     Iminus[1, 1] + Iplus[1, 1] + Iminus[1, 5]
                                     + Iplus[1, 5]), label='N2')
    plt.plot(wl_scale, Iplus[1, 1]/(P2 + Iminus[1, 0] + Iplus[1, 0] +
                                    Iminus[1, 1] + Iplus[1, 1] + Iminus[1, 5]
                                    + Iplus[1, 5]), label='AN2')
    plt.plot(wl_scale, Iminus[1, 5]/(P2 + Iminus[1, 0] + Iplus[1, 0] +
                                     Iminus[1, 1] + Iplus[1, 1] + Iminus[1, 5]
                                     + Iplus[1, 5]), label='N6')
    plt.plot(wl_scale, Iplus[1, 5]/(P2 + Iminus[1, 0] + Iplus[1, 0] +
                                    Iminus[1, 1] + Iplus[1, 1] + Iminus[1, 5]
                                    + Iplus[1, 5]), label='AN6')
    plt.grid()
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.subplot(2, 2, 3)
    plt.title('Splitting ratio for Beam 3')
    plt.plot(wl_scale, P3/(P3 + Iminus[2, 1] + Iplus[2, 1] + Iminus[2, 3] +
                           Iplus[2, 3] + Iminus[2, 4] + Iplus[2, 4]),
             label='Photometry')
    plt.plot(wl_scale, Iminus[2, 1]/(P3 + Iminus[2, 1] + Iplus[2, 1] +
                                     Iminus[2, 3] + Iplus[2, 3] + Iminus[2, 4]
                                     + Iplus[2, 4]), label='N2')
    plt.plot(wl_scale, Iplus[2, 1]/(P3 + Iminus[2, 1] + Iplus[2, 1] +
                                    Iminus[2, 3] + Iplus[2, 3] + Iminus[2, 4]
                                    + Iplus[2, 4]), label='AN2')
    plt.plot(wl_scale, Iminus[2, 3]/(P3 + Iminus[2, 1] + Iplus[2, 1] +
                                     Iminus[2, 3] + Iplus[2, 3] + Iminus[2, 4]
                                     + Iplus[2, 4]), label='N4')
    plt.plot(wl_scale, Iplus[2, 3]/(P3 + Iminus[2, 1] + Iplus[2, 1] +
                                    Iminus[2, 3] + Iplus[2, 3] + Iminus[2, 4]
                                    + Iplus[2, 4]), label='AN4')
    plt.plot(wl_scale, Iminus[2, 4]/(P3 + Iminus[2, 1] + Iplus[2, 1] +
                                     Iminus[2, 3] + Iplus[2, 3] + Iminus[2, 4]
                                     + Iplus[2, 4]), label='N5')
    plt.plot(wl_scale, Iplus[2, 4]/(P3 + Iminus[2, 1] + Iplus[2, 1] +
                                    Iminus[2, 3] + Iplus[2, 3] + Iminus[2, 4]
                                    + Iplus[2, 4]), label='AN5')
    plt.grid()
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.subplot(2, 2, 4)
    plt.title('Splitting ratio for Beam 4')
    plt.plot(wl_scale, P4/(P4 + Iminus[3, 2] + Iplus[3, 2] + Iminus[3, 5] +
                           Iplus[3, 5] + Iminus[3, 3] + Iplus[3, 3]),
             label='Photometry')
    plt.plot(wl_scale, Iminus[3, 2]/(P4 + Iminus[3, 2] + Iplus[3, 2] +
                                     Iminus[3, 5] + Iplus[3, 5] + Iminus[3, 3]
                                     + Iplus[3, 3]), label='N3')
    plt.plot(wl_scale, Iplus[3, 2]/(P4 + Iminus[3, 2] + Iplus[3, 2] +
                                    Iminus[3, 5] + Iplus[3, 5] + Iminus[3, 3]
                                    + Iplus[3, 3]), label='AN3')
    plt.plot(wl_scale, Iminus[3, 5]/(P4 + Iminus[3, 2] + Iplus[3, 2] +
                                     Iminus[3, 5] + Iplus[3, 5] + Iminus[3, 3]
                                     + Iplus[3, 3]), label='N6')
    plt.plot(wl_scale, Iplus[3, 5]/(P4 + Iminus[3, 2] + Iplus[3, 2] +
                                    Iminus[3, 5] + Iplus[3, 5] + Iminus[3, 3]
                                    + Iplus[3, 3]), label='AN6')
    plt.plot(wl_scale, Iminus[3, 3]/(P4 + Iminus[3, 2] + Iplus[3, 2] +
                                     Iminus[3, 5] + Iplus[3, 5] + Iminus[3, 3]
                                     + Iplus[3, 3]), label='N4')
    plt.plot(wl_scale, Iplus[3, 3]/(P4 + Iminus[3, 2] + Iplus[3, 2] +
                                    Iminus[3, 5] + Iplus[3, 5] + Iminus[3, 3]
                                    + Iplus[3, 3]), label='AN4')
    plt.grid()
    plt.ylim(0, 1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'splitting_ratios.png')


def plot_for_publication(wl_scale, Iminus, Iplus, P1, zeta_coeff,
                         output_path, save):
    """
    Plot zeta coefficients, coupling and splitting ratios for one coupler\
    and one splitter.

    The figure can be saved.

    :param wl_scale: wavelengths scale in nm
    :type wl_scale: array
    :param Iminus: fluxes in the null outputs for all beams
    :type Iminus: array
    :param Iplus: fluxes in the antinull outputs for all beams
    :type Iplus: array
    :param P1: fluxes in the photometric output of beam 1
    :type P1: array
    :param zeta_coeff: zeta coefficients
    :type zeta_coeff: dict
    :param output_path: path where to save or load the intermediate products.
    :type output_path: string
    :param save: save zeta coefficients or plots if ``True``
    :type save: boolean
    :return: Nothing
    :rtype: Emptiness
    """
    mask = (wl_scale >= 1300) & (wl_scale <= 1650)
    longueuronde = wl_scale[mask]
    a = P1/(P1 + Iminus[0, 0] + Iplus[0, 0] + Iminus[0, 2] +
            Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4])
    b = Iminus[0, 0]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                      Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4])
    c = Iplus[0, 0]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                     Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4])
    d = Iminus[0, 2]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                      Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4])
    e = Iplus[0, 2]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                     Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4])
    f = Iminus[0, 4]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                      Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4])
    g = Iplus[0, 4]/(P1 + Iminus[0, 0] + Iplus[0, 0] +
                     Iminus[0, 2] + Iplus[0, 2] + Iminus[0, 4] + Iplus[0, 4])

    keys = _format_zeta_keys(zeta_coeff)[0]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(19.20, 10.80))
    ax = plt.subplot(111)
    plt.plot(longueuronde, a[mask], lw=4,
             c='k', label='To photometric output')
    plt.plot(longueuronde, b[mask]+c[mask], ':',
             lw=4, c=colors[0], label='To coupler of Null 1')
    plt.plot(longueuronde, d[mask]+e[mask], '--',
             lw=4, c=colors[1], label='To coupler of Null 3')
    plt.plot(longueuronde, f[mask]+g[mask], '-.',
             lw=4, c=colors[2], label='To coupler of Null 5')
    plt.grid()
    plt.ylim(-0.03, 0.6)
    plt.legend(loc='best', fontsize=34, ncol=2)
    plt.xlabel('Wavelength (nm)', size=45)
    plt.ylabel('Splitting ratio', size=45)
    plt.xticks(size=38)
    plt.yticks(size=38)
    plt.text(-0.09, 1.02, r'a)', weight='bold',
             fontsize=40, transform=ax.transAxes)
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'splitting_ratioN1.pdf', format='pdf')

    plt.figure(figsize=(19.20, 10.80))
    ax = plt.subplot(111)
    a = Iminus[0, 0]/(Iminus[0, 0]+Iplus[0, 0])
    b = Iplus[0, 0]/(Iminus[0, 0]+Iplus[0, 0])
    c = Iminus[1, 0]/(Iminus[1, 0]+Iplus[1, 0])
    d = Iplus[1, 0]/(Iminus[1, 0]+Iplus[1, 0])
    plt.plot(longueuronde, a[mask], lw=4,
             c=colors[0], label='Beam 1 to null output')
    plt.plot(longueuronde, b[mask], '--', lw=6,
             c=colors[0], label='Beam 1 to antinull output')
    plt.plot(longueuronde, c[mask], ':', lw=4,
             c=colors[1], label='Beam 2 to null output')
    plt.plot(longueuronde, d[mask], '--', lw=4,
             c=colors[1], label='Beam 2 to antinull output')
    plt.grid()
    plt.legend(loc='best', fontsize=34, ncol=2)
    plt.ylim(-0.05, 1.3)
    # plt.xlim(1200)
    plt.xlabel('Wavelength (nm)', size=45)
    plt.ylabel('Coupling ratio', size=45)
    plt.xticks(size=38)
    plt.yticks(size=38)
    plt.text(-0.09, 1.02, r'b)', weight='bold',
             fontsize=40, transform=ax.transAxes)
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'coupling_ratioN1.pdf', format='pdf')

    plt.figure(figsize=(19.20, 10.80))
    ax = plt.subplot(111)
    plt.plot(wl_scale[mask], zeta_coeff[keys[0, 0]][0]
             [mask], '-', c=colors[0], lw=4, label='Beam 1 to null output')
    plt.plot(wl_scale[mask], zeta_coeff[keys[0, 3]][0][mask],
             ':', c=colors[0],  lw=4, label='Beam 1 to antinull output')
    plt.plot(wl_scale[mask], zeta_coeff[keys[1, 0]][0]
             [mask], '--', c=colors[1], lw=4, label='Beam 2 to null output')
    plt.plot(wl_scale[mask], zeta_coeff[keys[1, 3]][0][mask],
             '-.', c=colors[1],  lw=4, label='Beam 2 to antinull output')
    plt.grid()
    plt.legend(loc='best', fontsize=34)
    plt.xticks(size=38)
    plt.yticks(size=38)
    plt.ylim(-0.1)
    plt.xlabel('Wavelength (nm)', size=45)
    plt.ylabel(r'$\zeta$ coefficient', size=45)
    plt.text(-0.09, 1.02, r'c)', weight='bold',
             fontsize=40, transform=ax.transAxes)
    plt.tight_layout()
    if save:
        plt.savefig(output_path+'zeta_coeffN1.pdf', format='pdf')
