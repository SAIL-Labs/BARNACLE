# -*- coding: utf-8 -*-
"""
Classes used by the GLINT Data Reduction Software
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit


def gaussian_curve(x, A, loc, sig):
    """
    Computes a Gaussian curve

    :Parameters:

        **x**: values where the curve is estimated.

        **A**: amplitude of the Gaussian.

        **x0**: location of the Gaussian.

        **sig**: scale of the Gaussian.

    :Returns:

        Gaussian curve.
    """
    return A * np.exp(-(x-loc)**2/(2*sig**2))


def norm_gaussian_with_linear_curve(x, A, B, C, loc, sig):
    """
    Computes a gaussian curve

    :Parameters:

        **x: (N,) array**
            Values for which the gaussian is estimated

        **A: float**
            amplitude of the gaussian curve

        **loc: float**
            center of the curve

        **sig: float>0**
            scale factor of the curve

    :Returns:

        The gaussian curve respect to x values
    """
    gaus = np.exp(-(x-loc)**2/(2*sig**2))
    normalisation = np.sum(gaus)
    return A * gaus / normalisation + B * x + C


def gaussian_with_offset(x, A, x0, sig, offset):
    """
    Computes a Gaussian curve

    :Parameters:

        **x**: values where the curve is estimated.

        **a**: amplitude of the Gaussian.

        **x0**: location of the Gaussian.

        **sig**: scale of the Gaussian.

    :Returns:

        Gaussian curve.
    """
    return A * np.exp(-(x-x0)**2/(2*sig**2)) + offset


def getSpectralFluxDebug(nbimg, which_tracks, slices_axes, slices,
                         spectral_axis, positions, widths):
    """
    Debug version of _getSpectralFluxNumba.
    Called when ``debug`` is ``True``.

    For development and experimentation purpose.
    Plot the linear fit and the gaussian profil for one spectral channel
    of the first frame for every tracks.
    Read the description of ``_getSpectralFluxNumba`` for details about
    the inputs.
    """
    nb_tracks = 16
    amplitude_fit = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    amplitude = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    # integ_model = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    # integ_windowed = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    residuals_fit = np.zeros(
        (nbimg, nb_tracks, len(spectral_axis), slices_axes.shape[1]))
    error = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    residuals_reg = np.zeros(
        (nbimg, nb_tracks, len(spectral_axis), slices_axes.shape[1]))
    cov = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    # weights = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    labels = ['P4', 'N3', 'P3', 'N2', 'AN4', 'N5', 'N4', 'N6',
              'AN1', 'AN6', 'N1', 'AN2', 'P2', 'AN3', 'AN5', 'P1']

    # With fitted amplitude
    for k in range(nbimg):
        #        print(k)
        for i in which_tracks:
            for j in range(len(spectral_axis)):
                gaus = partial(norm_gaussian_with_linear_curve,
                               loc=positions[i, j], sig=widths[i, j])
                try:
                    popt, pcov = curve_fit(gaus, slices_axes[i], slices[k, j, i], p0=[
                                           slices[k, j, i].max(), 0, 0])
                except:
                    popt = np.zeros((3,))

                amplitude_fit[k, i, j] = popt[0]
                cov[k, i, j] = pcov[0, 0]
                # integ_model[k,i,j] = np.sum(gaus(slices_axes[i], *popt))
                # weight = gaus(slices_axes[i], 1., 0)
                # weight /= weight.sum()
                # integ_windowed[k,i,j] = np.sum(weight * slices[k,j,i])
                residuals_fit[k, i, j] = slices[k, j, i] - \
                    gaus(slices_axes[i], *popt)

                simple_gaus0 = np.exp(-(slices_axes[i] -
                                        positions[i, j])**2/(2*widths[i, j]**2))
                simple_gaus = simple_gaus0 / np.sum(simple_gaus0)
                simple_gaus[np.isnan(simple_gaus)] = 0.
                # A = np.vstack((simple_gaus, np.ones_like(simple_gaus)))
                A = np.vstack(
                    (simple_gaus, np.ones_like(simple_gaus), slices_axes[i]))
                A = np.transpose(A)
                try:
                    popt2 = np.linalg.lstsq(A, slices[k, j, i], rcond=None)[0]
#                    popt2 = np.linalg.solve(A.T.dot(A), A.T.dot(slices[k,j,i]))
                except ValueError as e:
                    print(simple_gaus0)
                    print(np.any(np.isnan(simple_gaus)),
                          np.any(np.isinf(simple_gaus)))
                    print(labels[i], 'Track', i, 'Frame', k, 'Column', j)
                    print('Centre axe', np.mean(
                        slices_axes[i]), 'Loc', positions[i, j], 'Width', widths[i, j])
                    raise e
                except np.linalg.LinAlgError as e:
                    print(simple_gaus0)
                    print(np.any(np.isnan(simple_gaus)),
                          np.any(np.isinf(simple_gaus)))
                    print(labels[i], 'Track', i, 'Frame', k, 'Column', j)
                    print('Centre axe', np.mean(
                        slices_axes[i]), 'Loc', positions[i, j], 'Width', widths[i, j])
                    print(e)
                    popt2 = np.zeros((3,))

                res = slices[k, j, i] - (popt2[0] * simple_gaus +
                                         popt2[1] + popt2[2] * slices_axes[i])
                chi2 = np.sum(res**2) / (slices_axes[i].size-len(popt2))
                error[k, i, j] = (
                    chi2 / np.sum((slices_axes[i] - slices_axes[i].mean())**2))**0.5
                residuals_reg[k, i, j] = res
                amplitude[k, i, j] = popt2[0]
                # integ_model[k,i,j] = np.sum(simple_gaus * popt2[0])
                # weight = simple_gaus.copy()
                # weight /= np.sum(weight)
                # integ_windowed[k,i,j] = np.sum(weight * slices[k,j,i])
                # weights[k,i,j] = weight.sum()

#                switch = True
#                if abs(popt) > 1.e+4 or abs(popt2[0]) > 1.e+4:
#                    if abs(popt) > 1.e+4:
#                        debug.append([0, k, i, j])
#                    if abs(popt2[0]) > 1.e+4:
#                        debug.append([1, k, i, j])
                if j == 59 and k == 0:
                    # print(k, i, j)
                    # print('Weight on std', (np.sum((simple_gaus/simple_gaus.sum())**2))**0.5)
                    # print(slices[k,j,i][:7].std())
                    plt.figure()
                    plt.subplot(211)
                    plt.plot(slices_axes[i],
                             slices[k, j, i], 'o', label='data')
                    plt.plot(slices_axes[i], gaus(
                        slices_axes[i], *popt), '+-', label='curve_fit %s' % (popt))
                    # plt.plot(slices_axes[i], popt2[0]* simple_gaus + popt2[1], '+--', label='linear reg %s'%(popt2))
                    plt.plot(slices_axes[i], popt2[0] * simple_gaus + popt2[1] +
                             popt2[2] * slices_axes[i], '+--', label='linear reg %s' % (popt2))
                    plt.xlabel('Spatial position (px)')
                    plt.ylabel('Amplitude')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.title('Frame '+str(k)+'/ Track '+str(i) +
                              '/ Column '+str(j)+'/ '+labels[i])
#                    plt.subplot(312)
#                    plt.plot(slices[k,j,i], residuals_fit[k,i,j], 'o', label='fit')
#                    plt.plot(slices[k,j,i], residuals_reg[k,i,j], 'd', label='linear reg')
#                    plt.xlabel('Amplitude')
#                    plt.ylabel('Residual')
#                    plt.grid()
#                    plt.legend(loc='best')
                    plt.subplot(212)
                    plt.plot(slices_axes[i], residuals_fit[k, i, j], 'o', label='fit (%s)' % (
                        np.mean(residuals_fit[k, i, j])))
                    plt.plot(slices_axes[i], residuals_reg[k, i, j], 'd', label='linear reg (%s, %s)' % (
                        np.mean(residuals_reg[k, i, j]), error[k, i, j]))
                    plt.xlabel('Spatial position (px)')
                    plt.ylabel('Residual')
                    plt.grid()
                    plt.legend(loc='best')
#
#                    if switch == True:
#                        temp = gaus(slices_axes[i], 1.)
#                        temp2 = simple_gaus
#                        switch = False

    # return amplitude_fit, amplitude, integ_model, integ_windowed, residuals_fit, residuals_reg, cov, weights
    return amplitude_fit, amplitude, residuals_fit, residuals_reg, cov, error


def get_channel_positions(nb_tracks):
    """
    Get the channel vertical positions.

    :return: vertical positions of the ouputs.
    :rtype: array

    """
    y_ends = [33, 329]  # row of top and bottom-most Track
    sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)
    channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))
    return channel_pos, sep
