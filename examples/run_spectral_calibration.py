"""
This script is an example on how to use the library ``spectral_calibration.py``
to obtain the spectral calibration.
"""

from barnacle.calibration.spectral_calibration import *

''' Settings '''
save = True

''' Inputs '''
datafolder = '20200130/spectral_calibration/'
data_path = '//tintagel.physics.usyd.edu.au/snert/GLINTData/'+datafolder
output_path = '//tintagel.physics.usyd.edu.au/snert/GLINTprocessed/'+datafolder


''' Iterate on wavelength '''
wavelength = [1450, 1550, 1650]

coeff_poly_wl_to_px, coeff_poly_px_to_wl, spectral_psf = \
    do_sectral_calibration(data_path, wavelength, save, output_path,
                           prompt_spectral_resolution=True, plotting=True)
