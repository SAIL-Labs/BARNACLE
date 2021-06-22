import os
import barnacle


def check_datalist_not_empty(data_list):
    """
    Check if the list of data to process is not empy.

    An AssertionError is raised if the list is empty.

    :param data_list: list of data to check
    :type data_list: list
    :raises AssertionError: raised if :data_list: is empty.

    """
    if len(data_list) == 0:
        raise AssertionError('Data list is empty')


def check_mode_flux_exists(mode_flux):
    """
    Check if the mode of measurement of flux in the outputs is \
    among the existing ones.

    :param mode_flux: mode of extraction of the flux in the data frames
    :type mode_flux: string
    :raises AssertionError: raised if :mode_flux: is not in the list

    """
    mode_flux_list = ['raw', 'fit']
    if mode_flux not in mode_flux_list:
        raise AssertionError(
            'Select mode of flux measurement among:', mode_flux_list)
