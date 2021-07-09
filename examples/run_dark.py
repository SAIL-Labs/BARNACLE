"""
This script is an example on how to use the library ``dark.py`` to
obtain the dark for future processing of the data.
"""
from barnacle.calibration.dark import *


''' Settings '''
save = True
monitor = True
nb_files = (None, None)
edges = -500, 500

''' Inputs '''
datafolder = 'data202104/20210429/alfBoo_1/'
data_path = '//tintagel.physics.usyd.edu.au/snert/'+'/GLINTData/'+datafolder
# data_path = '/mnt/96980F95980F72D3/glintData/'+datafolder

''' Output '''
output_path = '//tintagel.physics.usyd.edu.au/snert/GLINTprocessed/'+datafolder
# output_path = '/mnt/96980F95980F72D3/GLINTprocessed/'+datafolder

dk = get_average_dark(
    data_path, output_path, nb_files, save, monitor, edges)
