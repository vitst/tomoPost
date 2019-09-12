import os, sys
import postTomo as pt

import numpy as np
from skimage import io
from scipy.ndimage import measurements
from scipy import ndimage

__author__ = 'Vitaliy Starchenko'


class CleanNotAttached(pt.AbstractBaseTool):
    '''
    This tool reads a binary tiff tomography file 0 or 255
    255 - solid, 0 - pore
    converts 1 - solid, 0 - pore
    Adds solid (1) 1 pixel thick rim around and removes all clusters of solid
    that are not attached to the largest cluster
    '''
    
    def __init__(self):
        self.__toolName__ = 'cleanNotAttached'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['inputFile', 'input.tif', 'name of an input file']
        )
        self.parameters.append(
            ['outputFile', 'output.tif', 'name of an output file']
        )
        
    def clean_clusters(self, bin_image):
        sh = bin_image.shape
        sh = np.asarray(sh)
        sh = sh + 2
    
        aux = np.ones(shape=(sh))
        aux[1:-1, 1:-1, 1:-1] = bin_image
    
        lw, num = measurements.label(aux)
    
        # get a label of the biggest cluster
        minLab = np.min(lw)
        maxLab = np.max(lw)
        print("labels:  min: {}   max: {}".format(minLab, maxLab), flush=True)
    
        hist = measurements.histogram(lw, minLab + 1, maxLab, maxLab - minLab)
    
        # maxCl = np.max(hist)
        maxClLab = np.argmax(hist) + 1
        print("label of a biggest cluster: {}".format(maxClLab), flush=True)
        aux[lw != maxClLab] = 0
        return (aux[1:-1, 1:-1, 1:-1]).astype(np.uint8)

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__))

        lines = self.read_dict(dictFileName)
        empty, inputF, description = self.check_a_parameter('inputFile', lines)
        empty, outputF, description = self.check_a_parameter('outputFile', lines)
        
        stack_tif = io.imread(inputF, plugin='tifffile')
        # make binary 1/0
        print("Clean not attached solids")
        stack_tif = stack_tif.astype(bool).astype(np.int16)
        stack_tif = self.clean_clusters(stack_tif)

        print("Clean not attached holes")
        # invert
        stack_tif = np.logical_not(stack_tif.astype(bool)).astype(np.int16)
        stack_tif = self.clean_clusters(stack_tif)
        
        # invert back
        stack_tif = np.logical_not(stack_tif.astype(bool)).astype(np.int16)

        stack_tif *= 255  # 65535 #256
        io.imsave(outputF, stack_tif, plugin='tifffile')

        return True
