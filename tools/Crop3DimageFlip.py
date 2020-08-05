import os, sys
import postTomo as pt

import numpy as np
from skimage import io
from scipy import ndimage

__author__ = 'Vitaliy Starchenko'


class Crop3DimageFlip(pt.AbstractBaseTool):
    '''
    This tool crops the X-Ray 3D image in three directions.
    NOTE! at BMD13 line at APS data array comes as [Z][Y][X]
    Additionally, the array will be flipped along Z
    '''
    
    def __init__(self):
        self.__toolName__ = 'crop3DtomoFlip'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['inputDir', 'beforeCrop', 'directory with initial files']
        )
        self.parameters.append(
            ['outputDir', 'afterCrop', 'final directory with files '
                                             'with corrected tilt']
        )
        self.parameters.append(
            ['xmin', '0', 'min index of 3D array']
        )
        self.parameters.append(
            ['ymin', '0', 'min index of 3D array']
        )
        self.parameters.append(
            ['zmin', '0', 'min index of 3D array']
        )
        self.parameters.append(
            ['xmax', '1000', 'max index of 3D array']
        )
        self.parameters.append(
            ['ymax', '1000', 'max index of 3D array']
        )
        self.parameters.append(
            ['zmax', '1000', 'max index of 3D array']
        )

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__))

        lines = self.read_dict(dictFileName)
        empty, inputDir, description = self.check_a_parameter('inputDir', lines)
        empty, outputDir, description = self.check_a_parameter('outputDir', lines)
        empty, xmin, description = self.check_a_parameter('xmin', lines)
        empty, ymin, description = self.check_a_parameter('ymin', lines)
        empty, zmin, description = self.check_a_parameter('zmin', lines)
        empty, xmax, description = self.check_a_parameter('xmax', lines)
        empty, ymax, description = self.check_a_parameter('ymax', lines)
        empty, zmax, description = self.check_a_parameter('zmax', lines)

        xmin = int(xmin); xmax = int(xmax)
        ymin = int(ymin); ymax = int(ymax)
        zmin = int(zmin); zmax = int(zmax)
        os.makedirs(outputDir)

        tif_files = sorted([f for f in os.listdir(inputDir)
                            if (os.path.isfile(
                os.path.join(inputDir, f)) and ".tif" in f)])

        for j, filename in enumerate(tif_files):
            print('\n*********************************************')
            print('  Processing file {}'.format(filename))
            print('*********************************************')
            
            filen = os.path.join(inputDir, filename)
            stack_tif = io.imread(filen, plugin='tifffile')

            if zmax == -1: zmax = stack_tif.shape[0]+1
            if ymax == -1: ymax = stack_tif.shape[1]+1
            if xmax == -1: xmax = stack_tif.shape[2]+1

            stack_tif = stack_tif[zmin:zmax,ymin:ymax,xmin:xmax]

            stack_tif = np.flip(stack_tif, axis=0)
    
            resultFilename = os.path.splitext(filename)[0]
            savef = os.path.join(outputDir, '{}_crp.tif'.
                                 format(resultFilename))
            io.imsave(savef, stack_tif, plugin='tifffile')

        return True
