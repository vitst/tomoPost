import os, sys
import postTomo as pt

import numpy as np

from skimage import io
from scipy import ndimage

__author__ = 'Vitaliy Starchenko'


class SumIniCryst(pt.AbstractBaseTool):
    '''
    This tool adds crystal voxels to initial geometry
    '''
    
    def __init__(self):
        self.__toolName__ = 'SumIniCryst'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['iniDir',
             'initial',
             ':directory with initial pore structure']
        )
        self.parameters.append(
            ['crystalDir',
             'time_crystals',
             ':Directory which contains segmented images '
             'where crystals are 0, the rest is 1']
        )
        self.parameters.append(
            ['outputDir',
             'solid',
             ':final directory with files with solid - 1,'
                                             ' where solid are cryst and ini']
        )
        self.parameters.append(
            ['outputCryst',
             'cryst',
             ':final directory with files with crystals - 255, rest - 0']
        )

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__))

        lines = self.read_dict(dictFileName)
        empty, inputDir, description = self.check_a_parameter('iniDir', lines)
        empty, crystDir, description = \
            self.check_a_parameter('crystalDir', lines)
        empty, outputDir, description = \
            self.check_a_parameter('outputDir', lines)
        empty, outputCryst, description = \
            self.check_a_parameter('outputCryst', lines)

        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)

        if not os.path.isdir(outputCryst):
            os.makedirs(outputCryst)
            
        ini_tif_files = sorted([f for f in os.listdir(inputDir)
                            if (os.path.isfile(
                os.path.join(inputDir, f)) and ".tif" in f)])
        
        # 255 - glass, 0 - pore
        im0 = io.imread(os.path.join(inputDir, ini_tif_files[0]),
                        plugin='tifffile')

        tmp = os.path.splitext(ini_tif_files[0])[0].split('_')
        resultFilename = "{}_{}_{}.tif".format(tmp[0], tmp[1], tmp[2])
        savef = os.path.join(outputDir, resultFilename)
        io.imsave(savef, im0[4:-4,:,:], plugin='tifffile')

        tif_files = sorted([f for f in os.listdir(crystDir)
                            if (os.path.isfile(
                os.path.join(crystDir, f)) and ".tif" in f)])

        for j, filename in enumerate(tif_files):
            print('\n*********************************************')
            print('  Processing file {}'.format(filename))
            print('*********************************************')
            
            filen = os.path.join(crystDir, filename)
            # 255 - pore and glass, 0 - crystal
            stack_tif = io.imread(filen, plugin='tifffile')
            
            stack_tif = stack_tif.astype(bool)
            stack_tif = np.logical_not(stack_tif).astype(np.uint8)
            stack_tif *= 255

            tmp = os.path.splitext(filename)[0].split('_')
            resFilenameCr = "{}_{}_{}_cryst.tif".format(tmp[0], tmp[1], tmp[2])
            savef = os.path.join(outputCryst, resFilenameCr)
            io.imsave(savef, stack_tif, plugin='tifffile')
            
            # !!! initial image is 8 voxels larger in Z direction
            # !!! this is the result of TrjCor, see phase9()
            stack_tif[im0[4:-4,:,:] != 0] = 255
    
            resultFilename = "{}_{}_{}.tif".format(tmp[0], tmp[1], tmp[2])
            savef = os.path.join(outputDir, resultFilename)
            io.imsave(savef, stack_tif, plugin='tifffile')

        return True
