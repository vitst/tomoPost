import os, sys
import postTomo as pt

from skimage import io
from skimage import transform
from scipy import ndimage

__author__ = 'Vitaliy Starchenko'


class CorrectCapillaryTiltXZ(pt.AbstractBaseTool):
    '''
    This tool corrects the tilt of the capillary in the direction defined
    by XZ plane.
    '''
    
    def __init__(self):
        self.__toolName__ = 'correctTilt'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['inputDir', 'fullDomain', 'directory with initial files']
        )
        self.parameters.append(
            ['outputDir', 'fullDomain_corTilt', 'final directory with files '
                                             'with corrected tilt']
        )
        self.parameters.append(
            ['angle', '0.5', 'the angle to correct the tilt']
        )

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__))

        lines = self.read_dict(dictFileName)
        empty, inputDir, description = self.check_a_parameter('inputDir', lines)
        empty, outputDir, description = self.check_a_parameter('outputDir', lines)
        empty, angle, description = self.check_a_parameter('angle', lines)
        
        angle = float(angle)
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
    
            stack_tif = ndimage.interpolation.rotate(stack_tif, angle,
                            axes=(0, 1), reshape=False, output=None, order=3,
                            mode='constant', cval=0.0, prefilter=True)
    
            resultFilename = os.path.splitext(filename)[0]
            savef = os.path.join(outputDir, '{}_corTilt.tif'.
                                 format(resultFilename))
            io.imsave(savef, stack_tif, plugin='tifffile')

        return True
