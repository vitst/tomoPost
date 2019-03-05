import postTomo as pt

import os, sys
import numpy as np

import re
import math
import subprocess
from numpy import linalg
from time import localtime, strftime

from shutil import copy2, rmtree

from scipy import ndimage
from scipy.ndimage import measurements, morphology, convolve
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes

from skimage import io

import matplotlib.pylab as plt
from matplotlib import rc


__author__ = 'Vitaliy Starchenko'


class TomoAnalysis(pt.AbstractBaseTool):
    '''
    This tool does the two phase segmentation of crystals and reactive fluid.
    '''
    
    def __init__(self):
        self.__toolName__ = 'tomoAnalysis'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['crystalDir', 'trjCor/ini_TR_BP_subtr_cryst',
             'the directory which contains segmented images '
             'where crystals are 0, the rest is 1']
        )
        
        self.parameters.append(
            ['outputDir', 'resultDir', 'directory where the final files '
                                       'will be placed']
        )
        
        self.parameters.append(
            ['time0file',
             'trjCor/ini_TR_BP_cl0_cl/02_run02_0027min_trsmd_filtered_seg.tif',
             'the segmented tomo image at time 0, when there is no'
             ' precipitation. 0 - pore space, 1 - solid material']
        )

    # return (K*A)oA
    def N11(self, bin_image, KER):
        conv = convolve(bin_image, KER, mode='constant')
        conv = np.multiply(conv, bin_image)
        return conv

    # returns (K*B)oA
    def N12(self, bin_image, bin_zero, KER):
        conv = convolve(bin_zero, KER, mode='constant')
        conv = np.multiply(conv, bin_image)
        return conv

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__), flush=True)
        
        print("Start time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        lines = self.read_dict(dictFileName)
        empty, crystalDir, description = \
            self.check_a_parameter('crystalDir', lines)
        empty, outputDir, description = \
            self.check_a_parameter('outputDir', lines)
        empty, time0file, description = \
            self.check_a_parameter('time0file', lines)

        '''
        Define initial kernel which will be used to calculate a number of neares neighbours (NN).
        In 3D we have 6 NN, but the area around each pixel is 27 pixels.
        '''
        KER = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
        KER[1, 1, 0] = 1
        KER[1, 0, 1] = 1
        KER[0, 1, 1] = 1
        KER[1, 1, 2] = 1
        KER[1, 2, 1] = 1
        KER[2, 1, 1] = 1

        # get data file names
        data_files = sorted([f for f in os.listdir(crystalDir)
                             if os.path.isfile(os.path.join(crystalDir, f))])

        img0 = io.imread(time0file, plugin='tifffile')
        im0_bin = np.logical_not(img0.astype(bool)).astype(int)
        # TODO this needs to be adjasted
        im0_bin = im0_bin[1:-1,:,:-1]

        KGG = self.N11(im0_bin, KER)
        KGG[KGG == 6] = 0
        # only the surface pixels are not 0
        KGG = KGG.astype(bool).astype(int)
        #Ngw = np.sum(KGG)
        
        colors = ['g-', 'b-', 'r-', 'k-']
        colors1 = ['go', 'bo', 'ro', 'ko']
        #axX = []
        #axY = []
        for i, file in enumerate(data_files):
            
            file_path = os.path.join(crystalDir, file)
            img = io.imread(file_path, plugin='tifffile')
            # after this line crystals are 1, rest - 0
            im_bin = np.logical_not(img.astype(bool)).astype(int)
            
            print(im0_bin.shape, im_bin.shape)

            all = np.sum(im0_bin)

            nCryst = np.sum(im_bin)

            fnm = file.split("_")

            time = 0
            for ww in fnm:
                if "min" in ww:
                    time = int(ww[:-3])

            #axX.append(time)
            #axY.append(float(nCryst) / float(all))
            
            slice_size = 10
            NN = int(im_bin.shape[0] / slice_size)+1
            
            axX = []
            axY = []
            for ii in range(NN):
                bottom = ii * slice_size
                top = min(img.shape[0]-1, (ii + 1) * slice_size)
                
                pore = np.sum(im0_bin[bottom:top,:,:])
                surf = np.sum(KGG[bottom:top,:,:])
                #nCryst = np.sum(im_bin[bottom:top,:,:])
                
                # in um
                hight = (im_bin.shape[0] - bottom + slice_size/2) * 1.24

                axX.append(hight)
                #axY.append(float(nCryst) / float(pore))
                #axY.append(float(nCryst) / float(surf))
                axY.append(float(surf) / float(pore))


            axX = np.asarray(axX)
            axY = np.asarray(axY)

            YY1 = np.polyfit(axX, axY, 3)
            pf = np.poly1d(YY1)

            pfX = np.linspace(axX[0], axX[-1], 100)

            plt.plot(pfX, pf(pfX), colors[i])

            plt.plot(axX, axY, colors1[i])

        plt.tight_layout()
        #plt.show()
        plt.savefig("surfPerVolVSh.pdf", format='pdf', dpi=1000)

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
