import postTomo as pt

import os, sys
import numpy as np

import re
import math
import subprocess
from numpy import linalg
from time import localtime, strftime
from copy import copy, deepcopy

from shutil import copy2, rmtree

from scipy import ndimage
from scipy.ndimage import measurements, morphology, convolve
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes

from skimage import io

import matplotlib.pylab as plt
from matplotlib import rc


__author__ = 'Vitaliy Starchenko'


class ClusterAnalysisHight(pt.AbstractBaseTool):
    '''
    Cluster analysis of the crystal growth tomography data
    
    It is expected the next ordering in the current directory:
    1) initial geometry: 1 - glass beads; 0 - pore space
    ./initial/
    2) distribution of pore sizes in each voxel
    ./pores/
    3) crystal positions: 1 - crystals; 0 - the rest
    ./time_crystals/
    4) directory for result files
    ./resultDir/
    
    This tool is designed to plot global variables averaged or calculated along
    the direction of the flow. All plots are versus the hight of the sample.
    
    '''

    # a kernel variable to utilice convolution operation for faster processing
    KER = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
    # length scale of voxels, uniform in X, Y and Z
    SCALE = 1.24
    
    def __init__(self):
        self.__toolName__ = 'ClusterAnalysisHight'

        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description]
        self.parameters.append(
            ['initialDir',
             'initial',
             ':Directory with initial file - the segmented tomo image,'
             'when there is no precipitation. 0 - pore space, 1 - '
             'solid material']
        )
        
        self.parameters.append(
            ['poreDir',
             'pores',
             ':Directory with TIFF files where the values are '
             'the size of the pores']
        )
        
        self.parameters.append(
            ['crystalDir',
             'time_crystals',
             ':Directory which contains segmented images '
             'where crystals are 1, the rest is 0']
        )
        
        self.parameters.append(
            ['outputDir',
             'resultDir',
             ':Directory where the final files will be placed']
        )
        

        '''
        Define initial kernel which will be used to calculate
        a number of neares neighbours (NN). In 3D we have 6 NN,
        but the area around each pixel is 26 pixels.
        '''
        self.KER[1, 1, 0] = 1
        self.KER[1, 0, 1] = 1
        self.KER[0, 1, 1] = 1
        self.KER[1, 1, 2] = 1
        self.KER[1, 2, 1] = 1
        self.KER[2, 1, 1] = 1



    # return (K*A)oA
    # it returns the number of near neighbours for each solid voxel
    def N11(self, bin_image, KER):
        conv = convolve(bin_image, KER, mode='constant')
        conv = np.multiply(conv, bin_image)
        return conv

    # returns (K*B)oA
    # it returns number of particle voxels neighboring crystal voxels
    def N12(self, cryst_bin_image, bin_zero, KER):
        conv = convolve(bin_zero, KER, mode='constant')
        conv = np.multiply(conv, cryst_bin_image)
        return conv
    
    def extract_time_from_filename(self, filename):
        exp_time=0
        split_name = os.path.splitext(filename)[0].split('_')
        for ss in split_name:
            if 'min' in ss:
                exp_time=int(ss[:-3])
                break
        
        return exp_time
    
    # calculate surface area between 0 and 1 in surface units
    # which is SCALE^2
    # it will count number of faces between voxels 1 and voxels 0
    def calc_surface_area_vs_h(self, solidbin):
        sliceSize = 10
        NN = int(solidbin.shape[0] / sliceSize) + 1
        KGG = self.N11(solidbin, self.KER)
        KGG[KGG == 6] = 0
        # now only the surface pixels are not 0
        
        # the number from 0 - 5 is a nearest 1 neighbors count. Next line will
        # convert it to the number of nearest 0 neighbors, which approximately
        # represents the surface area
        KGG = 6 - KGG
        
        xarr = np.empty(0)
        yarr = np.empty(0)
        for ii in range(NN):
            bottom = ii * sliceSize
            top = min(solidbin.shape[0], (ii + 1) * sliceSize)
            if top <= bottom:
                break
            # in um
            hight = (solidbin.shape[0] - bottom - sliceSize / 2) * self.SCALE

            surfArea = np.sum(KGG[bottom:top, :, :])
    
            xarr = np.append(xarr, hight)
            yarr = np.append(yarr, surfArea)

        return xarr[1:-1], yarr[1:-1]

    def ncryst_vs_h(self, cryst):
        sliceSize = 10
        NN = int(cryst.shape[0] / sliceSize) + 1
    
        xarr = np.empty(0)
        yarr = np.empty(0)
        for ii in range(NN):
            bottom = ii * sliceSize
            top = min(cryst.shape[0], (ii + 1) * sliceSize)
            if top <= bottom:
                break
            # in um
            hight = (cryst.shape[0] - bottom - sliceSize / 2) * self.SCALE

            Ncryst = np.sum(cryst[bottom:top, :, :])
        
            xarr = np.append(xarr, hight)
            yarr = np.append(yarr, Ncryst)
    
        return xarr[1:-1], yarr[1:-1]

    def ncrystPerSurf_vs_h(self, solidbin, cryst):
        sliceSize = 10
        NN = int(solidbin.shape[0] / sliceSize) + 1
        KGG = self.N11(solidbin, self.KER)
        KGG[KGG == 6] = 0
        # now only the surface pixels are not 0
    
        # the number from 0 - 5 is a nearest 1 neighbors count. Next line will
        # convert it to the number of nearest 0 neighbors, which approximately
        # represents the surface area
        KGG = 6 - KGG
    
        xarr = np.empty(0)
        yarr = np.empty(0)
        for ii in range(NN):
            bottom = ii * sliceSize
            top = min(solidbin.shape[0], (ii + 1) * sliceSize)
            if top <= bottom:
                break
            # in um
            hight = (solidbin.shape[0] - bottom - sliceSize / 2) * self.SCALE
        
            surfArea = np.sum(KGG[bottom:top, :, :])
            Ncryst = np.sum(cryst[bottom:top, :, :])

            xarr = np.append(xarr, hight)
            yarr = np.append(yarr, float(Ncryst)/float(surfArea))
    
        return xarr[1:-1], yarr[1:-1]

    def mean_pore_size_vs_h(self, pores):
        sliceSize = 10
        NN = int(pores.shape[0] / sliceSize) + 1
        
        minPore = np.min(pores[np.nonzero(pores)])
        maxPore = np.max(pores[np.nonzero(pores)])
        numBin = 100

        xarr = np.empty(0)
        yarr = np.empty(0)
        for ii in range(NN):
            bottom = ii * sliceSize
            top = min(pores.shape[0], (ii + 1) * sliceSize)
            if top <= bottom:
                break
            # in um
            hight = (pores.shape[0] - bottom - sliceSize / 2) * self.SCALE
        
            histPores, binEdgesPores = \
                np.histogram(pores[bottom:top, :, :],
                             bins=numBin,
                             range=(minPore, maxPore))

            be = binEdgesPores[:-1] + (binEdgesPores[1] - binEdgesPores[0])/2.0
            
            mean = 0
            norm = 0
            for jj, val in enumerate(histPores):
                mean = mean + histPores[jj] / np.power(be[jj], 2)
                norm = norm + histPores[jj] / np.power(be[jj], 3)

            mean = mean / norm

            xarr = np.append(xarr, hight)
            yarr = np.append(yarr, mean)
    
        return xarr[1:-1], yarr[1:-1]

    def execute(self, dictFileName):
        # variables for plotting
        colorsFit = ['g-', 'b-', 'r-', 'k-']
        colorsExperimental = ['go', 'bo', 'ro', 'ko']
        
        print('Starting {0:s} tool'.format(self.__toolName__), flush=True)
        print("Start time {}\n".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        # reading parameters from dictionary
        lines = self.read_dict(dictFileName)
        empty, initialDir, description = \
            self.check_a_parameter('initialDir', lines)
        empty, poreDir, description = \
            self.check_a_parameter('poreDir', lines)
        empty, crystalDir, description = \
            self.check_a_parameter('crystalDir', lines)
        empty, outputDir, description = \
            self.check_a_parameter('outputDir', lines)
        
        # there should be only one file in initialDir
        iniDirFiles = sorted([f for f in os.listdir(initialDir)
                             if os.path.isfile(os.path.join(initialDir, f))
                                and '.tif' in f])
        # read initial geometry (255 - solid glass beads, 0 - pore space)
        img0 = io.imread(os.path.join(initialDir, iniDirFiles[0]),
                         plugin='tifffile')
        # convert to binary (1 - solid glass beads, 0 - pore space)
        im0bin = img0.astype(bool).astype(np.int8)
        
        # total pore volume in voxels
        totPoreVol = im0bin.shape[0] * im0bin.shape[1] * im0bin.shape[2] \
                      - np.sum(im0bin)
        
        print("Pore volume: {}".format(totPoreVol))

        # get data file names
        crystalFiles = sorted([f for f in os.listdir(crystalDir)
                             if os.path.isfile(os.path.join(crystalDir, f))])
        
        # get file names from pore dir
        poreFiles = sorted([f for f in os.listdir(poreDir)
                             if os.path.isfile(os.path.join(poreDir, f))])
        
        # calculate a surface area
        #axisX, axisY = self.calc_surface_area_vs_h(im0bin)
        
        ############## specioal for pore vs h in time ##########################
        #porePath = os.path.join(poreDir, poreFiles[0])
        #poreImg = io.imread(porePath, plugin='tifffile')
        #axisX, axisY = self.mean_pore_size_vs_h(poreImg)
        ########################################################################
        #plt.plot(axisX, axisY, 'mo', mfc='none')

        print("ini file time: {} min".
              format( self.extract_time_from_filename(iniDirFiles[0]) ))
        
        # the current solid material (initial beads with crystals)
        solidBin = deepcopy(im0bin)
        # auxiliary varialbe to calc increment in crystal voxels (initially - 0)
        previousCrystals = np.zeros(im0bin.shape)
        iniTime = 0

        for i, file in enumerate(crystalFiles):
            # read crystal files (crystals at current moment are 255, rest - 0)
            filePath = os.path.join(crystalDir, file)
            img = io.imread(filePath, plugin='tifffile')
            # convert img to binary: crystals - 1, rest - 0
            imBin = img.astype(bool).astype(np.int8)
            
            # read pore files (0 - solid, the rest is in float numbers
            # - the size of the pore in voxels)
            porePath = os.path.join(poreDir, poreFiles[i])
            poreImg = io.imread(porePath, plugin='tifffile')


            ########################
            # calculation of additional volume information
            previousCrystals = np.multiply(previousCrystals, imBin)
            # new crystal voxels for this time
            addCryst = np.subtract(imBin, previousCrystals)
            addCryst[addCryst < 0] = 0
            ########################
            
            fnm = file.split("_")
            time = 0
            for ww in fnm:
                if "min" in ww:
                    time = int(ww[:-3])

            print("time: {} min   crFile: {};   poreFile: {}".
                  format(time, file, poreFiles[i]))
            
            ####################################################################
            #axisX, axisY = self.calc_surface_area_vs_h(solidBin)
            #axisX, axisY = self.ncryst_vs_h(imBin)
            #axisX, axisY = self.mean_pore_size_vs_h(poreImg)
            axisX, axisY = self.ncrystPerSurf_vs_h(solidBin, imBin)
            ####################################################################
            
            
            # apdate the solid part
            solidBin = np.add(im0bin, imBin)

            previousCrystals = imBin
            
            dtime = time - iniTime
            iniTime = time
            
            plt.plot(axisX, axisY, colorsExperimental[i])

        
        #plt.xscale('log')
        #plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        #plt.show()
        
        # file names
        #res_file_name = "meanPoreSize_vs_h.png"
        
        res_file_name = "nCrystPerSurf_vs_h.png"
        res_file_path = os.path.join(outputDir,res_file_name)
        plt.savefig(res_file_path, format='png', dpi=300)
        #plt.savefig(res_file_path, format='pdf', dpi=300)

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
