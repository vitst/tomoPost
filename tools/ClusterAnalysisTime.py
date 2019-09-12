import postTomo as pt

import os, sys
import numpy as np
import multiprocessing as mp

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


class ClusterAnalysisTime(pt.AbstractBaseTool):
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
    
    This tool is designed to plot global variables averaged or calculated for
    the whole sample versus time
    
    '''

    # a kernel variable to utilice convolution operation for faster processing
    KER = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
    # length scale of voxels, uniform in X, Y and Z
    SCALE = 1.24
    
    def __init__(self):
        self.__toolName__ = 'ClusterAnalysisTime'

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
    def calc_surface_area(self, solidbin):
        KGG = self.N11(solidbin, self.KER)
        KGG[KGG == 6] = 0
        # now only the surface pixels are not 0
        
        # the number from 0 - 5 is a nearest 1 neighbors count. Next line will
        # convert it to the number of nearest 0 neighbors, which approximately
        # represents the surface area
        KGG = 6 - KGG
        
        surfArea = np.sum(KGG)
        
        return surfArea

    def N_clusters(self, tot_cryst_bin, prev_cryst_bin, Nprev, cutoff):
    
        lw, num = measurements.label(tot_cryst_bin)
        
        #res = np.multiply(lw, prev_cryst_bin)
        #elem_tot = np.unique(lw)
        #elem = np.unique(res)
    
        #n_new_cl = elem_tot.shape[0] - elem.shape[0]
    
        #yarr = float(elem_tot.shape[0])

        minLab = np.min(lw)
        maxLab = np.max(lw)
        print("labels:  min: {}   max: {}".format(minLab, maxLab), flush=True)
        hist = measurements.histogram(lw, minLab + 1, maxLab, maxLab - minLab)
        
        #ss = np.sum(hist)
        
        # remove clusters below 3 from count
        hist[ hist < cutoff ] = 0
        hist[ hist !=0 ] = 1

        sum_hist = np.sum(hist[np.nonzero(hist)])
        
        #print("hist: {}  num: {}".format(ss, num))
        #exit(0)

        Ntotal = float(sum_hist)

        NoldMerged = 0
        if np.max(prev_cryst_bin)>0:
            # remove all new crystal pixels
            no_new = np.multiply(lw, prev_cryst_bin)
            # remove all new crystal pixels
            no_new_hist = measurements.histogram(no_new,
                                                 minLab + 1,
                                                 maxLab,
                                                 maxLab - minLab)
            
            no_new_hist[ no_new_hist !=0 ] = 1
            NoldMerged = np.sum(no_new_hist[np.nonzero(no_new_hist)])
            
        N_plus = Ntotal - NoldMerged

        N_minus = Nprev - NoldMerged
        
        print("tot: {}   prev: {}   merged: {}".
              format(Ntotal, Nprev, NoldMerged), flush=True)

        return Ntotal, N_plus, N_minus
    
    def proc_files_in_loop(self):
        pass

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
        Nsurf = self.calc_surface_area(im0bin)
        
        print("Surface area in A0: {}".format(Nsurf))
        print("ini file time: {} min".
              format( self.extract_time_from_filename(iniDirFiles[0]) ))
        
        # the current solid material (initial beads with crystals)
        #solidBin = deepcopy(im0bin)
        # auxiliary varialbe to calc increment in crystal voxels (initially - 0)
        previousCrystals = np.zeros(im0bin.shape)
        iniTime = 0
        Nprev=0
        
        axisX = np.empty(0)
        axisY = np.empty(0)
        axisY1 = np.empty(0)
        axisY2 = np.empty(0)

        #axisX = np.append(axisX, iniTime)
        #axisY = np.append(axisY, Nsurf)
        
        # initialize parallel
        pool = mp.Pool(int(mp.cpu_count()/2))
        #results = [pool.apply(howmany_within_range, args=(row, 4, 8))
        #           for row in data]
        
        for i, file in enumerate(crystalFiles):
            # read crystal files (crystals at current moment are 255, rest - 0)
            filePath = os.path.join(crystalDir, file)
            img = io.imread(filePath, plugin='tifffile')
            # convert img to binary: crystals - 1, rest - 0
            imBin = img.astype(bool).astype(np.int8)
            # apdate the solid part
            #solidBin = np.add(im0bin, imBin)

            # read pore files (0 - solid, the rest is in float numbers
            # - the size of the pore in voxels)
            #porePath = os.path.join(poreDir, poreFiles[i])
            #poreImg = io.imread(porePath, plugin='tifffile')
            
            ########################
            # calculation of additional volume information
            previousCrystals = np.multiply(previousCrystals, imBin)
            # new crystal voxels for this time
            #addCryst = np.subtract(imBin, previousCrystals)
            #addCryst[addCryst < 0] = 0
            ########################
            
            fnm = file.split("_")
            time = 0
            for ww in fnm:
                if "min" in ww:
                    time = int(ww[:-3])

            #currentSurf = self.calc_surface_area(solidBin)
            #yVal = np.sum(imBin) # total amount of crystal voxels
            # total number of clusters
            cutoff = 10
            yVal, N_plus, N_minus \
                = self.N_clusters(imBin, previousCrystals, Nprev, cutoff)
            Nprev = yVal
            print("time: {} min  val:  {}".format(time, yVal))

            previousCrystals = imBin
            
            dtime = time - iniTime
            iniTime = time
            
            axisX = np.append(axisX, time)
            axisY = np.append(axisY, yVal)
            axisY1 = np.append(axisY1, N_plus)
            axisY2 = np.append(axisY2, N_minus)


        plt.plot(axisX, axisY, "ko")
        #plt.plot(axisX, axisY1, "ro")
        #plt.plot(axisX, axisY2, "bo")

        #plt.xscale('log')
        #plt.yscale('log')
        plt.tight_layout()
        plt.show()
        
        # file names
        res_file_name = "totalNClustPlusMinus.png"
        res_file_path = os.path.join(outputDir,res_file_name)
        #plt.savefig(res_file_path, format='png', dpi=300)
        #plt.savefig(res_file_path, format='pdf', dpi=300)

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
