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
    
    By default it is expected the next ordering in the current directory:
    1) initial geometry: 255 - glass beads; 0 - pore space
    ./01_zero/
    2) only crystals:  255 - crystals; 0 - the rest
    ./02_crystals/
    3) pore distribution from local thickness tool: float number = pore size
    ./03_pores/
    4) solid and pores: 255 - glass beads and crystals; 0 - pore space
    ./04_solid/
    
    Optional:
    5)  
    ./05_CFDfields
    
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
             '01_zero',
             ':Directory with initial file - the segmented tomo image, '
             'no precipitation. 0 - pore space, 255 - solid material']
        )
        
        self.parameters.append(
            ['crystalDir',
             '02_crystals',
             ':Directory which contains segmented images '
             'where crystals are 255, the rest is 0']
        )

        self.parameters.append(
            ['poreDir',
             '03_pores',
             ':Directory with TIFF files where the values are float numbers '
             'the size of the pores in voxels']
        )
        
        self.parameters.append(
            ['solidDir',
             '04_solid',
             ':Directory with merged crystals and glass: '
             '255 - pore space, 0 - the rest (solid material)']
        )
        
        self.parameters.append(
            ['cfdDir',
             '05_CFDfields',
             ':Directory with CFD fields TIFF files']
        )

        self.parameters.append(
            ['outputDir',
             'res',
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
        KGG = KGG[1: -1, 1: -1, 1: -1]
        # now only the surface pixels are not 0
        # the number from 0 - 5 is a nearest 1 neighbors count. Next line will
        # convert it to the number of nearest 0 neighbors, which approximately
        # represents the surface area
        KGG = 6 - KGG
        KGG[KGG == 6] = 0
        #print("min KGG: {}  max KGG: {}".format(np.min(KGG), np.max(KGG)))

        surfArea = np.sum(KGG)
        
        return surfArea
    
    def filter_size(self, solid_bin, cutoff):
        lw, num = measurements.label(solid_bin)
        minLab = np.min(lw)
        maxLab = np.max(lw)
        print("labels:  min: {}   max: {}".format(minLab, maxLab), flush=True)
        hist = measurements.histogram(lw, minLab + 1, maxLab, maxLab - minLab)
        print("histogram shape: {}".format(hist.shape))
        for i, hi in  enumerate(hist):
            if i % 100000 == 0:
                print("i: {}".format(i))
            if hi < cutoff:
                #print("   hi: {}   i: {}".format(hi, i))
                lw[lw == hi] = 0
            
        lw[lw != 0] = 1
        
        return lw, hist

    def N_clusters(self, cr_bin, prev_cr_bin, Nprev, cutoff):
    
        lw, num = measurements.label(cr_bin)
        
        minLab = np.min(lw)
        maxLab = np.max(lw)
        #print("labels:  min: {}   max: {}".format(minLab, maxLab), flush=True)
        hist = measurements.histogram(lw, minLab + 1, maxLab, maxLab - minLab)
        
        # remove clusters below cutoff from count
        hist[ hist < cutoff ] = 0
        hist[ hist !=0 ] = 1

        sum_hist = np.sum(hist[np.nonzero(hist)])
        
        Ntotal = float(sum_hist)

        NoldMerged = 0
        if np.max(prev_cr_bin)>0:
            # remove all new crystal pixels
            no_new = np.multiply(lw, prev_cr_bin)
            no_new_hist = measurements.histogram(no_new,
                                                 minLab + 1,
                                                 maxLab,
                                                 maxLab - minLab)
            
            no_new_hist[ no_new_hist !=0 ] = 1
            NoldMerged = np.sum(no_new_hist[np.nonzero(no_new_hist)])
            
        #print("  ---NoldMerged:  {}".format(NoldMerged))
        
        N_new = Ntotal - NoldMerged

        N_merged = Nprev - NoldMerged
        
        return Ntotal, N_new, N_merged
    
    def proc_files_in_loop(self):
        pass

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__), flush=True)
        print("Start time {}\n".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        # reading parameters from dictionary
        lines = self.read_dict(dictFileName)
        empty, initialDir, description = \
            self.check_a_parameter('initialDir', lines)
        empty, crystalDir, description = \
            self.check_a_parameter('crystalDir', lines)
        empty, poreDir, description = \
            self.check_a_parameter('poreDir', lines)
        empty, solidDir, description = \
            self.check_a_parameter('solidDir', lines)
        empty, cfdDir, description = \
            self.check_a_parameter('cfdDir', lines)
        empty, outputDir, description = \
            self.check_a_parameter('outputDir', lines)
        
        # variables for plotting
        colorsFit = ['g-', 'b-', 'r-', 'k-']
        colorsExperimental = ['go', 'bo', 'ro', 'ko']

        ###########################
        ##  file reading
        ###########################

        # there should be only one file in initialDir
        iniDirFiles = sorted([f for f in os.listdir(initialDir)
                             if os.path.isfile(os.path.join(initialDir, f))
                                and '.tif' in f])
        # read initial geometry (255 - solid glass beads, 0 - pore space)
        img0 = io.imread(os.path.join(initialDir, iniDirFiles[0]),
                         plugin='tifffile')

        # convert to binary (1 - solid glass beads, 0 - pore space)
        im0bin = img0.astype(bool).astype(np.int8)
        
        print("3D tomo image dimensions: {}".format(im0bin.shape))

        # total pore volume in voxels
        totPoreVol = im0bin.shape[0] * im0bin.shape[1] * im0bin.shape[2] \
                      - np.sum(im0bin)
        
        print("Pore volume: {}".format(totPoreVol))

        # get crystal file names
        crystalFiles = sorted([f for f in os.listdir(crystalDir)
                             if os.path.isfile(os.path.join(crystalDir, f))])
        
        # get file names from pore dir
        poreFiles = sorted([f for f in os.listdir(poreDir)
                             if os.path.isfile(os.path.join(poreDir, f))])
        
        # get file names from solid dir
        solidFiles = sorted([f for f in os.listdir(solidDir)
                             if os.path.isfile(os.path.join(solidDir, f))])

        # get CFD tiff filenames
        cfdFilesCbos = sorted([f for f in os.listdir(cfdDir)
                    if os.path.isfile(os.path.join(solidDir,'C_Bosbach', f))])
        cfdFilesUbos = sorted([f for f in os.listdir(cfdDir)
                    if os.path.isfile(os.path.join(solidDir,'U_Bosbach', f))])

        cfdFilesCzh = sorted([f for f in os.listdir(cfdDir)
                    if os.path.isfile(os.path.join(solidDir,'C_ZhenWu', f))])
        cfdFilesUzh = sorted([f for f in os.listdir(cfdDir)
                    if os.path.isfile(os.path.join(solidDir,'U_ZhenWu', f))])

        ###########################
        ##  calculations
        ###########################

        # calculate surface area in initial sample
        Nsurf = self.calc_surface_area(im0bin)

        print("Surface area in A0: {}".format(Nsurf))
        print("ini file time: {} min".
              format( self.extract_time_from_filename(iniDirFiles[0]) ))
        
        # the current solid material (initial beads with crystals)
        iniTime = 0
        Nprev=0
        
        axisX = np.empty(0)
        axisY = np.empty(0)

        axisX = np.append(axisX, iniTime)
        #axisY = np.append(axisY, Nsurf)
        axisY = np.append(axisY, 0)

        totN  = 0
        
        for i, file in enumerate(solidFiles):
            # read files (solid at current moment are 255, rest - 0)
            filePath = os.path.join(solidDir, file)
            img = io.imread(filePath, plugin='tifffile')
            # convert img to binary: crystals - 1, rest - 0
            solidBin = img.astype(bool).astype(np.int8)

            time = self.extract_time_from_filename(file)
            #print("\ntime: {} minutes\n".format(time))
            axisX = np.append(axisX, time)
            
            #print("Calcularing surface area of the solid")
            currentSurf = self.calc_surface_area(solidBin)
            #axisY = np.append(axisY, currentSurf)
            
            # calculate total number of crystal voxels
            #print("calculate total number of crystal voxels")
            totPoreVol = solidBin.shape[0] * solidBin.shape[1] * solidBin.shape[2] \
                       - np.sum(solidBin)
            
            print("{}    surf: {}  poreVol: {}".
                  format(time, currentSurf, totPoreVol))

        print("Analise crystals")
        totN  = 0
        totN1 = 0
        #xN = 0
        previousCrystals = np.zeros(im0bin.shape)

        for i, file in enumerate(crystalFiles):
            # read crystal files (0 - pore and glass, 255 - crystals)
            filePath = os.path.join(crystalDir, file)
            imgCr = io.imread(filePath, plugin='tifffile')
            # convert img to binary: crystals - 1, rest - 0
            imCrBin = imgCr.astype(bool).astype(np.int8)
            #merge with previous
            imCrBin = np.logical_or(imCrBin, previousCrystals)
            imCrBin = imCrBin.astype(np.int8)


            time = self.extract_time_from_filename(file)
            #print("\ntime: {} minutes\n".format(time))

            totalCryst = np.sum( imCrBin )

            # total number of clusters
            cutoff1 = 3
            totN, N_new, N_merged \
                = self.N_clusters(imCrBin, previousCrystals, totN1, cutoff1)
            axisY = np.append(axisY, totN)

            print("{}    ++  totalClust: {}  Nnew: {}  Nmerged: {}  +++++".
                  format(time, totN, N_new, N_merged))

            #!!axisY1 = np.append(axisY1, N_plus)
            #!!axisY2 = np.append(axisY2, N_minus)
            totN1 = totN
            previousCrystals = deepcopy(imCrBin)

            

        plt.plot(axisX, axisY, "ko")
        #plt.plot(axisX, axisY1, "ro")
        #plt.plot(axisX, axisY2, "bo")

        #plt.xscale('log')
        #plt.yscale('log')
        plt.tight_layout()
        #plt.show()
        
        # file names
        res_file_name = "res.png"
        res_file_path = os.path.join(outputDir,res_file_name)
        plt.savefig(res_file_path, format='png', dpi=300)
        #plt.savefig(res_file_path, format='pdf', dpi=300)

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
