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


class TomoAnalysis(pt.AbstractBaseTool):
    '''
    Cluster analysis of the crystal growth tomography data
    It is expected the next ordering in the current directory:
    1) initial geometry: 1 - glass beads; 0 - pore space
    ./initial/
    2) distribution of pore sizes in each voxel
    ./pores/
    3) crystal positions: 1 - crystals; 0 - the rest
    ./time_crystals/
    
    '''

    # a kernel variable
    KER = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
    SCALE = 1.24
    
    def __init__(self):
        self.__toolName__ = 'tomoAnalysis'

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

    
    def plot_nCryst_per_surf(self, KGG, im_bin):
        slice_size = 10
        NN = int(im_bin.shape[0] / slice_size) + 1
    
        xarr = []
        yarr = []
        for ii in range(NN):
            bottom = ii * slice_size
            top = min(im_bin.shape[0], (ii + 1) * slice_size)
            
            if top <= bottom:
                break
        
            #pore = np.sum(im0_bin[bottom:top, :, :])
            surf = np.sum(KGG[bottom:top, :, :])
            
            #print("bottom: {}  top: {}  surf: {}".format(bottom, top, surf))
            
            nCryst = np.sum(im_bin[bottom:top,:,:])
        
            # in um
            hight = (im_bin.shape[0] - bottom - slice_size / 2) * self.SCALE
        
            xarr.append(hight)
            # axY.append(float(nCryst) / float(pore))
            #yarr.append(float(surf) / float(pore))
            yarr.append(float(nCryst) / float(surf))
            
        xarr = np.asarray(xarr)
        yarr = np.asarray(yarr)
        
        return xarr[1:-1], yarr[1:-1]

    def plot_nCryst_per_surf1(self, solid_bin, add_cryst_bin):
        slice_size = 10
        NN = int(solid_bin.shape[0] / slice_size) + 1
        
        KGG = self.N11(solid_bin, self.KER)
        KGG[KGG == 6] = 0
        # only the surface pixels are not 0
        KGG = KGG.astype(bool).astype(np.int8)
    
        xarr = []
        yarr = []
        for ii in range(NN):
            bottom = ii * slice_size
            top = min(solid_bin.shape[0], (ii + 1) * slice_size)
        
            if top <= bottom:
                break

            # pore = np.sum(im0_bin[bottom:top, :, :])
            surf = np.sum(KGG[bottom:top, :, :])

            # print("bottom: {}  top: {}  surf: {}".format(bottom, top, surf))
        
            nCryst = np.sum(add_cryst_bin[bottom:top, :, :])
        
            # in um
            hight = (solid_bin.shape[0] - bottom - slice_size / 2) * self.SCALE
            
            xarr.append(hight)
            # axY.append(float(nCryst) / float(pore))
            # yarr.append(float(surf) / float(pore))
            yarr.append(float(nCryst) / float(surf))
            #yarr.append( float(surf))

        xarr = np.asarray(xarr)
        yarr = np.asarray(yarr)
    
        return xarr[1:-1], yarr[1:-1]


    def plot_Ncryst_vsPoreSize(self, im_bin_cryst, im_pore):
        #minPore = self.SCALE * np.min(im_pore)
        #maxPore = self.SCALE * np.max(im_pore)
        #minPore = self.SCALE * np.min(im_pore[np.nonzero(im_pore)])
        #maxPore = self.SCALE * np.max(im_pore[np.nonzero(im_pore)])
        minPore = np.min(im_pore[np.nonzero(im_pore)])
        maxPore = np.max(im_pore[np.nonzero(im_pore)])
        num_bin = 200
        
        lw, num = measurements.label(im_bin_cryst)
        minLab = np.min(lw)
        maxLab = np.max(lw)
        print("Number of features: {}".format(num))
        print("labels:  min: {}   max: {}".format(minLab, maxLab), flush=True)
        hist = measurements.histogram(lw, minLab + 1, maxLab, maxLab - minLab)

        #for i1, size in enumerate(hist):
        #    if size < 2: #10
        #        im_bin_cryst[ im_bin_cryst==i1 ] = 0
        #a = [0 if a_ > thresh else a_ for a_ in a]


        #in_list_hist = hist < 10
        #im_bin_cryst1 = [0 if (element in in_list_hist) else element for element in im_bin_cryst]
        #im_bin_cryst[ in_list ] = 0

        #in_list_hist = np.where( np.isin(L,R) )
        # cut out clusters that are smaller than 10
        in_list_hist = np.where( hist < 10 )
        lw1 = np.where( np.isin(lw, in_list_hist[0]), 0, lw)
         
        im_bin_cryst1 = np.where( np.isin(lw, in_list_hist[0]), 0, im_bin_cryst)
        print( "0: min: {}    max: {}".format( np.min(im_bin_cryst), np.max(im_bin_cryst)) )
        print( "1: min: {}    max: {}".format( np.min(im_bin_cryst1), np.max(im_bin_cryst1)) )
        print( "sum 0: {}".format(np.sum(im_bin_cryst)) )
        print( "sum 1: {}".format(np.sum(im_bin_cryst1)) )

        hist_pores, bin_edges_pores = \
            np.histogram(im_pore, bins=num_bin, range=(minPore, maxPore))

        print( "Pore 0: min: {}    max: {}".format( minPore, maxPore) )
        
        CP = np.multiply(im_bin_cryst1, im_pore)
        minPore1 = np.min(im_pore[np.nonzero(CP)])
        maxPore1 = np.max(im_pore[np.nonzero(CP)])
        print( "Pore 1: min: {}    max: {}\n".format( minPore1, maxPore1) )
        print( "sum binCP: {}\n".format(np.sum(CP.astype(bool).astype(np.int8))) )

        tot_cr = np.sum(im_bin_cryst1)
        tot_cr1 = np.sum(CP.astype(bool).astype(np.int8))

        hist, bin_edges = \
            np.histogram(CP, bins=num_bin, range=(minPore, maxPore))


        print( "sum totHist: {}".format(np.sum(hist)) )
        
        be = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2.0
        
        print( tot_cr )
        #exit(0)
        
        ret_hist = hist / tot_cr1
        #ret_hist = np.divide(hist, hist_pores)
        #ret_hist = hist_pores

        print( "sum hist: {}".format(np.sum(ret_hist)) )
        print("*************************************************\n")

        return be, ret_hist
        

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

        if os.path.exists(cfdDir):
            # get CFD tiff filenames
            if os.path.exists(os.path.join(cfdDir,'C_Bosbach')):
                cfdFilesCbos = sorted([f for f in os.listdir(os.path.join(cfdDir,'C_Bosbach'))
                            if os.path.isfile(os.path.join(cfdDir,'C_Bosbach', f))])
                cfdFilesUbos = sorted([f for f in os.listdir(os.path.join(cfdDir,'U_Bosbach'))
                            if os.path.isfile(os.path.join(cfdDir,'U_Bosbach', f))])
    
            cfdFilesCzh = sorted([f for f in os.listdir(os.path.join(cfdDir,'C_ZhenWu'))
                        if os.path.isfile(os.path.join(cfdDir,'C_ZhenWu', f))])
            cfdFilesUzh = sorted([f for f in os.listdir(os.path.join(cfdDir,'U_ZhenWu'))
                        if os.path.isfile(os.path.join(cfdDir,'U_ZhenWu', f))])

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
        
        for i, file_n in enumerate(crystalFiles):
            
            file_path = os.path.join(crystalDir, file_n)
            img = io.imread(file_path, plugin='tifffile')
            # crystals are 255, rest - 0
            cryst_bin = img.astype(bool).astype(np.int8)

            time = self.extract_time_from_filename(file_n)
            print("time: {} minutes".format(time))
            
            # read pore files for the previous time
            # Note: pore file dir have one more ini time file
            pore_path = os.path.join(poreDir, poreFiles[i])
            pore_img = io.imread(pore_path, plugin='tifffile')
            
            axX, axY = self.plot_Ncryst_vsPoreSize(cryst_bin, pore_img)

            plt.plot(axX,axY, colorsExperimental[i], label='t={}'.format(time))

        
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        #plt.show()

        #res_file_name = "histPoreSize.png"
        res_file_name = "Ncr_perPoreVol_vsSize.png"
        res_file_path = os.path.join(outputDir,res_file_name)
        plt.savefig(res_file_path, format='png', dpi=300)
        #plt.savefig(res_file_name, format='pdf', dpi=300)

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
