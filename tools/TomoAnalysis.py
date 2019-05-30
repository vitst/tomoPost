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
             'initial',
             'a directory with initial file - the segmented tomo image,'
             'when there is no precipitation. 0 - pore space, 1 - '
             'solid material']
        )
        
        self.parameters.append(
            ['poreDir',
             'pores',
             'directory with TIFF files where the values are '
             'the size of the pores']
        )
        
        self.parameters.append(
            ['crystalDir',
             'time_crystals',
             'the directory which contains segmented images '
             'where crystals are 1, the rest is 0']
        )
        
        self.parameters.append(
            ['outputDir',
             'resultDir',
             'directory where the final files will be placed']
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

    def plot_nucleation(self, im0_bin, tot_cryst_bin, prev_cryst_bin, add_cryst_bin):
        slice_size = 10
        NN = int(im0_bin.shape[0] / slice_size) + 1
        
        newCrystOnGlass = self.N12(add_cryst_bin, im0_bin, self.KER)
        newCrystOnGlass[newCrystOnGlass == 6] = 0
        newCrystOnGlass = newCrystOnGlass.astype(bool).astype(np.int8)
        
        newCrystOnCryst = np.subtract(add_cryst_bin, newCrystOnGlass)
        newCrystOnCryst[newCrystOnCryst == -1] = 0
        newCrystOnCryst = newCrystOnCryst.astype(bool).astype(np.int8)

        xarr = []
        yarr = []
        for ii in range(NN):
            bottom = ii * slice_size
            top = min(im0_bin.shape[0], (ii + 1) * slice_size)
        
            if top <= bottom:
                break
        
            # pore = np.sum(im0_bin[bottom:top, :, :])
            #surf = np.sum(KGG[bottom:top, :, :])
            nCG = np.sum(newCrystOnGlass[bottom:top, :, :])
            nCC = np.sum(newCrystOnCryst[bottom:top, :, :])
        
            # in um
            hight = (im0_bin.shape[0] - bottom - slice_size / 2) * self.SCALE
        
            xarr.append(hight)
            yarr.append( float(nCC) )
    
        xarr = np.asarray(xarr)
        yarr = np.asarray(yarr)
    
        return xarr[1:-1], yarr[1:-1]

    def plot_clusters(self, im0_bin, tot_cryst_bin, prev_cryst_bin,
                        add_cryst_bin):
        slice_size = 50
        NN = int(im0_bin.shape[0] / slice_size) + 1

        lw, num = measurements.label(tot_cryst_bin)
        res = np.multiply(lw, prev_cryst_bin)

        xarr = []
        yarr = []
        for ii in range(NN):
            bottom = ii * slice_size
            top = min(im0_bin.shape[0], (ii + 1) * slice_size)
        
            if top <= bottom:
                break
        
            # pore = np.sum(im0_bin[bottom:top, :, :])
            # surf = np.sum(KGG[bottom:top, :, :])
            #lw, num = measurements.label(tot_cryst_bin[bottom:top, :, :])
            elem_tot = np.unique(lw[bottom:top, :, :])
            elem = np.unique(res[bottom:top, :, :])
            
            n_new_cl = elem_tot.shape[0] - elem.shape[0]

            # in um
            hight = (im0_bin.shape[0] - bottom - slice_size / 2) * self.SCALE
        
            xarr.append(hight)
            #yarr.append(float(elem.shape[0]-1))
            yarr.append(float(n_new_cl))
    
        xarr = np.asarray(xarr)
        yarr = np.asarray(yarr)
    
        return xarr, yarr
        #return xarr[:-1], yarr[1:-1]

    def plot_Ncryst_vsPoreSize(self, im_bin_cryst, im_pore):
        #minPore = self.SCALE * np.min(im_pore)
        #maxPore = self.SCALE * np.max(im_pore)
        minPore = self.SCALE * np.min(im_pore[np.nonzero(im_pore)])
        maxPore = self.SCALE * np.max(im_pore[np.nonzero(im_pore)])
        num_bin = 200
        
        hist_pores, bin_edges_pores =\
            np.histogram(im_pore, bins=num_bin, range=(minPore, maxPore))
        
        CP = np.multiply(im_bin_cryst, im_pore)
        #CP = im_pore

        hist, bin_edges = np.histogram(CP, bins=num_bin, range=(minPore, maxPore))
        
        be = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2.0
        
        ret_hist = np.divide(hist, hist_pores)
        
        return be, ret_hist
        
        

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__), flush=True)
        
        print("Start time {}\n".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
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
        ini_dir_files = sorted([f for f in os.listdir(initialDir)
                             if os.path.isfile(os.path.join(initialDir, f))
                                and '.tif' in f])
        img0 = io.imread(os.path.join(initialDir,ini_dir_files[0]),
                         plugin='tifffile')
        
        im0_bin = img0.astype(bool).astype(np.int8)
        
        pore_volume = im0_bin.shape[0] * im0_bin.shape[1] * im0_bin.shape[2] \
                      - np.sum(im0_bin)
        
        print("Pore volume: {}".format(pore_volume))

        # get data file names
        crystal_files = sorted([f for f in os.listdir(crystalDir)
                             if os.path.isfile(os.path.join(crystalDir, f))])
        
        # get file names from pore dir
        pore_files = sorted([f for f in os.listdir(poreDir)
                             if os.path.isfile(os.path.join(poreDir, f))])

        KGG = self.N11(im0_bin, self.KER)
        KGG[KGG == 6] = 0
        # only the surface pixels are not 0
        KGG = KGG.astype(bool).astype(np.int8)
        # number of surface voxels
        Nsv = np.sum(KGG)
        print("Number of surface voxels: {}".format(Nsv))
        print("ini file time: {} min".
              format( self.extract_time_from_filename(ini_dir_files[0]) ))
        
        colors = ['g-', 'b-', 'r-', 'k-']
        colors1 = ['go', 'bo', 'ro', 'ko']
        
        solid_bin = deepcopy(im0_bin)
        previous_crystals = np.zeros(im0_bin.shape)
        ini_time = 0
        for i, file in enumerate(crystal_files):
            
            file_path = os.path.join(crystalDir, file)
            img = io.imread(file_path, plugin='tifffile')
            # crystals are 1, rest - 0
            im_bin = img.astype(bool).astype(np.int8)
            
            # read pore files
            pore_path = os.path.join(poreDir, pore_files[i])
            pore_img = io.imread(pore_path, plugin='tifffile')
            
            # number of crystal voxels
            nCryst = np.sum(im_bin)

            fnm = file.split("_")
            time = 0
            for ww in fnm:
                if "min" in ww:
                    time = int(ww[:-3])

            print("time: {} min  im0 shape: {}; current im shape:{}".
                  format(time, im0_bin.shape, im_bin.shape))

            #axX, axY = self.plot_nCryst_per_surf(KGG, im_bin)
            #axX, axY = self.plot_Ncryst_vsPoreSize(im_bin, pore_img)
            previous_crystals = np.multiply(previous_crystals, im_bin)
            add_cryst = np.subtract(im_bin, previous_crystals)
            add_cryst[add_cryst < 0] = 0
            #axX, axY = self.plot_nucleation(im0_bin, im_bin,
            #                                previous_crystals, add_cryst)
            axX, axY = self.plot_clusters(im0_bin, im_bin,
                                            previous_crystals, add_cryst)
            previous_crystals = im_bin
            
            dtime = time - ini_time
            axY = axY / dtime
            ini_time = time

            YY1 = np.polyfit(axX, axY, 3)
            pf = np.poly1d(YY1)

            pfX = np.linspace(axX[0], axX[-1], 100)

            #plt.plot(pfX, pf(pfX), colors[i])

            plt.plot(axX, axY, colors1[i])
            
            solid_bin = np.add(im0_bin, im_bin)
            
            
        #plt.xscale('log')
        #plt.yscale('log')
        plt.tight_layout()
        plt.show()
        #res_file_name = "Ncr_perSurfperdt_vsH.png"
        res_file_name = "Ncr_perPoreVol_vsSize.png"
        #res_file_name = "Surf_vsH.png"
        #res_file_name = "histPoreSize.png"
        #plt.savefig(res_file_name, format='png', dpi=300)
        #plt.savefig(res_file_name, format='pdf', dpi=300)


        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
