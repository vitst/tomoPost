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

import numpy as np
import os, sys
import time
import gc

from skimage import io
from skimage import measure

import pymesh

from scipy.ndimage import measurements
from scipy.ndimage import morphology
from scipy.ndimage import convolve
from scipy.ndimage import filters
from scipy.ndimage import binary_erosion


from skimage import io


__author__ = 'Vitaliy Starchenko'


class Tomo2Mesh(pt.AbstractBaseTool):
    '''
    This tool does copy the crystall binary files and combines them with initial
    geometry to create a binary solid porous structure which results binary
    tif files where 255 - solid, 0 - pore
    Files are copied into the directory tif_solid
    '''
    
    def __init__(self):
        self.__toolName__ = 'tomo2mesh'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['zeroDir', 'zeroDir', ':directory with initial geometry file ']
        )
        
        self.parameters.append(
            ['crystDir', 'crystDir', ':directory with segmented crystal files ']
        )
        
        self.parameters.append(
            ['maskDir', 'maskDir', ':directory with a mask']
        )
        
        self.parameters.append(
            ['solidDir', 'solid', ':directory with binary solid-pore']
        )
        
    def clean_not_attached(self, bin_image):
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
        return aux[1:-1, 1:-1, 1:-1]
    
    
    def erode_NN(self, bin_image):
        kernel = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
        kernel[1, 1, 0] = 1
        kernel[1, 0, 1] = 1
        kernel[0, 1, 1] = 1
        kernel[1, 1, 2] = 1
        kernel[1, 2, 1] = 1
        kernel[2, 1, 1] = 1
    
        conv = convolve(bin_image, kernel, mode='constant')
        conv = np.multiply(conv, bin_image)
        conv = conv.astype(np.float)
        # 0.2 for 2 neighbors will give 0.4 and 3 - 0.6
        #conv *= 0.2
        # 0.3 for 1 neighbor will give 0.3 and 2 - 0.6
        conv *= 0.3
        conv = np.around(conv)
        conv = conv.astype(np.int16)
        return (np.logical_and(conv, bin_image)).astype(np.int16)
    
    
    def erode_converge(self, bin_image):
        print("Erode array", flush=True)
        size = bin_image.shape[0] * bin_image.shape[1] * bin_image.shape[2]
        next_size = measurements.sum(bin_image)
        while (next_size < size):
            print("size: {}  next: {}".format(size, next_size), flush=True)
            bin_image = self.erode_NN(bin_image)
            size = next_size
            next_size = measurements.sum(bin_image)
    
        return bin_image

    
    def makeSolidBinary(self, path2ini, path2cryst, path2mask, solid_dir):
        tif_files = sorted([f for f in os.listdir(path2cryst)
            if (os.path.isfile( os.path.join(path2cryst, f)) and ".tif" in f)])
        
        tif_files_from0 = sorted([f for f in os.listdir(path2ini)
            if (os.path.isfile(os.path.join(path2ini, f)) and ".tif" in f)])
        
        time0file = os.path.join(path2ini, tif_files_from0[0])
        stack_tif0 = io.imread(time0file, plugin='tifffile')
        
        # cut tif0 4 from the bottom and top
        stack_tif0 = stack_tif0[4:-4,:,:]
        #stack_tif0 = stack_tif0[:,:,:]
        
        # apply mask to time zero
        mask_tif = sorted([f for f in os.listdir(path2mask)
            if (os.path.isfile(os.path.join(path2mask, f)) and ".tif" in f)])
        path_file_mask = os.path.join(path2mask, mask_tif[0])
        mask = io.imread(path_file_mask, plugin='tifffile')
        im0Sh = stack_tif0.shape
        mSh = mask.shape
        if im0Sh[1:3] != mSh:
            print('Shape of the mask is {} '
                  'and it does not match the shape of the image {}'.
                  format(mSh, im0Sh), flush=True)
            sys.exit(2)
        maxval = np.max(stack_tif0)
        stack_tif0[:, mask == 0] = maxval
        
        # save 0
        savef = os.path.join(solid_dir, tif_files_from0[0])
        io.imsave(savef, stack_tif0, plugin='tifffile')
        
        for i, filename in enumerate(tif_files):
            print('\n*********************************************')
            print('  Processing file {}'.format(filename))
            print('*********************************************')
            filen = os.path.join(path2cryst, filename)
            stack_tif = io.imread(filen, plugin='tifffile')
            stack_tif = np.logical_not(stack_tif.astype(bool)).astype(np.uint8)
        
            if stack_tif0.shape != stack_tif.shape:
                print("Initial geometry file shape {} doesn't match crystall"
                      " size {}".
                       format(stack_tif0.shape, stack_tif.shape), flush=True)
                sys.exit(2)
        
            print("adding initial geometry...")
            stack_tif[stack_tif0 != 0] = 1
            stack_tif *= 255
        
            print("Saving array of size {} as tif...".format(stack_tif.shape))
            filename = os.path.splitext(filename)[0]
            savef = os.path.join(solid_dir, '{}_cryst.tif'.format(filename))
            io.imsave(savef, stack_tif, plugin='tifffile')
            
    def makeMesh(self, path2file, do_clean=False):
        path, file_name = os.path.split(path2file)
        fname, fext = os.path.splitext(file_name)
        
        # read image
        stack_tif = io.imread(path2file, plugin='tifffile')
        
        print('tif shape: {}'.format(stack_tif.shape), flush=True)
        
        bin_stack_tif = stack_tif.astype(bool).astype(np.int16)
        
        # clean from isolated and small pixels (optional)
        if do_clean:
            bin_stack_tif = self.erode_converge(bin_stack_tif)
            bin_stack_tif = self.clean_not_attached(bin_stack_tif)
        
            print('Saving image after cleaning', flush=True)
            io.imsave("aux.tif", (255*bin_stack_tif).astype(np.uint8), plugin='tifffile')
        
        bin_stack_tif = binary_erosion(bin_stack_tif, iterations=1)
        
        # create 1 pixel layer to make close meshes
        NZ = bin_stack_tif.shape[0]
        NY = bin_stack_tif.shape[1]
        NX = bin_stack_tif.shape[2]
        aux = np.zeros((NZ + 10, NY + 10, NX + 10))
        aux[5:NZ + 5, 5:NY + 5, 5:NX + 5] = bin_stack_tif
        
        print('Triangulation of a set of points ...', flush=True)
        
        aux = aux.astype(np.float32)
        
        aux = filters.gaussian_filter(aux, 2.0, truncate=1.0)
        
        maxA = np.max(aux)
        
        aux = aux / maxA * 4.0 - 1.0
        # aux = aux * 4.0 - 1.0
        
        minA = np.min(aux)
        maxA = np.max(aux)
        
        print('Cleaning memory', flush=True)
        del stack_tif
        del bin_stack_tif
        gc.collect()
        
        print("min aux: {}   max aux: {}".format(minA, maxA), flush=True)
        verts, faces, normals, values = measure.marching_cubes_lewiner(aux, 0)
        
        print('Cleaning memory', flush=True)
        del aux
        gc.collect()
        
        print("Mesh data 0:",flush=True)
        print("Verts: {}  Faces: {}".format(verts.shape, faces.shape))
        
        print('Remove isolated vertices', flush=True)
        verts, faces, info = pymesh.remove_isolated_vertices_raw(verts, faces)
        print(info, flush=True)
        
        print("Mesh data 1:",flush=True)
        print("Verts: {}  Faces: {}".format(verts.shape, faces.shape))
        
        print('Remove duplicated vertices', flush=True)
        verts, faces, info = pymesh.remove_duplicated_vertices_raw(verts, faces)
        print(info, flush=True)
        
        print("Mesh data 2:",flush=True)
        print("Verts: {}  Faces: {}".format(verts.shape, faces.shape))
        verts = verts - 5.0
        
        print('Set the mesh ...', flush=True)
        final_mesh = pymesh.form_mesh(verts, faces)
        
        savefilename = "{}.obj".format(fname)
        savefilepath = os.path.join('./', savefilename)
        
        print("saving file {}".format(savefilename), flush=True)
        
        pymesh.save_mesh(savefilepath, final_mesh)
        
        print('done.', flush=True)
        

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__), flush=True)

        print("Start time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)

        lines = self.read_dict(dictFileName)
        empty, zeroDir, description = \
            self.check_a_parameter('zeroDir', lines)
        empty, crystDir, description = \
            self.check_a_parameter('crystDir', lines)
        empty, maskDir, description = \
            self.check_a_parameter('maskDir', lines)
        empty, solidDir, description = \
            self.check_a_parameter('solidDir', lines)
        
        os.makedirs(solidDir)
        
        ########################################################################
        # stage 1 read crystall files, combine with ini geometry and save
        self.makeSolidBinary(zeroDir, crystDir, maskDir, solidDir)
        
        ########################################################################
        # stage 2 makeMesh
        tif_files = sorted([f for f in os.listdir(solidDir)
                            if (os.path.isfile(
                os.path.join(solidDir, f)) and ".tif" in f)])
        for j, filename in enumerate(tif_files):
            print('\n*********************************************', flush=True)
            print('  Processing file {}'.format(filename), flush=True)
            print('*********************************************', flush=True)
            filen = os.path.join(solidDir, filename)

            self.makeMesh(filen)
        
        ########################################################################
        # stage 3 rotate and scale the mesh
        

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
