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


class Tomo2MeshTIFFdir(pt.AbstractBaseTool):
    '''
    Usual Tomo2Mesh modified to convert TIF with 1 - solid; 0 - pore
    from a single dir to OBJ. Reads all tif files
    tif files where 255 - solid, 0 - pore
    '''
    
    def __init__(self):
        self.__toolName__ = 'tomo2meshTIFFdir'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        
        self.parameters.append(
            ['sourceDir', 'sourceDir', ':directory with segmented .tif files ']
        )
        
        self.parameters.append(
            ['objDir', 'objDir', ':directory with obj surface mesh']
        )

    def obj_writer(self, fname, verts, faces):

        with open(fname, 'w') as the_file:
            the_file.write('# Generated with my own obj writer\n')
            for vert in verts:
                #the_file.write('v {0:10.3g} {1:10.3g} {2:10.3g}\n'.format(vert[0],vert[1],vert[2]))
                v0 = '{:.3f}'.format(vert[0]).rstrip('0')
                v1 = '{:.3f}'.format(vert[1]).rstrip('0')
                v2 = '{:.3f}'.format(vert[2]).rstrip('0')
                the_file.write('v {} {} {}\n'.format(v0,v1,v2))
            for face in faces:
                the_file.write('f {} {} {}\n'.format(face[0]+1, face[1]+1, face[2]+1))

    def return_min_max(self, vertices):
        min_x = 1000.0
        min_y = 1000.0
        min_z = 1000.0
        max_x = 0.0
        max_y = 0.0
        max_z = 0.0
    
        for coord in vertices:
            x = coord[0]
            y = coord[1]
            z = coord[2]
    
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
    
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
        return min_x,min_y,min_z,max_x,max_y,max_z;

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

    
    def makeMesh(self, path2file):
        path, file_name = os.path.split(path2file)
        fname, fext = os.path.splitext(file_name)
        
        # read image
        stack_tif = io.imread(path2file, plugin='tifffile')
        
        print('tif shape: {}'.format(stack_tif.shape), flush=True)
        
        bin_stack_tif = stack_tif.astype(bool).astype(np.int16)
        
        # create 5 pixel layer to make close meshes
        NZ = bin_stack_tif.shape[0]
        NY = bin_stack_tif.shape[1]
        NX = bin_stack_tif.shape[2]
        aux = np.zeros((NZ + 10, NY + 10, NX + 10))
        aux[5:NZ + 5, 5:NY + 5, 5:NX + 5] = bin_stack_tif

        aux = np.swapaxes(aux, 0, 2)
        aux = np.flip(aux, 1)
        aux = np.flip(aux, 2)

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
        #verts, faces, normals, values = measure.marching_cubes_lewiner(aux, 0)
        verts, faces, normals, values = measure.marching_cubes(aux, 0, allow_degenerate=False)
        
        print('Cleaning memory', flush=True)
        del aux
        gc.collect()

        ###########################################
        # mesh cleaning, scaling, transforming ...
        ###########################################

        print("Mesh data 0:",flush=True)
        print("Verts: {}  Faces: {}".format(verts.shape, faces.shape))
        # shift back the initial padding of tif
        verts = verts - 5.0

        # no scale
        #scale = 0.0124
        #print("scaling mesh uniformly by: {}".format(scale))
        #scaled_verts = scale * mesh.vertices
        scale = 1.0
        scaled_verts = scale * verts

        x1,y1,z1,x2,y2,z2 = self.return_min_max(scaled_verts)
        print("Boundary box: [{}  {}  {}]  [{}  {}  {}]".format(x1,y1,z1,x2,y2,z2))

        halfLx = (x2 - x1)/2.
        halfLy = (y2 - y1)/2.
        halfLz = (z2 - z1)/2.

        #dx = - x1
        #dy = - (halfLy + y1)
        #dz = - (halfLz + z1)
        dx = - (halfLx + x1)
        dy = - (halfLy + y1)
        #dz = - z1
        dz = 0.0

        ftr = open("translate.dat", "a")
        ftr.write("{}\t\t{}\t{}\t{}\n".format(fname, dx, dy, dz))
        ftr.close()
        
        dr = np.asarray([dx, dy, dz])

        # mesh transform
        print("shift mesh by: {}".format(dr))
        scaled_verts += dr


        # cleaning

        #print("Mesh data 0:",flush=True)
        #print("Verts: {}  Faces: {}".format(verts.shape, faces.shape))
        
        #print('Remove isolated vertices', flush=True)
        #verts, faces, info = pymesh.remove_isolated_vertices_raw(verts, faces)
        #print(info, flush=True)

        '''
        print("Mesh data 1:",flush=True)
        print("Verts: {}  Faces: {}".format(verts.shape, faces.shape))
        
        print('Remove duplicated vertices', flush=True)
        verts, faces, info = pymesh.remove_duplicated_vertices_raw(verts, faces)
        print(info, flush=True)
        '''
        
        print("Mesh data before saving:",flush=True)
        print("Verts: {}  Faces: {}".format(verts.shape, faces.shape))
        
        print('Set the mesh ...', flush=True)
        scaled_verts = np.round(scaled_verts, 3)
        final_mesh = pymesh.form_mesh(scaled_verts, faces)
        
        savefilename = "{}.obj".format(fname)
        savefilepath = os.path.join('./', objDir, savefilename)
        
        print("saving file {}".format(savefilename), flush=True)
        
        #pymesh.save_mesh(savefilepath, final_mesh)
        #pymesh.save_mesh(savefilepath, final_mesh, use_float=True)
        #pymesh.save_mesh_raw(savefilepath, scaled_verts, faces)
        self.obj_writer(savefilepath, scaled_verts, faces)
        
        print('done.', flush=True)
        

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__), flush=True)

        print("Start time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)

        lines = self.read_dict(dictFileName)
        empty, sourceDir, description = \
            self.check_a_parameter('sourceDir', lines)
        empty, objDir, description = \
            self.check_a_parameter('objDir', lines)
        
        os.makedirs(objDir)
        
        ########################################################################
        # makeMesh
        tif_files = sorted([f for f in os.listdir(sourceDir)
                            if (os.path.isfile(
                os.path.join(sourceDir, f)) and ".tif" in f)])

        for j, filename in enumerate(tif_files):
            print('\n*********************************************', flush=True)
            print('  Processing file {}'.format(filename), flush=True)
            print('*********************************************', flush=True)
            filen = os.path.join(sourceDir, filename)

            self.makeMesh(filen)
        
        ########################################################################
        # stage 3 rotate and scale the mesh
        

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
