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


__author__ = 'Vitaliy Starchenko'


class SegmentCryst(pt.AbstractBaseTool):
    '''
    This tool does the two phase segmentation of crystals and reactive fluid.
    '''
    
    def __init__(self):
        self.__toolName__ = 'segmentCrystals'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['inputDir', 'subtrackedDir', 'directory with initial files '
                                       'which are subtracked images']
        )
        
        self.parameters.append(
            ['outputDir', 'resultDir', 'directory where the final files '
                                       'will be placed']
        )
        # if needed to check correctness quickly z1 and z2 will define a slice
        # to analyze. If at least one is -1 then analyzing the whole image.
        self.parameters.append(
            ['z1', -1, 'lower Z position in slice']
        )
        self.parameters.append(
            ['z2', -1, 'higher Z position in slice']
        )
        
        self.parameters.append(
            ['classModel', 'wekaModelTime', 'directory with classifier files']
        )
        
    def compare_times(self, name_file, name_class):
    
        file_time = \
            os.path.splitext(name_file)[0].split('_')
        
        file_time_1 = [f for f in file_time if 'min' in f]
        file_time_int = int(file_time_1[0][:-3])
        
        class_model_time = \
            os.path.splitext(name_class)[0].split('_')[-1]
    
        class_model_time_int = int(class_model_time[:-3])
        
        return file_time_int > class_model_time_int

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__), flush=True)

        print("Start time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)

        lines = self.read_dict(dictFileName)
        empty, inputDir, description = \
            self.check_a_parameter('inputDir', lines)
        empty, outputDir, description = \
            self.check_a_parameter('outputDir', lines)
        empty, z1, description = \
            self.check_a_parameter('z1', lines)
        empty, z2, description = \
            self.check_a_parameter('z2', lines)
        empty, wekaDir, description = \
            self.check_a_parameter('classModel', lines)

        z1 = int(z1)
        z2 = int(z2)

        # number of layers the image is split to in Z direction
        #n_ = int(ags[3])

        # make output directoriry
        #cryst_dir = "{}_cryst".format(inputDir)
        #os.mkdir(cryst_dir)
        os.mkdir(outputDir)

        # make temporary directorires
        temp_data_dir = "{}_tmp_split".format(inputDir)
        os.mkdir(temp_data_dir)
        class_split_data_dir = "{}_class".format(temp_data_dir)
        os.mkdir(class_split_data_dir)

        # get the classifier model file names
        filesclass = sorted([f for f in os.listdir(wekaDir)
                             if os.path.isfile(os.path.join(wekaDir, f))])

        # get data file names
        data_files = sorted([f for f in os.listdir(inputDir)
                             if os.path.isfile(os.path.join(inputDir, f))])

        n_layers = 50
        # read one file to get the shape of the arrays
        test_img = \
            io.imread(os.path.join(inputDir, data_files[0]), plugin='tifffile')
        sizeZ = test_img.shape[0]

        n_ = int(sizeZ / n_layers)
        # set the limit on n_
        if n_ <= 1:
            n_ = 2
            
        print("######################", flush=True)
        print("Split the files into {} pieces".format(n_), flush=True)
        print("######################", flush=True)

        # aux variables to the classification files
        i1 = 0
        class_model = ''
        for i, file in enumerate(data_files):
            
            if i==0:
                continue
    
            current_file_dir = "{}".format(os.path.splitext(file)[0])
            current_file_dir = os.path.join(temp_data_dir, current_file_dir)
            os.mkdir(current_file_dir)
    
            print("  Processing: {}".format(file), flush=True)
            fp = os.path.join(inputDir, file)
            img = io.imread(fp, plugin='tifffile')
    
            print("    Image dimentions:  {}".format(img.shape), flush=True)
            cur_shape = img.shape
    
            slice_size = round(img.shape[0] / n_)
            for j in range(n_):
                splitted_file_name = "{}_part{}.tif".format(
                    os.path.splitext(file)[0], j)
                savef = os.path.join(current_file_dir, splitted_file_name)
                bottom = max(0, j * slice_size - 5)
                top = min(img.shape[0], (j + 1) * slice_size + 5)
                if j == (n_ - 1) and top < (img.shape[0] - 1):
                    top = img.shape[0] - 1
                io.imsave(savef, img[bottom:top, :, :].astype(np.uint16), plugin='tifffile')
    
            # comparing the time of current file and classifier
            
            
            #if class_model == '' or self.compare_times(file, filesclass[i1]):
            #    class_model = os.path.join(wekaDir, filesclass[i1])
            #    i1 += 1
            #else:
            #    class_model = os.path.join(wekaDir, filesclass[i1 - 1])
            class_model = os.path.join(wekaDir, filesclass[i - 1])

            # read the model
            print("tiff file: {}   weka model file: {}".
                  format(current_file_dir, class_model), flush=True)
    
            print("  Classify files: {}".format(file), flush=True)
            bashCommand = "java -Xmx100G  bsh.Interpreter wekaClsfc3D.bsh {} {} {}". \
                format(current_file_dir, class_split_data_dir, class_model)
            print("bash: {}".format(bashCommand), flush=True)
            process = subprocess.call(bashCommand, shell=True)
    
            rec_f_name = "{}_seg.tif".format(os.path.splitext(file)[0])
            res = np.zeros(shape=(cur_shape), dtype=np.uint8)
            print("  Reconstructing: {}\n".format(rec_f_name), flush=True)
            current_min_pos = 0
            for j in range(n_):
                spl_rec_f_name = "{}_part{}_seg.tif".format(
                    os.path.splitext(file)[0], j)
                # spl_rec_f_name = "{}_part{}.tif".format(os.path.splitext(file)[0], j)
                fp = os.path.join(class_split_data_dir, spl_rec_f_name)
                img = io.imread(fp, plugin='tifffile')
        
                added = 10
        
                min_sl = int(added / 2.)
                max_sl = -min_sl
        
                if j == 0:
                    added = 5
                    min_sl = 0
                if j == n_ - 1:
                    added = 5
                    max_sl = img.shape[0]
        
                this_slice = img.shape[0] - added
                current_max_pos = current_min_pos + this_slice
                res[current_min_pos:current_max_pos, :, :] = img[min_sl:max_sl,
                                                             :, :]
                current_min_pos = current_max_pos
    
            #savef = os.path.join(cryst_dir, rec_f_name)
            savef = os.path.join(outputDir, rec_f_name)
            io.imsave(savef, res[:, :, :], plugin='tifffile')

        rmtree(temp_data_dir)
        rmtree(class_split_data_dir)

        print("End time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())), flush=True)
        
        return True
