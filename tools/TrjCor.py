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


class TrjCor(pt.AbstractBaseTool):
    '''
    This tool does a trajectory correction on a small piece of data 50 in Z
    direction. The processing and result will be in directory `testTrjCor`.
    The tilt should be already fixed and the image should be cropped at this
    point.
    '''
    
    def __init__(self):
        self.__toolName__ = 'trajCor'
        
        # hardcoded directory and file names
        self.__resDir__ = 'trjCor'
        self.__voidsfile__ = 'cropdata'
        
        # an array of parameters for current generator
        self.parameters = []
        
        # the format for parameters: [name, initial_value, description
        self.parameters.append(
            ['inputDir', 'beforeCrop', 'directory with initial files']
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
            ['classModel', 'wekaModel', 'directory with classifier files']
        )
    
    def phase1(self, inputDir):
        os.makedirs(self.__resDir__)
        # dirrectory for voids
        voids_dir = os.path.join(self.__resDir__, 'voids')
        os.makedirs(voids_dir)
    
        # make initial directory with small datafiled (50 in Z)
        iniDir = os.path.join(self.__resDir__, 'ini')
        os.makedirs(iniDir)
    
        tif_files = sorted([f for f in os.listdir(inputDir)
                            if (os.path.isfile(
                os.path.join(inputDir, f)) and ".tif" in f)])
    
        # read and check void coordinates
        with open(self.__voidsfile__) as f:
            lines = f.read().splitlines()
        voids = {}
        for line in lines:
            vd = line.split()
            vdcoords = [int(x) for x in vd[1:]]
            voids[vd[0]] = vdcoords
        for key, value in voids.items():
            if (value[5] - value[2]) <= 0 or \
                    (value[4] - value[1]) <= 0 or \
                    (value[3] - value[0]) <= 0:
                print(' *** Error. Void coordinates are wrong, check {} file'.
                      format(self.__voidsfile__))
                print("x: {}  y: {}  z: {}".format
                    (
                    (value[5] - value[2]),
                    (value[4] - value[1]),
                    (value[3] - value[0])
                )
                )
                sys.exit(2)
            void_dir = os.path.join(voids_dir, key)
            os.mkdir(void_dir)
    
        for j, filename in enumerate(tif_files):
            print('\n*********************************************')
            print('  Processing file {}'.format(filename))
            print('*********************************************')
        
            filen = os.path.join(inputDir, filename)
            stack_tif = io.imread(filen, plugin='tifffile')
        
            for key, value in voids.items():
                if value[2] < 20 or value[1] < 20 or value[0] < 20 or \
                        value[5] >= stack_tif.shape[2] - 20 or \
                        value[4] >= stack_tif.shape[2] - 20 or \
                        value[3] >= stack_tif.shape[2] - 20:
                    print(' *** Error. Some void coordinates are out of range, '
                          'check {} file'.format(self.__voidsfile__))
                    sys.exit(2)
        
            stack_tif_cut = stack_tif[200:250, :, :]
        
            longName = os.path.splitext(filename)[0]
            longNameSpl = os.path.splitext(filename)[0].split('_')
        
            resultFilename = longName[0:8]
        
            for wrd in longNameSpl:
                if 'min' in wrd:
                    timeI = int(wrd[:-3])
                    time = str(timeI).zfill(4)
                    resultFilename = resultFilename + '_' + time + 'min'
        
            # save the whole image (cropped in this case)
            savef = os.path.join(iniDir, '{}.tif'.format(resultFilename))
            io.imsave(savef, stack_tif_cut, plugin='tifffile')
        
            # save voids
            for key, value in voids.items():
                crvoid = stack_tif \
                    [
                         value[2] - 20: value[5] + 20,
                         value[1] - 20: value[4] + 20,
                         value[0] - 20: value[3] + 20
                         ]
                void_dir = os.path.join(voids_dir, key)
                saveVf = os.path.join(void_dir, '{}.tif'.format(resultFilename))
                io.imsave(saveVf, crvoid, plugin='tifffile')
    
    def phase2(self, wekaDir):
        wekaModel = 'weka.model'
        wekaclassifierpath = os.path.join(wekaDir, wekaModel)
        
        voids_dir = os.path.join(self.__resDir__, 'voids')

        dirs = [d for d in os.listdir(voids_dir) if
                (os.path.isdir(os.path.join(voids_dir, d)) and
                 "void" in d and
                 not "seg" in d)
                ]
        dirs = sorted(dirs)
        
        for i, dr in enumerate(dirs):
            print("Classifying files from directory: {}".format(dr))
            curdir = os.path.join(voids_dir, dr)
            resdir = os.path.join(voids_dir, "seg_{}".format(dr))
            os.mkdir(resdir)
            bashCommand = "java bsh.Interpreter wekaClsfc3D.bsh {} {} {}".\
                    format(curdir, resdir, wekaclassifierpath)
            process = subprocess.call(bashCommand, shell=True)
            
    def phase3(self):
        voids_dir = os.path.join(self.__resDir__, 'voids')
    
        # choose only seg voids to clean
        dirs = [d for d in os.listdir(voids_dir) if
                (os.path.isdir(os.path.join(voids_dir, d)) and
                 "seg_void" in d)
                ]
        dirs = sorted(dirs)
    
        for i, dr in enumerate(dirs):
            curdir = os.path.join(voids_dir, dr)
            
            tif_files = sorted([f for f in os.listdir(curdir)])
            for ind, filename in enumerate(tif_files):
                print('\n*********************************************')
                print('  Processing file {}'.format(filename))
                print('  In directory {}'.format(dr))
                print('*********************************************')
                im_class = io.imread(os.path.join(curdir, filename),
                                     plugin='tifffile')
                im_bin = np.logical_not(im_class.astype(bool)).astype(int)
    
                slsl = im_bin.astype(np.uint8)
    
                # clean boundaries in slsl
                cleanCl = True
                while cleanCl:
        
                    lw, num = measurements.label(slsl)
        
                    minLab = np.min(lw)
                    maxLab = np.max(lw)
                    print("labels:  min: {}   max: {}".format(minLab, maxLab))
        
                    hist = measurements.histogram(lw, minLab + 1, maxLab,
                                                  maxLab - minLab)
        
                    maxClLab = np.argmax(hist) + 1
        
                    print("label of a biggest cluster: {}".format(maxClLab))
                    indMaxCl = np.where(lw == maxClLab)
        
                    ccc = False
                    for ii, indx in enumerate(indMaxCl):
                        minI = min(indx)
                        maxI = max(indx)
            
                        if minI == 0 or maxI == lw.shape[ii] - 1:
                            ccc = True
                            break
        
                    if ccc:
                        slsl[lw == maxClLab] = 0
                    else:
                        cleanCl = False
    
                lw, num = measurements.label(slsl)
    
                minLab = np.min(lw)
                maxLab = np.max(lw)
                print("labels:  min: {}   max: {}".format(minLab, maxLab))
    
                hist = measurements.histogram(lw, minLab + 1, maxLab,
                                              maxLab - minLab)
    
                maxClLab = np.argmax(hist) + 1
                slsl[lw != maxClLab] = 0

                inlw = slsl.astype(np.uint8)
                inlw *= 255  # 65535 #256
    
                io.imsave(os.path.join(curdir, filename), inlw,
                          plugin='tifffile')
    
    def phase4(self):
        # read and check void coordinates
        with open(self.__voidsfile__) as f:
            lines = f.read().splitlines()
        voids = {}
        for line in lines:
            vd = line.split()
            vdcoords = [int(x) for x in vd[1:]]
            voids[vd[0]] = vdcoords
    
        centers = []
    
        # iterate over different voids
        for key, value in voids.items():
    
            void_dir = "seg_{}".format(key)
            
            cur_dir = os.path.join(self.__resDir__, 'voids', void_dir)
        
            tif_files = sorted([f for f in os.listdir(cur_dir)])
        
            cc = []
            for ind_1, filename in enumerate(tif_files):
                print('\n*********************************************')
                print('  Processing file {}'.format(filename))
                print('*********************************************')
                im_class = io.imread(os.path.join(cur_dir, filename),
                                     plugin='tifffile')
                im_bin = im_class.astype(bool).astype(int)
            
                # calculating centers
                slsl = im_bin.astype(np.uint8)
            
                lw, num = measurements.label(slsl)
            
                COM = measurements.center_of_mass(slsl)
                COM = np.asarray(COM)
                print("COM  {}".format(COM))
            
                ini = np.asarray(value[0:3]).astype(float)
            
                print('center of mass: ', COM, ini)
    
                cc.append(COM + ini)
        
            centers.append(cc)
    
        centers = np.asarray(centers)
    
        f = open('traj.xyz', 'w')
    
        for i in range(centers.shape[1]):
            f.write("{}\n".format(centers.shape[0]))
            # write comment
            f.write("void centers\n")
            for j in range(centers.shape[0]):
                wrc = "{}  {}\n".format(j, centers[j][i])
                wrc = wrc.replace('[', '')
                wrc = wrc.replace(']', '')
                f.write(wrc)
                
        f.close()
        
    ############################################################################
    ############################################################################
    ############################################################################
    # read positions from file
    def readPos(self, file):
        traj = []
        with open(file) as f:
            line = f.readline()
            while line:
                N = int(line)
                comment = f.readline()
            
                for i in range(N):
                    line = f.readline()
                    pos = np.asarray([float(x) for x in line.split()][1:4])
                
                    if i >= len(traj):
                        traj.append([])
                    traj[i].append(pos)
            
                line = f.readline()
        
            traj = np.asarray(traj)
        return traj

    # read crop data
    def readCrop(self, file):
        cropf = open(file)
    
        voids = {}
        for line in cropf:
            splline = line.split()
            voids[splline[0]] = np.asarray(splline[1:4]).astype(float)
    
        return voids

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):
    
        assert (self.isRotationMatrix(R))
    
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
        singular = sy < 1e-6
    
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
    
        return np.array([x, y, z])

    # Input: expects Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    def rigid_transform_3D(self, A, B):
        assert len(A) == len(B)
    
        N = A.shape[0];  # total points
    
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
    
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
    
        # dot is matrix multiplication for array
        # H = np.transpose(AA) * BB
        H = np.matmul(np.transpose(AA), BB)
    
        U, S, Vt = np.linalg.svd(H)

        # R = Vt.T * U.T
        R = np.matmul(Vt.T, U.T)
    
        # special reflection case
        # if np.linalg.det(R) < 0:
        #   print("Reflection detected")
        #   Vt[2,:] *= -1
        #   R = Vt.T * U.T

        # t = -R*centroid_A.T + centroid_B.T
        t = -np.matmul(R, centroid_A.T) + centroid_B.T

        # print(t)
    
        return R, t
    ############################################################################
    ############################################################################
    ############################################################################
    
    def phase5(self):
        # trajectory file
        datafile = 'traj.xyz'
        # initial position of cropped part
        cropfile = self.__voidsfile__
        # directory to process. should be the same number of files for time
        datadir = os.path.join(self.__resDir__, 'ini')
    
        resdir = '{}_TR'.format(datadir)
        os.makedirs(resdir)
    
        # trajectories
        trs = self.readPos(datafile)
    
        # *************************************************************
    
        sub_trs = []
        # subract 0
        for tjs in trs:
            sub_trs.append(np.subtract(tjs, tjs[0]))
    
        sub_trs = np.asarray(sub_trs)
    
        tr1 = np.average(sub_trs, axis=0)
    
        # *************************************************************
    
        # crop info
        #voids = self.readCrop(cropfile)
    
        centroid = np.average(trs, axis=0)
        print("centroid shape  {}".format(centroid.shape))
    
        print("trs shape  {}".format(trs.shape))
    
        # trs dimentions
        # axis 0 - different voids
        # axis 1 - time
        # axis 2 - coordinates
        #rotationAngles = np.zeros(trs.shape[1])
        rotationAngles0 = np.zeros(trs.shape[1])
        rotationAngles1 = np.zeros(trs.shape[1])
        rotationAngles2 = np.zeros(trs.shape[1])
        translation = np.zeros((trs.shape[1], 3))
    
        # iterate time
        for time in range(1, trs.shape[1]):
            points0 = trs[:, 0, :]
            pointsTime = trs[:, time, :]
        
            ret_R, ret_t = self.rigid_transform_3D(points0, pointsTime)
    
            eulerAngles = self.rotationMatrixToEulerAngles(ret_R)
        
            #rotationAngles[time] = math.degrees(eulerAngles[2])
            rotationAngles0[time] = math.degrees(eulerAngles[0])
            rotationAngles1[time] = math.degrees(eulerAngles[1])
            rotationAngles2[time] = math.degrees(eulerAngles[2])

            translation[time] = ret_t
        
            print("{}   translation: "
                  "{}   tr: "
                  "{}   degrees: {}".format(
                time, translation[time], tr1[time], rotationAngles0[time])
            )
    
        print('\n*********************************************')
        print(' Apply translational transformation ')
        print('*********************************************')
    
        tif_files = sorted([f for f in os.listdir(datadir)
                            if (os.path.isfile(
                os.path.join(datadir, f)) and ".tif" in f)])
    
        for j, filename in enumerate(tif_files):
            # don't need to move first image
            print('\n*********************************************')
            print('  Processing file {}'.format(filename))
            print('*********************************************')
        
            filen = os.path.join(datadir, filename)
            stack_tif = io.imread(filen, plugin='tifffile')
        
            print("shape: {}".format(stack_tif.shape))
        
            print("current time {}".format(
                strftime("%Y-%m-%d %H:%M:%S", localtime())))
    
            # apply shift not to first file
            shift = -translation[j]
        
            res = stack_tif
    
            #angl = - rotationAngles[j]
            angl0 = - rotationAngles0[j]
            angl1 = - rotationAngles1[j]
            angl2 = - rotationAngles2[j]

            #print("rotation angle: {}".format(angl))
            #res = ndimage.rotate(res, angl, axes=(1, 2), reshape=False,
            #                     output=None, order=3, mode='constant', cval=0.0,
            #                     prefilter=True)

            if j != 0:
                print("rotation angle: {}".format(angl0))
                res = ndimage.rotate(res, angl0, axes=(1, 2), reshape=False,
                                     output=None, order=3, mode='constant', cval=0.0,
                                     prefilter=True)
                '''
                print("rotation angle: {}".format(angl1))
                res = ndimage.rotate(res, angl1, axes=(0, 2), reshape=False,
                                     output=None, order=3, mode='constant', cval=0.0,
                                     prefilter=True)
                print("rotation angle: {}".format(angl2))
                res = ndimage.rotate(res, angl2, axes=(0, 1), reshape=False,
                                     output=None, order=3, mode='constant', cval=0.0,
                                     prefilter=True)
                '''
    
                
                print("#{} shift: {}".format(j, shift))
                res = ndimage.interpolation.shift(res, shift)
        
            filename = os.path.splitext(filename)[0]
            savef = os.path.join(resdir, '{}_trsmd.tif'.format(filename))
            io.imsave(savef, res, plugin='tifffile')
        
    def phase6(self):
        path_ini = os.path.join(self.__resDir__, 'ini_TR')
        temp_data_dir = "{}_tmp_split".format(path_ini)
        os.mkdir(temp_data_dir)
        temp_data_dir_filtered = "{}_filtered".format(temp_data_dir)
        os.mkdir(temp_data_dir_filtered)
    
        path_res = os.path.join(self.__resDir__, 'ini_TR_BP')
        os.mkdir(path_res)
    
        data_files = sorted([f for f in os.listdir(path_ini) if
                             os.path.isfile(os.path.join(path_ini, f))])
    
        shapes = []
        
        # calculate n_ based on the thickness of processed layer.
        # It is chosen 50 for reason of limited memory
        n_layers = 50
        # read one file to get the shape of the arrays
        test_img = \
            io.imread(os.path.join(path_ini, data_files[0]), plugin='tifffile')
        sizeZ = test_img.shape[0]
        
        n_ = int( sizeZ / n_layers )
        # set the limit on n_
        if n_ == 1:
            n_ = 2

        print("######################")
        print("Split the files into {} pieces".format(n_))
        print("######################")

        for i, file in enumerate(data_files):
            print("  Processing: {}".format(file))
            fp = os.path.join(path_ini, file)
            img = io.imread(fp, plugin='tifffile')
        
            print("    Image dimentions: {}".format(img.shape))
            shapes.append(img.shape)
        
            slice_size = round(img.shape[0] / n_)
            for j in range(n_):
                splitted_file_name = "{}_part{}.tif".format(
                    os.path.splitext(file)[0], j)
                savef = os.path.join(temp_data_dir, splitted_file_name)
                bottom = j * slice_size
                top = min(img.shape[0], (j + 1) * slice_size)
                # print("slice: {}  bot: {} top: {}".format(j, bottom, top))
                if j == (n_ - 1) and top < (img.shape[0] - 1):
                    top = img.shape[0] - 1
                io.imsave(savef, img[bottom:top, :, :], plugin='tifffile')
    
        print("######################")
        print("Filter files")
        print("######################")
        bashCommand = "java -Xmx100G  bsh.Interpreter ijFFTbandpass.bsh {} {}". \
            format(temp_data_dir, temp_data_dir_filtered)
        process = subprocess.call(bashCommand, shell=True)
    
        print("######################")
        print("Reconstruct files")
        print("######################")
    
        for i, file in enumerate(data_files):
            rec_f_name = "{}_filtered.tif".format(os.path.splitext(file)[0])
            res = np.zeros(shape=(shapes[i]), dtype=np.uint16)
            print("  Reconstructing: {}".format(rec_f_name))
            current_min_pos = 0
            for j in range(n_):
                spl_rec_f_name = "{}_part{}_fil.tif".format(
                    os.path.splitext(file)[0], j)
                fp = os.path.join(temp_data_dir_filtered, spl_rec_f_name)
                img = io.imread(fp, plugin='tifffile')
            
                this_slice = img.shape[0]
                current_max_pos = current_min_pos + this_slice
                res[current_min_pos:current_max_pos, :, :] = img
                current_min_pos = current_max_pos
        
            savef = os.path.join(path_res, rec_f_name)
            io.imsave(savef, res, plugin='tifffile')
    
        rmtree(temp_data_dir)
        rmtree(temp_data_dir_filtered)
        
    def phase7(self, wekaDir):
        # file to be segmented
        #path_file_ini = os.path.join(self.__testDir__, 'ini_TR_BP', '02_run06_0015min_trsmd_filtered.tif')

        path_file_iniDir = os.path.join(self.__resDir__, 'ini_TR_BP')

        tif_files_from_ini = sorted([f for f in os.listdir(path_file_iniDir)
                                  if (os.path.isfile(
                os.path.join(path_file_iniDir, f)) and ".tif" in f)])

        path_file_ini = os.path.join(path_file_iniDir, tif_files_from_ini[0])

        # a directory where result should be stored
        path_res = os.path.join(self.__resDir__, 'ini_TR_BP_cl0')
        # number of pieces to split the tomo data
        
        # model file
        #model_file = os.path.join(wekaDir, 'run6t0wekaFFTband.model')
        weka_files = sorted([f for f in os.listdir(wekaDir)
                             if (os.path.isfile(os.path.join(wekaDir, f))
                                 and "FFTband" in f)])
        model_file = os.path.join(wekaDir, weka_files[0])

        os.mkdir(path_res)
    
        # make temporary directorires
        temp_data_dir = "{}_tmp_split".format(path_res)
        os.mkdir(temp_data_dir)
        temp_data_dir_class = "{}_class".format(temp_data_dir)
        os.mkdir(temp_data_dir_class)
    
        img = io.imread(path_file_ini, plugin='tifffile')
    
        file_ini = os.path.split(path_file_ini)[1]
    
        print("    Image dimentions: {}".format(img.shape))
        shape = img.shape
        
        # calculate n_ based on the thickness of processed layer.
        # It is chosen 50 for reason of limited memory
        n_layers = 50
        # read one file to get the shape of the arrays
        sizeZ = shape[0]
        n_ = int(sizeZ / n_layers)
        # set the limit on n_
        if n_ == 1:
            n_ = 2

        print("######################")
        print("Split the file into {} pieces".format(n_))
        print("######################")

        slice_size = round(shape[0] / n_)
    
        for j in range(n_):
            splitted_file_name = "{}_part{}.tif".format(
                os.path.splitext(file_ini)[0], j)
            savef = os.path.join(temp_data_dir, splitted_file_name)
        
            bottom = max(0, j * slice_size - 5)
            top = min(img.shape[0], (j + 1) * slice_size + 5)
        
            if j == (n_ - 1) and top < (img.shape[0] - 1):
                top = img.shape[0] - 1
            io.imsave(savef, img[bottom:top, :, :], plugin='tifffile')
    
        print("######################")
        print("Classify files")
        print("######################")
        bashCommand = "java -Xmx100G  bsh.Interpreter wekaClsfc3D.bsh {} {} {}". \
            format(temp_data_dir, temp_data_dir_class, model_file)
        process = subprocess.call(bashCommand, shell=True)
    
        print("######################")
        print("Reconstruct files")
        print("######################")
    
        rec_f_name = "{}_seg.tif".format(os.path.splitext(file_ini)[0])
        res = np.zeros(shape=shape, dtype=np.uint8)
        print("  Reconstructing: {}".format(rec_f_name))
        current_min_pos = 0
        for j in range(n_):
            spl_rec_f_name = "{}_part{}_seg.tif".format(
                os.path.splitext(file_ini)[0], j)
            fp = os.path.join(temp_data_dir_class, spl_rec_f_name)
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
            res[current_min_pos:current_max_pos, :, :] = img[min_sl:max_sl, :,
                                                         :]
            current_min_pos = current_max_pos
    
        savef = os.path.join(path_res, rec_f_name)
        io.imsave(savef, res, plugin='tifffile')
    
        rmtree(temp_data_dir)
        rmtree(temp_data_dir_class)
        
    def phase7p5(self):
        choice = ''
        while choice != 'y' and choice != 'n':
            # need to know what is the version of the interpreter
            choice = input('\nAt this point the area of the capillary needs to\n'
                           ' be manually cleaned. Erase all the noise out of\n'
                           ' the internal area of the capillary and then\n'
                           ' enter `y` here.\n'
                           ' Did you clean the image and want to'
                           ' continue? (y/n): ')
        if choice == 'y':
            return
        else:
            print('Running interrupted')
            sys.exit(2)


    ############################################################################
    ############################################################################
    ############################################################################


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
        print("labels:  min: {}   max: {}".format(minLab, maxLab))
        
        hist = measurements.histogram(lw, minLab + 1, maxLab, maxLab - minLab)
        
        # maxCl = np.max(hist)
        maxClLab = np.argmax(hist) + 1
        print("label of a biggest cluster: {}".format(maxClLab))
        aux[lw != maxClLab] = 0
        return (aux[1:-1, 1:-1, 1:-1]).astype(np.uint8)
    
    
    # erode if number of near neighbors less then 3
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
        conv *= 0.2
        conv = np.around(conv)
        conv = conv.astype(np.uint8)
        # at this point conv is 0 or 1, because max was 6
        return (np.logical_and(conv, bin_image)).astype(np.uint8)
    
    
    def erode_converge(self, bin_image):
        print("Erode array")
        size = bin_image.shape[0] * bin_image.shape[1] * bin_image.shape[2]
        next_size = measurements.sum(bin_image)
        while (next_size < size):
            print("size: {}  next: {}".format(size, next_size))
            bin_image = self.erode_NN(bin_image)
            size = next_size
            next_size = measurements.sum(bin_image)
        
        return bin_image
    
    
    def clean_image(self, bin_image, mark="  "):
        print("{}  A0 ** Number: {}".format(mark, measurements.sum(bin_image)))
        
        bin_image = binary_fill_holes(bin_image).astype(np.uint8)
        
        print("{}  A1 ** Number: {}".format(mark, measurements.sum(bin_image)))
        
        bin_image = binary_erosion(bin_image, iterations=2).astype(np.uint8)
        
        print("{}  A ** Number: {}".format(mark, measurements.sum(bin_image)))
        
        bin_image = binary_fill_holes(bin_image).astype(np.uint8)
        
        print("{}  B ** Number: {}".format(mark, measurements.sum(bin_image)))
        
        bin_image = binary_dilation(bin_image, iterations=2).astype(np.uint8)
        
        print("{}  C ** Number: {}".format(mark, measurements.sum(bin_image)))
        
        bin_image = self.erode_converge(bin_image)
        
        print("{}  D ** Number: {}".format(mark, measurements.sum(bin_image)))
        
        bin_image = self.clean_not_attached(bin_image)
        
        print("{}  E ** Number: {}".format(mark, measurements.sum(bin_image)))
        
        return bin_image.astype(np.uint8)
    
    
        ############################################################################
        ############################################################################
        ############################################################################

    
    def phase8(self):
        dir_path = os.path.join(self.__resDir__, 'ini_TR_BP_cl0')
        res_path = os.path.join(self.__resDir__, 'ini_TR_BP_cl0_cl')
        
        os.mkdir(res_path)

        # get the .tif files from the directory
        tif_files = sorted([f for f in os.listdir(dir_path)
                            if (os.path.isfile(
                os.path.join(dir_path, f)) and ".tif" in f)])
    
        for i, file in enumerate(tif_files):
            print('\n*********************************************')
            print('  Processing file {}'.format(file))
            print('*********************************************')
            file_path = os.path.join(dir_path, file)
            im_class = io.imread(file_path, plugin='tifffile')
            im_bin = im_class.astype(bool).astype(np.uint8)
        
            # 1 - is pore space here, 0 - solid
        
            print("######################")
            print("Clean bright")
            print("######################")
        
            im_bin = self.clean_image(im_bin, mark="I ")
        
            print("######################")
            print("Inverting image")
            print("######################")
        
            im_bin = np.logical_not(im_bin).astype(np.uint8)
        
            # 0 - pore space, 1 - is solid
        
            print("######################")
            print("Clean dark")
            print("######################")
        
            im_bin = self.clean_image(im_bin, mark="II")
    
            im_bin *= 255
            
            file_res = os.path.join(res_path, file)
        
            io.imsave(file_res, im_bin, plugin='tifffile')
            
    def phase9(self):
        dir_path = os.path.join(self.__resDir__, 'ini_TR_BP')
    
        resdir = '{}_subtr'.format(dir_path)
        os.mkdir(resdir)
    
        tif_files = sorted([f for f in os.listdir(dir_path)
                            if (os.path.isfile(
                os.path.join(dir_path, f)) and ".tif" in f)])

        time0fileDir = os.path.join(self.__resDir__, 'ini_TR_BP_cl0_cl')
        
        tif_files_from0 = sorted([f for f in os.listdir(time0fileDir)
                            if (os.path.isfile(
                os.path.join(time0fileDir, f)) and ".tif" in f)])

        time0file = os.path.join(time0fileDir, tif_files_from0[0])
        
        stack_tif0 = io.imread(time0file, plugin='tifffile')

        for i, filename in enumerate(tif_files):
            print('\n*********************************************')
            print('  Processing file {}'.format(filename))
            print('*********************************************')
        
            filen = os.path.join(dir_path, filename)
            stack_tif = io.imread(filen, plugin='tifffile')
        
            print("Subtracking 0 time...")
        
            stack_tif[stack_tif0 != 0] = 0
        
            print("Saving array of size {} as tif...".format(stack_tif.shape))
            print("array is shorter in Z direction in order to remove "
                "segmentation proc boundaries effect")
            filename = os.path.splitext(filename)[0]
            savef = os.path.join(resdir, '{}_subtr.tif'.format(filename))
            io.imsave(savef, stack_tif[1:-1, :, :], plugin='tifffile')
            
            

    def execute(self, dictFileName):
        print('Starting {0:s} tool'.format(self.__toolName__))

        print("Start time {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())))

        lines = self.read_dict(dictFileName)
        empty, inputDir, description = \
            self.check_a_parameter('inputDir', lines)
        empty, wekaDir, description = \
            self.check_a_parameter('classModel', lines)
        
        self.phase1(inputDir)
        
        self.phase2(wekaDir)
        
        # segmenting voids
        self.phase3()
        
        # calculating center of mass
        self.phase4()
        
        # apply rotation and translation
        self.phase5()
        
        # apply bandpass filter
        self.phase6()
        
        # segment time 0
        self.phase7(wekaDir)
        
        print("Current time before manual cleaning {}".format(
            strftime("%Y-%m-%d %H:%M:%S", localtime())))
        # clean segmented time 0
        # probably manually
        self.phase7p5()
        
        # remove small clusters from segmented time 0
        self.phase8()
        
        # subtrack
        self.phase9()

        return True
