import os
from os.path import join, exists, split
from os import makedirs, listdir
from shutil import rmtree

import numpy as np

from scipy.misc import imread, imresize, imsave

from A_pick_samples import read_files, reset_directory, pack2foldersAWET #, reset_directory

np.random.seed(42)

if __name__ == "__main__":

    doPackAWET = True


    if doPackAWET:
        src_path = "/home/blaz/AWEToolBox/AWEToolbox/databases-to-pack/"
        dst_path = "./databases-packed/"

        src_dirs = read_files(src_path, isDir=True)
        for src_dir in src_dirs:

            src_folds = read_files(join(src_path, src_dir), isDir=True)
            for src_fold in src_folds:

                images = read_files(join(src_path, src_dir, src_fold)) 
                images = [i for i in images if not '.mat' in i]
                pack2foldersAWET(join(src_path, src_dir, src_fold), join(dst_path, src_dir, src_fold), images)