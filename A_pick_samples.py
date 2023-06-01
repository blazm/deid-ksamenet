import os
from os.path import join, exists, split
from os import makedirs, listdir
from shutil import rmtree

import numpy as np

from scipy.misc import imread, imresize, imsave


np.random.seed(42)

# TODO: rename to list_files
"""
def read_files(path):
    '''Read filenames in the path. '''
    filelist = []
    for item in os.listdir(path):
       # ident, sess, num = item.split('_')
       # ids.append(int(ident))
        if item=='.DS_Store': continue
        filelist.append(item)
        #os.path.join(path,item)

    return filelist
"""

def read_files(path, isDir=False):
    '''Read filenames in the path. '''
    filelist = []
    for item in os.listdir(path):
       # ident, sess, num = item.split('_')
       # ids.append(int(ident))
        if isDir and os.path.isdir(os.path.join(path, item)):
            filelist.append(item)
        elif not isDir and not os.path.isdir(os.path.join(path, item)):
            filelist.append(item)
        #os.path.join(path,item)
    return filelist


def extractIDs(filelist):
    #ids = []
    ids = {}

    for item in filelist: # os.listdir(path):
        ident, sess, num = item.split('.')[0].split('_')
        try:
            ident = int(ident)
        except ValueError:
            # handle CK+ DB: ValueError: invalid literal for int() with base 10: 'S053'
            ident = int(ident[1:]) 

        if ident in ids:
            ids[ident].append((sess, num))
        else:
            ids[ident] = [(sess, num)]

        #ids.append(ident)

    good_ids = []
    for k in ids.keys():
        #print ids[k]
        if len(ids[k]) >= 2:
            if ('1', '1') in ids[k] and ('2', '1') in ids[k]:
                good_ids.append(k)

    print "LEN GOOD: ", len(good_ids)

    return good_ids

def sample_folds(ids, num_folds, num_ids): 
    '''Perform random sampling without replacing and return num_folds of num_ids size.'''
    ids = set(ids)
    folds = []

    #picks = np.random.choice(list(ids), num_ids*num_folds, replace=False)
    for i in range(num_folds):
    
        #fold = picks[i*num_ids:(i+1)*num_ids]
        #print len(fold)
        

        #print len(list(ids))
        fold = np.random.choice(list(ids), num_ids, replace=False)
        fold = set(fold)
        ids = ids - fold
        folds.append(list(fold))

    return folds

def assemble_gallery(ids, session, db='xm2vts', emo=None):
    '''Assemble filenames in DB format from list of ids and session (1 or 2). '''
    
    #if db is "ck+":
        #name_format = "S%03d_%03d_%08d.ppm" # ck+
    #    name_format = "%s"

        #ident, emo, max_ix = params
        #filename = name_format % params
    #elif db is "xm2vts":
    num = 1 # first frontal shot
    name_format = "%03d_%d_%d.ppm" # xm2vts
    params = (ids, [session]*len(ids), [num]*len(ids))
        #ident, emo, max_ix = params
        #filename = name_format % params

    filenames = []
    for ident in ids:
        if db == 'ck+':
            #if emo:
                #ident, ext = ident.split('.')
            #    filename = ident #+ '_' + emo[ident] + '.' + ext
            #else:
            filename = ident

        else:
            filename = name_format % (ident, session, num)
        filenames.append(filename)

    return filenames

def write2file(filename, items):
    '''Write list to text file. '''
    with open(filename, "w") as f: 
        print items
        f.write('\n'.join(items))


def loadFold(filename):
    with open(filename, "r") as f: 
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

def idsFromFold(filenames):
    ids = [];
    for item in filenames:
        ident, sess, num = item.split('_')
        ids.append(int(ident))
    return ids

# TODO: change this to generator, to yield image by image
def loadImages(path, names=[], newdim=None):
    images = []
    if not names:
        names = read_files(path)

    for name in names:
        img = imread(os.path.join(path, name))
        if newdim:
            h, w, ch = img.shape
            img = imresize(img, (newdim, newdim, ch), interp='bicubic')
        images.append(img)
    images = np.array(images)
    #print images.shape

    return images

def loadPackedImages(path, names, newdim=None):
    ids = idsFromFold(names)
    images = []
    for id_, name in zip(ids, names):
        img = imread(os.path.join(path, "%03d" %(id_), name))
        if newdim:
            h, w, ch = img.shape
            img = imresize(img, (newdim, newdim, ch), interp='bicubic')
        images.append(img)
    images = np.array(images)
    print images.shape

    return images

def cropImages(path, names):
    pass

def reset_directory(d, purge=True):
    '''Remove directory recursively and re-create it as empty.'''
    if exists(d) and purge:
        rmtree(d)
    makedirs(d)  # , exist_ok=True

# This remaps images to AWE ToolBox proper form!
def pack2folders(src_path, dest_path, items):
    '''Pack images from src_path by their IDs to respective 
    folders into dest_path. Items represent ids.'''
    reset_directory(dest_path)

    for item in items:
        #ident, sess, num = item.split('_')

        #id_path = os.path.join(dest_path, ident)
        #reset_directory(id_path)

        #print " NAME: ", item


        img = imread(os.path.join(src_path, item))

        #print "IMG SHAPE: ", img.shape, " NAME: ", item


        try:
            imsave(os.path.join(dest_path, item), img)

        except ValueError:


            if img.shape[2] is 2: # if only two channels
               # img = img.flatten()
                img = np.swapaxes(np.array([img[:, :, 0]]), 0, 1).swapaxes(1, 2) #.shape
                img = np.repeat(img, 3, axis=2)
                print img.shape

            else: 
                def rgb2gray(rgb):
                    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
                img = rgb2gray(img)

            imsave(os.path.join(dest_path, item), img)
            print img.shape

        #imsave(os.path.join(id_path, item), img)
        


def pack2foldersAWET(src_path, dest_path, items):
    '''Pack images from src_path by their IDs to respective 
    folders into dest_path. Items represent ids.'''
    reset_directory(dest_path)

    for item in items:
        #print item
        try:
            ident, sess, num = item.split('_')
        except ValueError:
            print item

        try:
            ident = int(item.split('_')[0])
        except ValueError:
            # handle CK+ DB: ValueError: invalid literal for int() with base 10: 'S053'
            ident = int(item.split('_')[0][1:])

        id_path = os.path.join(dest_path, '{id:03d}/'.format(id=ident))
        reset_directory(id_path)

        img = imread(os.path.join(src_path, item))

        try:
            imsave(os.path.join(id_path, item), img)

        except ValueError:

            if img.shape[2] is 2: # if only two channels
               # img = img.flatten()
                img = np.swapaxes(np.array([img[:, :, 0]]), 0, 1).swapaxes(1, 2) #.shape
                img = np.repeat(img, 3, axis=2)
                print img.shape

            else: 
                def rgb2gray(rgb):
                    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
                img = rgb2gray(img)

            imsave(os.path.join(id_path, item), img)


        

if __name__ == "__main__":


    ''' process and prepare folds from xm2vts dataset
    # assemble gallery
    '''
    src_path = "./DB/xm2vts/crop/"
    src_path_aam = "./DB/xm2vts/aam/" # to make sure all ids exists in aam
    fold_path = "./folds/"
    fold_path_aam = "./folds_aam/"
    awet_path = "./awet/"
    awet_path_aam = "./awet_aam/"

    filelist = read_files(path=src_path_aam)
    ids = extractIDs(filelist)
    folds = sample_folds(ids, num_folds=5, num_ids=50)
    #print type(folds)

    for i, l in enumerate(folds):
        
        gallery = assemble_gallery(l, 1)
        probes = assemble_gallery(l, 2)

        print type(gallery)

        write2file("fold_gallery_%d.txt" % (i+1), gallery)
        write2file("fold_probes_%d.txt" % (i+1), probes)

        pack2folders(src_path, os.path.join(fold_path, "fold_gallery_%d" % (i+1)), gallery)
        pack2folders(src_path, os.path.join(fold_path, "fold_probes_%d" % (i+1)), probes)

        pack2folders(src_path_aam, os.path.join(fold_path_aam, "fold_gallery_%d" % (i+1)), gallery)
        pack2folders(src_path_aam, os.path.join(fold_path_aam, "fold_probes_%d" % (i+1)), probes)

        pack2foldersAWET(src_path, os.path.join(awet_path, "fold_gallery_%d" % (i+1)), gallery)
        pack2foldersAWET(src_path, os.path.join(awet_path, "fold_probes_%d" % (i+1)), probes)

        pack2foldersAWET(src_path_aam, os.path.join(awet_path_aam, "fold_gallery_%d" % (i+1)), gallery)
        pack2foldersAWET(src_path_aam, os.path.join(awet_path_aam, "fold_probes_%d" % (i+1)), probes)
    #ids

    #for fname in filelist:
