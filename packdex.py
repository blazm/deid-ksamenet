import os
from os.path import join, exists, split
from os import makedirs, listdir
from shutil import rmtree

import numpy as np

from scipy.misc import imread, imresize, imsave

from A_pick_samples import read_files, write2file #, reset_directory

from A_emo_pick_samples import pack21folderCKPLUS #, pack2foldersCKPLUS

from B_k_same_net import makeSquare

np.random.seed(42)

def reset_directory(d, purge=True):
    '''Remove directory recursively and re-create it as empty.'''
    if exists(d) and purge:
        rmtree(d)
    if not exists(d):
        makedirs(d)  # , exist_ok=True


def readAllFiles(src_path):
    all_files = []
    for root, subdirs, files in os.walk(src_path):
        for filename in files:
            if filename.endswith('png'):
                all_files.append(filename)
        
    return all_files

def pack2foldersCKPLUS(src_path, dst_path, items):
    '''Pack images from src_path by their IDs to respective 
    folders into dst_path. Items represent ids.'''
    reset_directory(dst_path)

    for item in items:
        ident, emo, num = item.split('.')[0].split('_')

        id_path = join(dst_path, ident, emo)
        reset_directory(id_path, purge=False)

        img = imread(join(src_path, item))

        try:
            imsave(join(id_path, item), img)

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

            imsave(join(id_path, item), img)
            
        #imsave(join(id_path, item), img)

def remapRafd(items):

    emotionsStr = {'neutral': 0, 'angry': 1, 'contemptuous': 2, 'disgusted': 3, 'fearful': 4, 'happy': 5, 'sad': 6, 'surprised': 7}
   
    remapped = []
    emo_labels = []
    for item in items:

        item, ext = item.split('.')

        db_label, id_, ethnicity, gender, emotion, pose = item.split('_')

        sess = emotionsStr[emotion]

        print int(id_)+600, sess
        ext = 'png'
        new_name = 'S{:03d}_{:03d}_{:08d}.{}'.format(int(id_)+600, sess+1, 1, ext) # ids from 600 onward are rafd
        label = sess # same notation
        
        remapped.append(new_name)
        emo_labels.append(sess)

    return remapped, emo_labels


def pack2foldersRaFD(src_path, dst_path, dst_path_label, all_images, new_files, labels):

    reset_directory(dst_path)
    reset_directory(dst_path_label)

    for old_name, new_name, label in zip(all_images, new_files, labels):

        ident, sess, num = new_name.split('.')[0].split('_')

        id_path = join(dst_path, ident, sess)
        reset_directory(id_path, purge=False)
        
        # TODO: repack labels to Emotions, images to Images
        img = imread(join(src_path, old_name))

        # TODO: resize
        #img = makeSquare(img, newdim=420, yoffset=40)
        img = makeSquare(img, newdim=380, yoffset=40)

        #img = imresize

        h, w, ch = img.shape
        img = imresize(img, (320, 320, ch), interp='bicubic')

        #try:
        imsave(join(id_path, new_name), img)

        # save labels
        #label_path = join(dst_path_label, ident, sess)
        #reset_directory(label_path, purge=False)
        #label_name = new_name.split('.')[0] + '_emotion.txt'
        #write2file(join(label_path, label_name), ['%1.7f' %(label)])


if __name__ == "__main__":

    doJoin = False
    doPackBack = False

    doPackRaFD = True
  
    #src_path = "./DB/ck+/cohn-kanade-images/"
    #max_emotions = selectMaxEmotions(src_path) # useless
    #dst_path = "./DB/ck+/max-emotion-images/"
    #pack2foldersCKPLUS(src_path, dst_path, max_emotions) # pack to separate folders
    
    # pack to single folder
    src_path = "./DB/ck+/cohn-kanade-images/"
    all_images = readAllFiles(src_path) 

    if doJoin:
        src_path = './DB/ck+/cohn-kanade-images'
        dst_path = './DB/ck+/cohn-kanade-images-joined'
        pack21folderCKPLUS(src_path, dst_path, all_images)


    # detect and crop faces with MTCNN


    # pack back to original structure (to train DeX)
    if doPackBack:
        src_path = './DB/ck+/cohn-kanade-images-detected'
        dst_path = './DB/ck+/cohn-kanade-images-detected-structured'
        pack2foldersCKPLUS(src_path, dst_path, all_images)

    #src_path = './DB/ck+/crop-emotion'
    #dst_path = './DB/ck+/crop-structured'

    if doPackRaFD:
        src_path = '../de-id/DB/rafd2-frontal/'
        dst_path = './DB/rafd/dex-packed/squared/Images/'
        dst_path_lbl = './DB/rafd/dex-packed/squared/Emotion/'


        all_images = read_files(src_path)
        new_files, labels = remapRafd(all_images)

        #print new_files, labels
        #exit()



        pack2foldersRaFD(src_path, dst_path, dst_path_lbl, all_images, new_files, labels)

        
    #pack2folders(src_path, dst_path, items)