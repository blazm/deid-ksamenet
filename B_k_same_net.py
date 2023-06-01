

from A_pick_samples import read_files, reset_directory
from C_extract_features import loadFold, loadImages #, idsFromFold

from E_compare_features import cosine

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.misc import imresize, imsave

#from collections import Counter 
from clustering.equal_groups import EqualGroupsKMeans # TODO: if time

import networkx as nx
import numpy as np
import time
import os


def clusterImagesEqualGroups(src_path, image_list, k=2, imsize=None):
    n_clusters = len(image_list) / k
    
    X = loadImages(src_path, image_list, imsize)

    # for now perform clustering on 
    num, h, w, ch = X.shape
    X = np.reshape(X, (num, h*w*ch)) # reshape pixels to vector form

    # TODO: extract features in final version!
    #X = extractFeatures(X) # TODO: clustering can be done by evaluating some extracted features, to speedup the process

    clf = EqualGroupsKMeans(n_clusters, random_state=0)
    clf.fit(X)
    clf.labels_
    
    print "CLF.labels_", clf.labels_

    #print "CLF.centers_", clf.cluster_centers_
    #predict = clf.predict([[0, 0], [4, 4]])
    #return clf.labels_

    clusters = {}

    #for label in clf.labels_:
    for name, label in zip(image_list, clf.labels_):
        if label in clusters:
            clusters[label].append(name)
        else:
            clusters[label] = [name]

    return clusters

def saveClusters(clusters, filename):

    with open(filename, "w") as f: 
        for k in clusters.keys():
            f.write(' '.join(clusters[k]) + '\n')


def remapEmotions(src_path, image_list):
    # if DeXpression predicts emotions wrong,
    # then filenames in clusters are not correct,
    # this remaps filenames to actual filenames on disk
    actual_image_list = read_files(src_path)
    new_image_list = []
    #print len(actual_image_list)
    #print len(image_list)

    for name in image_list:
        parts = name.split('_')

        if len(parts) == 3: 
            return image_list # no emotions present

        kept_name = '_'.join(parts[:-1]) # remove emotion
        
        for real_name in actual_image_list:
            if kept_name in real_name:
                new_image_list.append(real_name)
                break

    return new_image_list


def loadClusteredImages(src_path, clusters):

    clustered_images = []
    for k in clusters.keys():
        image_list = clusters[k]

        image_list = remapEmotions(src_path, image_list)
        print(image_list)
        images = loadImages(src_path, image_list)
        clustered_images.append(images)
    return clustered_images


def loadClusters(filename):
    
    clusters = {}
    with open(filename, "r") as f: 
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            cluster = line.split(' ')

            clusters[i] = cluster

    return clusters


def clusterImagesRandomly(src_path, image_list, k=2, mode='pixelwise'):
    # TODO
    pass


def clusterImagesDummy(src_path, image_list, k=2):
    # dummy clustering (just sequentially zip it)
    print image_list
    total = len(image_list)

    clusters = total / k
    remaining = total % k # TODO: handle rest of the images

    ix = 0

    cluster_dict = {}

    clustered = [];
    for i in range(clusters):
        cluster = []
        for j in range(k):
            cluster.append(image_list[i*k + j]) # correct indexing
            ix += 1
        #print len(cluster)
        cluster_dict[i] = cluster

    if ix < total:
        cluster_dict[clusters] = image_list[ix:]



    #print clustered
    #print "TOT: ", total, " k:", k, "CLU: ", clusters, " REM:", remaining 

    return cluster_dict # 2D array [[cluster][cluster][cluster]]

def clusterFeatures(path, image_list, k):
    pass


def clusterImages(path, image_list, k):
    # TODO: real clustering
    total = len(image_list)
    clusters = total / k

    images = loadImages(path, image_list)

    vectorized = []

    for img in images:
        #print(img.shape)
        # TODO: here handle vectors and images! (to cluster feats and images)

        h, w, ch = img.shape
        img = imresize(img, (h/2, w/2, ch), interp='bicubic')
        imgv = img.flatten()
        vectorized.append(imgv)

    vectorized = np.array(vectorized)
    km = KMeans(n_clusters = clusters, random_state=0)
    km.fit(vectorized)
    labels = km.labels_

    print len(list(set(labels)))
    print Counter(labels)

    #for name, img in zip(image_list, images):
    # useless!
    #(C, M, f) = constrained_kmeans(vectorized, [k for k in range(clusters)], maxiter=500)

    return labels


def filterProxy(proxy_list, emotion='neutral'):
    new_list = []

    males = 0; females = 0; caucasians = 0; moroccans = 0

    for item in proxy_list:
        if emotion in item:

            if 'female' in item:
                females += 1
            elif 'male' in item:
                males += 1
            
            if 'Caucasian' in item:
                caucasians += 1
            if 'Moroccan' in item:
                moroccans += 1

            new_list.append(item)

    print "M:", males, "F:", females, "Ca:", caucasians, "Mo:", moroccans

    return new_list

def extractProxyIDs(clustered_images, sorted_images):

    #print sorted_images
    #print clustered_images

    # dummy clustering (just sequentially zip it)
    #print image_list
    clusters_old = []
    clusters = {}
    for k in clustered_images.keys():

        cluster = clustered_images[k]

        if type(cluster) is list:
            ids = []
            for item in cluster:
                #print item
                if type(item) is str:
                    #id_ = int(item.split('_')[1])-1 # Identities are 1-indexed
                    id_ = sorted_images.index(item)
                    #clusters.append(id_)
                else: #if type(cluster) is int:
                    id_ = item

                ids.append(id_)
            clusters[k] = ids

        elif type(cluster) is str:
            id_ = int(cluster.split('_')[1])-1 # Identities are 1-indexed
            clusters[k] = [id_]
        elif type(cluster) is int:
            id_ = cluster

    #print clustered
    #print "TOT: ", total, " k:", k, "CLU: ", clusters, " REM:", remaining 

    return clusters

def mapRandomly(probes, proxys):

    #np.ra

    print len(probes), len(proxys)

    pass

def mapSimilar(probes, proxys):
    pass


def makeSquare(image, newdim=320.0, yoffset=0):
    h, w, ch = image.shape
    if h > w:
        image = image[ (h - w)/2: w + (h - w)/2, :, :]
    elif w > h:
        image = image[ :, (w - h)/2: h + (w - h)/2, :]    
    
    # crop from center
    if newdim < image.shape[0]:
        image = image[ w/2 - newdim/2 - yoffset: w/2 + newdim/2 - yoffset,  w/2 - newdim/2: w/2 + newdim/2, :]
    else:
        print("New dimension larger than original!")

    return image

if __name__ == "__main__":
    from keras import backend as K
    K.set_image_dim_ordering('tf')
    from Generator import Generator

    deconv_layer = 6 # 5 or 6
    model_name = 'FaceGen.RaFD.model.d{}.adam'.format(deconv_layer)
#    model_path = '../de-id/generator/output/{}.h5'.format(model_name)
    model_path = './models/{}.h5'.format(model_name) # locally stored models


    gen = Generator(model_path, deconv_layer=deconv_layer)

    do_emotions = True
    emotions = {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}

    # load probe set from fold (gallery stays the same)
    path = "./DB/crop/"
    fold_path = "./folds/"
    awet_path = "./awet/"

    if do_emotions:
        fold_path = "./emo_folds/"
        awet_path = "./emo_awet/"

    num_folds = 5

    proxy_path = "../de-id/DB/rafd2-frontal/"
    proxys = read_files(proxy_path)
    proxys = filterProxy(proxys)
    proxys = range(len(proxys))
    print proxys
    print(sorted(extractProxyIDs(proxys)))

    # TODO: for various k! 2, 3, 4, 5, ... 10?
    k = 1
    for k in range(1, 11): # [2, .., 10]
        
        # TODO: cluster proxy
        #clustered_proxys = clusterImagesDummy(proxys, k)
        clustered_proxys = clusterImagesDummy(proxys, k)
        clustered_proxys = extractProxyIDs(clustered_proxys)

        print clustered_proxys
        print len(proxys)
        
        
        #proxy_imgs = loadImages(proxy_path, proxys)
        #proxy_imgs = clusterImages(proxy_path, proxys, k)

        for fold in range(num_folds):

            path = os.path.join(fold_path, './fold_probes_{}_k={}/'.format(fold+1, k))
            reset_directory(path, purge=True)

            #probes = loadFold("fold_probes_%d.txt" % (fold+1))
            #print "PROBES 1: ", probes
            probe_path = os.path.join(fold_path, 'fold_probes_{}/'.format(fold+1))
            print fold_path, ":", probe_path
            print os.path.join(fold_path, ('fold_probes_{}/'.format(fold+1)))
            
            probes = read_files(probe_path) # read directly from fold folder
            # ids = idsFromFold(probes);
            #print "PROBES 2: ", probes

            # TODO: cluster by pixels or features
            #probe_images = loadImages(path, probes)
            # optional probe_descriptors = extractFeatures(probe_images)
            #clustered_probes = clusterImages(probe_images)

            clustered_probes = clusterImagesDummy(probes, k)
            print "CLUSTERED PROBES LEN:", len(clustered_probes)
            # TODO
            #mapIndices = mapRandomly(clustered_probes, clustered_proxys)

            # TODO: map probe and proxy clusters
            # connect clustered probes with clustered proxy set?

            #print clustered_proxys
            for i, probe_cluster in enumerate(clustered_probes):

                proxy_cluster = clustered_proxys[i]     
                print "PROBE CLUSTER LEN:", len(probe_cluster)    
                #print proxy_cluster
                # TODO: estimate emotion from original image
                #emotion = 
                emotion = 'neutral' # 'happy' #'neutral'

                # generate deidentified 
                image = gen.generate(proxy_cluster, emotion)

                # crop and resize img
                # make it square, crop from center
                image = makeSquare(image, newdim=320)
                

                # save generated result for all ids
                
                
                for item in probe_cluster:

                    if do_emotions:
                        # TODO: deidentify using emotions for each item
                        #print item # read emotion from probe names
                        emo_id = int(item.split('_')[1])
                        try:
                            emotion = emotions[emo_id-1] #'happy' # 
                        except:
                            print item, " has unknown emotion, resetting to 'neutral'!"
                            emotion = 'neutral'

                        image = gen.generate(proxy_cluster, emotion)
                        image = makeSquare(image, newdim=320)

                    try:
                        _id = int(item.split('_')[0])
                    except ValueError:
                        # handle CK+ DB: ValueError: invalid literal for int() with base 10: 'S053'
                        _id = int(item.split('_')[0][1:])

                    #_id = int(item.split('_')[0])
                #    path = './fold_probes_{}_k={}/{num:03d}/'.format(fold+1, k, num=_id)
                    pathAWET =  os.path.join(awet_path,'fold_probes_{}_k={}/{num:03d}/'.format(fold+1, k, num=_id))
                    reset_directory(pathAWET, purge=True)

                    filename = '{num:03d}_{emo:s}_1.ppm'.format(num=_id, emo=emotion)
                    imsave(os.path.join(path, filename), image)
                    imsave(os.path.join(pathAWET, filename), image)


    # cluster probe to equal sized clusters of k

    # save cluster pairs to txt


