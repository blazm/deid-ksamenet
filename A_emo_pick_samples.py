from A_pick_samples import read_files, reset_directory

from A_pick_samples import extractIDs, sample_folds, assemble_gallery, write2file, pack2foldersAWET, pack2folders

from scipy.misc import imread, imresize, imsave
from os.path import join, exists, split
import numpy as np

np.random.seed(42)

#WARNING CK+ has the following in the User Agreement: 
#if I reproduce images in electronic or print media, to use only those from the following subjects and include notice of copyright ((c)Jeffrey Cohn).
#S52, S55, S74, S106, S111, S113, S121, S124, S125, S130, S132

#def emotionToString(int_number):
#    emotions = {'neutral': 0, 'anger': 1, 'contempt': 2, 'disgust': 3, 'fear': 4, 'happy': 5, 'sadness': 6, 'surprise': 7}
#    return emotions[int_number]

def pack2foldersCKPLUS(src_path, dest_path, items):
    '''Pack images from src_path by their IDs to respective 
    folders into dest_path. Items represent ids.'''
    reset_directory(dest_path)

    for item in items:
        ident, emo, num = item.split('.')[0].split('_')

        id_path = join(dest_path, ident, emo)
        reset_directory(id_path)

        img = imread(join(src_path, ident, emo, item))
        imsave(join(id_path, item), img)

def pack21folderCKPLUS(src_path, dest_path, items):
    '''Pack images from src_path by their IDs to respective 
    folders into dest_path. Items represent ids.'''
    reset_directory(dest_path)

    for item in items:
        ident, sess, num = item.split('.')[0].split('_')

        #id_path = join(dest_path, ident, emo)
        #reset_directory(id_path)

        img = imread(join(src_path, ident, sess, item))
        imsave(join(dest_path, item), img)

def readEmotions(src_path):

    ck_subjects = read_files(src_path, isDir=True)
    emotions = {}

    for subject in ck_subjects:

        sessions = read_files(join(src_path, subject), True)

        for sess in sessions:

            annotations = read_files(join(src_path, subject, sess))
            if len(annotations) == 1:
                #print annotations
                filename = annotations[0]
                #print filename
                with open(join(src_path, subject, sess, filename), "r") as f: 
                    lines = f.readlines()
                    #print lines
                    emotion = (int)((float)(lines[0].strip()))-1 #
                    emotions['_'.join(filename.split('.')[0].split('_')[:-1])] = emotion

                #print '_'.join(filename.split('.')[0].split('_')[:-1]), emotions['_'.join(filename.split('.')[0].split('_')[:-1])]

    return emotions


def selectMaxEmotions(src_path):
    reset_directory(dst_path)

    ck_subjects = read_files(src_path)
    print ck_subjects

    max_emotions = []

    for subject in ck_subjects:

        sessions = read_files(join(src_path, subject))
        #print emotions
       
        for sess in sessions:

            images = read_files(join(src_path, subject, sess))

#            print images

            images = sorted(images)
            max_emotion = images[-1]

            max_emotions.append(max_emotion)

    return max_emotions 


def sample_folds_ckplus(filelist, emo_gtru, num_folds, num_ids): 
    '''Perform random sampling without replacing and return num_folds of num_ids size.'''
    filelist = set(filelist)
    folds = []

    # separate filelist to the groups with the same ids
    id_groups = {} # filenames by id

    for item in filelist:

        id_ = int(item.split('_')[0][1:])
        if id_ in id_groups:
            id_groups[id_].append(item)
        else:
            id_groups[id_] = [item]

    ids = id_groups.keys()

    # make sure that emotion annotation label is present in the DB
    for id_ in ids:
        filenames = id_groups[id_]
        filenames = [name for name in filenames if name.split('.')[0] in emo_gtru]  

        if len(filenames) > 0:  
            id_groups[id_] = filenames
        else:
            id_groups.pop(id_, None)

    ids = id_groups.keys()

    for i in range(num_folds):
        
        fold = []
        id_selection = np.random.choice(list(ids), num_ids, replace=False)
        
        for id_ in id_selection:
            filenames = id_groups[id_]
            filename = np.random.choice(filenames, 1, replace=False)[0]
            fold.append(filename)


        folds.append(list(fold))

    return folds

if __name__ == "__main__":

    ''' process and prepare ck+ dataset
        # pick images with max emotion
        # put them in one folder
        # detect faces and crop
        # generate folds from folder
        # repeat protocol experiments
    '''

    emo_path = "./DB/ck+/Emotion/" # read real emotions from annotatios
    emo_gtru = readEmotions(emo_path)
    
    #src_path = "./DB/ck+/cohn-kanade-images/"
    #max_emotions = selectMaxEmotions(src_path) # useless
    #dst_path = "./DB/ck+/max-emotion-images/"
    #pack2foldersCKPLUS(src_path, dst_path, max_emotions) # pack to separate folders
    
    # pack to single folder
    #pack21folderCKPLUS(src_path, dst_path, max_emotions)

    # detect faces with MTCNN detector
    # DONE!

    src_path = "./DB/ck+/crop-emotion/"
    fold_path = "./emo_folds/original/"
    awet_path = "./emo_awet/"

    # generate folds?

    filelist = read_files(path=src_path)
    #ids = extractIDs(filelist)
    #print len(set(ids)) 
    #folds = sample_folds(filelist, num_folds=5, num_ids=50) # since we dont have enough ids for all folds, but emotions count!
    folds = sample_folds_ckplus(filelist, emo_gtru, num_folds=5, num_ids=50)
    #print folds
    
    for i, l in enumerate(folds):
        
        gallery = assemble_gallery(l, 1, db='ck+')
        #probes = assemble_gallery(l, 2, db='ck+')

        write2file("emo_fold_gallery_%d.txt" % (i+1), gallery)
        #write2file("emo_fold_probes_%d.txt" % (i+1), probes)

        pack2folders(src_path, join(fold_path, "fold_gallery_%d" % (i+1)), gallery)
        #pack2folders(src_path, join(fold_path, "fold_probes_%d" % (i+1)), probes)

        #pack2foldersAWET(src_path, join(awet_path, "fold_gallery_%d" % (i+1)), gallery)
        #pack2foldersAWET(src_path, join(awet_path, "fold_probes_%d" % (i+1)), probes)
    #ids