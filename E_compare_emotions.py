
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, exists, split, isfile, isdir

from A_pick_samples import read_files, reset_directory
from A_emo_pick_samples import readEmotions

emotionsStr = {'neutral': 0, 'anger': 1, 'contempt': 2, 'disgust': 3, 'fear': 4, 'happy': 5, 'sadness': 6, 'surprise': 7}
emotionsInt = {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}

def generate_conf_matrix(true, pred):
   
    s = len(emotionsStr.keys())
    matrix = np.zeros((s, s))

    for t, p in zip(true, pred):
        x = emotionsStr[t]
        y = emotionsStr[p]
        matrix[x,y] += 1

    return matrix

def get1Emotion(file_names):
    pred = []

    for item in file_names:
        vals = item.split('.')[0].split('_')
        pred.append(vals[-1])

    return pred

def getEmotions(file_names):

    true = []
    pred = []

    for item in file_names:
        vals = item.split('.')[0].split('_')
        #print vals[-2:]
        true.append(vals[-2])
        pred.append(vals[-1])

    return true, pred

if __name__ == '__main__':


    #gallery_path = "./emo_folds/fold_gallery_1"
    #probe_path = "./emo_folds/fold_gallery_1"

    emo_path = "./DB/ck+/Emotion/" # read real emotions from annotatios
    emo_gtru = readEmotions(emo_path)
    #print emo_gtru
    # TODO: validate original and dexpression predictions
    #exit()


    src_path = "./emo_folds_predicted/original/"
    src_dirs = read_files(src_path, isDir=True)
    src_dirs = [d for d in src_dirs if "gallery" in d] # first only preddict for gallery sets

    s = len(emotionsStr.keys())
    valid_arr = np.zeros((s, s))
    
    for src_dir in src_dirs:
        gallery = read_files(join(src_path, src_dir))
        pred = get1Emotion(gallery)
        
        anno = []
        for name in gallery:
            #try: 
            n = '_'.join(name.split('.')[0].split('_')[:-1])
            try:
                e = emo_gtru[n]+1
                #print "ID OK!"
            except:
                print "\tUnknown ID: ", n
                e = 0
            anno.append(emotionsInt[e])

        #valid_arr = np.zeros((s, s))
        #print gallery
        print anno
        #print pred

        valid_arr += generate_conf_matrix(anno, pred)

    valid_arr = valid_arr[1:, 1:] # ignore neutral axis

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(valid_arr), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = np.array(valid_arr).shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(valid_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    #alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabet = 'nACDFHSS'
    alphabet = ['ANG', 'CON', 'DIS', 'FEA', 'HAP', 'SAD', 'SUR'] # 'neu'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('valid-mat-emo.pdf', format='pdf', bbox_inches='tight')



    for k in range(2, 11):

        #src_path = "./emo_folds_predicted/k_same_net_k=%d/" %(k) # fold_gallery_1
        src_path = "./emo_folds_predicted/k_same_net_k=%d/" %(k) # fold_gallery_1

        src_dirs = read_files(src_path, isDir=True)
        src_dirs = [d for d in src_dirs if "gallery" in d] # first only preddict for gallery sets

        s = len(emotionsStr.keys())
        conf_arr = np.zeros((s, s))

        
        for src_dir in src_dirs:
            #probe = read_files(probe_path)
            gallery = read_files(join(src_path, src_dir))

            true, pred = getEmotions(gallery)

            #print true, pred

            num_true = len(set(true))
            num_pred = len(set(pred))

            print "k: ", k, " gallery: ", src_dir, " Length: ", len(gallery), " Unique GT: ", num_true, " Unique Pred: ", num_pred

           
                #anno.append(emotions[emo_gtru['_'.join(name.split('.')[0].split('_')[:-2])]])
                #except:
                #anno.append("neutral")

            #anno = [emo_gtru['_'.join(name.split('.')[0].split('_')[:-2])] for name in gallery]

            #print gallery
            #print anno 
            anno = []
            for name in gallery:
                #try: 
                n = '_'.join(name.split('.')[0].split('_')[:-2])
                try:
                    e = emo_gtru[n]+1
                    #print "ID OK!"
                except:
                    print "\tUnknown ID: ", n
                    e = 0
                anno.append(emotionsInt[e])

            
            conf_arr_t = generate_conf_matrix(true, pred) # per fold
            conf_arr += conf_arr_t  # total folds

            

            '''
                conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], 
                            [3,31,0,0,0,0,0,0,0,0,0], 
                            [0,4,41,0,0,0,0,0,0,0,1], 
                            [0,1,0,30,0,6,0,0,0,0,1], 
                            [0,0,0,0,38,10,0,0,0,0,0], 
                            [0,0,0,3,1,39,0,0,0,0,4], 
                            [0,2,2,0,4,1,31,0,0,0,2],
                            [0,1,0,0,0,0,0,36,0,2,0], 
                            [0,0,0,0,0,0,1,5,37,5,1], 
                            [3,0,0,0,0,0,0,0,0,39,0], 
                            [0,0,0,0,0,0,0,0,0,0,38]]
            '''

            # TODO: normalize
            '''
            norm_conf = []
            for i in conf_arr:
                a = 0
                tmp_arr = []
                a = sum(i, 0)
                for j in i:
                    tmp_arr.append(float(j)/float(a))
                norm_conf.append(tmp_arr)
            '''

        conf_arr = conf_arr[1:, 1:] # ignore neutral axis

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(conf_arr), cmap=plt.cm.jet, #'GnBu', #plt.cm.bones,  # jet
                        interpolation='nearest')

        width, height = np.array(conf_arr).shape

        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center') # , color='lightgray'

        cb = fig.colorbar(res)
        #alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        alphabet = 'nACDFHSS'
        alphabet = ['ANG', 'CON', 'DIS', 'FEA', 'HAP', 'SAD', 'SUR'] #'neu', 
        plt.xticks(range(width), alphabet[:width]) # , rotation=90
        plt.yticks(range(height), alphabet[:height])
        plt.savefig('conf-mat-emo-k=%d.pdf' %(k), format='pdf', bbox_inches='tight')

    
    