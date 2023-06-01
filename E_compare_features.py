
from numpy import genfromtxt
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

the_font_size = 16
matplotlib.rcParams.update({'font.size': the_font_size})

#from matplotlib import rc
# activate latex text rendering
#rc('text', usetex=True)
# ERR: RuntimeError: dvipng was not able to process the following file:

from A_pick_samples import write2file

def cosine(a, b):
    return np.dot(a,b.T)/np.linalg.norm(a)/np.linalg.norm(b)

def gen_dist_mtx(gallery_idents, gallery_descriptors, probe_idents, probe_descriptors):

    np.testing.assert_equal(gallery_idents, probe_idents) # IDs must be the same in both sets

    dist_matrix = np.empty((gallery_idents.shape[0], probe_idents.shape[0]))

    for gix in range(gallery_feats.shape[0]):
        for pix in range(probe_feats.shape[0]):

            score = cosine(gallery_descriptors[gix, :], probe_descriptors[pix, :])
            dist_matrix[gix, pix] = score

            #dist_matrix[gix, pix] = int(gallery_idents[gix] == probe_idents[pix])

    return dist_matrix


def rank(n, dist_matrix, probe_idents, gallery_idents):

    rank = 0
    for i in range(probe_idents.shape[0]):

        gallery_scores = dist_matrix[:, i] # pick all gallery values under i-th probe
        probe_id = probe_idents[i]
        
        # sort
        sorted_ixs = np.argsort(gallery_scores)

        # pick n bests
        n_ixs = sorted_ixs[-n:]

        # check if class from n is in gallery?
        rank_n = np.sum([1 for ix in n_ixs if gallery_idents[ix] == probe_id])
        #print rank_n

        rank += rank_n
        
    # return total stats (percentage)
    return float(rank)/probe_idents.shape[0]

def generate_CMC_curve(dist_matrix, probe_idents, gallery_idents):

    x = []
    y = []    
    for i in range(gallery_idents.shape[0]):
        rank_i = rank(i+1, dist_matrix, probe_idents, gallery_idents)
        x.append(i+1)
        y.append(rank_i)

    return x, y

def plot_RNKK_curve(x, y, label, xlim, ylim, std=None, ix=0, xlim_ext=0.0):
    styles = ['--', '-.', ':', '-', '--', '-.']
    solid_styles = ['-'] * 7 #, '-.', ':', '-', '--', '-.']
    colors = ['b', 'g', 'r', 'y', 'c', 'm', 'k']
    err_colors = ['skyblue', 'lightgreen', 'tomato', 'lightyellow', 'aqua', 'violet', 'grey']
   
   
    # TODO: grid!
    # TODO: curve width!

    #plt.legend(loc='upper left')
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    
    #if std is None:
    plt.plot(x, y, colors[ix]+solid_styles[ix], label=label, linewidth=2.0)
    # x, y, e is error
    #else:
    #    plt.errorbar(x, y, yerr=std, fmt=colors[ix]+styles[ix], label=label, linewidth=2.0)

    # alternative for std
    if std is not None:
        #plt.rcParams['hatch.color'] = colors[ix] # works only for one instance!
        #plt.fill_between(x, y-std, y+std, alpha=0.75, facecolor='none', hatch="X", edgecolor=colors[ix], linewidth=0.0) # , edgecolor='#CC4F1B', facecolor='#FF9848'
        plt.plot(x, y-std, colors[ix]+styles[ix], linewidth=0.5)
        plt.plot(x, y+std, colors[ix]+styles[ix], linewidth=0.5)

    plt.xlim([2 - xlim_ext, xlim + xlim_ext])
    plt.ylim([0.0, ylim])

    plt.ylabel('Identification rate')
    plt.xlabel('k', style='italic')
    plt.legend(loc='upper right', fontsize=the_font_size)

def plot_BAR_chart(x, y, label, xlim, ylim, std=None, ix=0, maxIx = 5):

    styles = ['--', '-.', ':', '-', '--', '-.']
    solid_styles = ['-'] * 6 #, '-.', ':', '-', '--', '-.']
    colors = ['b', 'g', 'r', 'y', 'c', 'm']
    err_colors = ['skyblue', 'lightgreen', 'tomato', 'lightyellow', 'aqua', 'violet']
   

    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    #fig, ax = plt.subplots()    
    ax = plt.gca() # get current axes

    ind = np.array(x)
    
    print "LEN X BAR: ", len(x)

    ticks = x
    ticks.append(x[-1]+1)


    width = 0.15

    rects = ax.bar(ind + width*ix - width*((maxIx-1)/2.), y, width, color=colors[ix], label=label, yerr=std)

    ax.set_ylabel('Identification rate')
   # ax.set_title('Scores by group and gender')
    ax.set_xlim(left=ticks[0], right=ticks[-1])
    #ax.set_xticklabels([str(k) for k in x])
    #ax.set_aspect('equal', 'datalim')
    ax.set_aspect('auto', 'datalim')


    # update and save legend
    if ix == 0:
        ax.legend([rects[0]], [label])
    else:
        h, l = ax.get_legend_handles_labels()
        #print h, l # TODO!
        h.append(rects[0])
        l.append(label)
        ax.legend(h, l)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height+0.04,
                    '%.3f' % float(height),
                    ha='center', va='bottom', rotation='vertical')

    autolabel(rects)



def plot_CMC_curve(x, y, label, xlim, ylim, std=None, ix=0):
    styles = ['--']*6 #, '-.', ':', '-', '--', '-.']
    solid_styles = ['-'] * 6 #, '-.', ':', '-', '--', '-.']
    colors = ['b', 'g', 'r', 'y', 'c', 'm']
    err_colors = ['skyblue', 'lightgreen', 'tomato', 'lightyellow', 'aqua', 'violet']
   
    #plt.legend(loc='upper left')
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    
    #if std is None:
    plt.plot(x, y, colors[ix]+solid_styles[ix], label=label, linewidth=2.0)
    # x, y, e is error
    #else:
    #    plt.errorbar(x, y, yerr=std, fmt=colors[ix]+styles[ix], label=label, linewidth=2.0)

    # alternative for std
    if std is not None:
        # , facecolor=err_colors[ix]
        #plt.rcParams['hatch.color'] = colors[ix] # works only for one instance!
        #plt.fill_between(x, y-std, y+std, alpha=0.75, facecolor='none', hatch="X", edgecolor=colors[ix], linewidth=0.0) # , edgecolor='#CC4F1B', facecolor='#FF9848'
        plt.plot(x, y-std, colors[ix]+styles[ix], linewidth=0.5)
        plt.plot(x, y+std, colors[ix]+styles[ix], linewidth=0.5)

    plt.xlim([1, xlim])
    plt.ylim([ylim, 1])

    plt.ylabel('Identification rate')
    plt.xlabel('Rank')
    plt.legend(loc='lower right', fontsize=the_font_size)

if __name__ == "__main__":

    num_folds = 5
    path = "./feats/"
    methods = os.listdir(path)
    methods = [m for m in methods if not "." in m ] # and not 'awet_' in m

    
    os.chdir(path)
    
    doValidate = True # validate all recognition models
    doRecogNet = True # recognize k-same-net using all recognition models
    doReidentify = not True # reidentify using two best recognition models

    if doValidate:
        fancy = {"alexnet":"AlexNet", "awet_lbp":"LBP", "awet_poem":"POEM", 
                "squeezenet":"SqueezeNet", "vggface":"VGG-Face", "inception":"InceptionV3"}

        f = plt.figure()
        x = None
        y = []
        text_table = []

        # identification validation (verification)
        for ix, method_name in enumerate(methods):
            print method_name
            for i in range(num_folds):
                #print i+1
                
                if "awet_" in method_name:
                    gallery_filename = "fold_gallery_{}-{}.csv".format(i+1, method_name)
                    probe_filename = "fold_probes_{}-{}.csv".format(i+1, method_name)
                else:
                    gallery_filename = "fold_gallery_{}.csv".format(i+1)
                    probe_filename = "fold_probes_{}.csv".format(i+1)

                gallery_path = os.path.join(method_name, gallery_filename)
                probe_path = os.path.join(method_name, probe_filename)


                gallery_feats = genfromtxt(gallery_path, delimiter=',', skip_header=("awet_" in method_name))
                probe_feats = genfromtxt(probe_path, delimiter=',', skip_header=("awet_" in method_name))

                print gallery_feats.shape
                print probe_feats.shape

                gallery_idents = gallery_feats[:, 0]
                gallery_descriptors = gallery_feats[:, 1:]

                probe_idents = probe_feats[:, 0]
                probe_descriptors = probe_feats[:, 1:]

                dist_matrix = gen_dist_mtx(gallery_idents, gallery_descriptors, probe_idents, probe_descriptors)

                print dist_matrix.shape

                # rank recognition
                x, iy = generate_CMC_curve(dist_matrix, probe_idents, gallery_idents)

                y.append(iy)
            
            x = np.array(x)
            #print y.shape, x.shape
            avgY = np.array(y).mean(axis=0)
            stdY = np.array(y).std(axis=0)


            #write2file("valid_mean_{}.txt".format(method_name), list(avgY))
            #write2file("valid_stdev_{}.txt".format(method_name), list(stdY))
            
            #for mean, std in zip(avgY, stdY):
            #s = "{} \\newline & {:0.3f} & {:0.3f} \\\\".format(fancy[method_name], avgY[0], stdY[0])
            s = "{}  &  ${:0.3f}\\pm {:0.3f}$ ".format(fancy[method_name], avgY[0], stdY[0])
            
            text_table.append(s)

            np.savetxt("valid_mean_{}.txt".format(method_name), avgY)   # X is an array
            np.savetxt("valid_stdev_{}.txt".format(method_name), stdY)   # X is an array
            
            #print stdY.shape, avgY.shape, x.shape
            #print avgY
            #print stdY

            plot_CMC_curve(x, avgY, fancy[method_name], gallery_idents.shape[0]/2, 0.75, None, ix) # stdY for None

        write2file("cmc-verif-valid.txt", text_table)

        f.set_size_inches(7.5, 6.0)
        f.savefig("cmc-verif-valid.pdf", bbox_inches='tight', dpi=50)
#    plt.show()

    if doRecogNet:
        fancy = {"alexnet":"AlexNet", "awet_lbp":"LBP", "awet_poem":"POEM", 
                "squeezenet":"SqueezeNet", "vggface":"VGG-Face", "inception":"InceptionV3", "theoretical":"Theoretical 1/k"}
        deid = 'k_same_net'

        f = plt.figure()
        x = None
        rank1 = {}
        text_table = []

        for ix, method_name in enumerate(methods):
            print method_name,

            for i in range(num_folds):
                print " fold: ", i+1,

                if "awet_" in method_name:
                    gallery_filename = "fold_gallery_{}-{}.csv".format(i+1, method_name)
                else:
                    gallery_filename = "fold_gallery_{}.csv".format(i+1)

                gallery_path = os.path.join(method_name, gallery_filename)
                gallery_feats = genfromtxt(gallery_path, delimiter=',', skip_header=("awet_" in method_name))
                
                gallery_idents = gallery_feats[:, 0]
                gallery_descriptors = gallery_feats[:, 1:]

                for k in range(2, 11):
                    #print " k=", k
                
                    if "awet_" in method_name:
                        probe_filename = "fold_probes_{}-{}.csv".format(i+1, method_name)
                    else:
                        probe_filename = "fold_probes_{}.csv".format(i+1)

                    probe_path = os.path.join(method_name, "{}_k={}".format(deid, k), probe_filename)
                    probe_feats = genfromtxt(probe_path, delimiter=',', skip_header=("awet_" in method_name))
                    
                    probe_idents = probe_feats[:, 0]
                    probe_descriptors = probe_feats[:, 1:]

                    dist_matrix = gen_dist_mtx(gallery_idents, gallery_descriptors, probe_idents, probe_descriptors)

                    x, iy = generate_CMC_curve(dist_matrix, probe_idents, gallery_idents)

                    if k not in rank1:
                        rank1[k] = [iy[0]]
                    else: 
                        rank1[k].append(iy[0])

                    #print len(iy)
            
            x = np.array([k for k in range(2, 11)])
            avgY = []
            stdY = []

            for k in range(2, 11):
                avergY = np.array(rank1[k]).mean(axis=0)
                stdevY = np.array(rank1[k]).std(axis=0)

                avgY.append(avergY)
                stdY.append(stdevY)

            avgY = np.array(avgY)
            stdY = np.array(stdY)

            #write2file("valid_mean_{}.txt".format(method_name), list(avgY))
            #write2file("valid_stdev_{}.txt".format(method_name), list(stdY))
            
            #for mean, std in zip(avgY, stdY):
            #s = "{} \\newline & {:0.3f} & {:0.3f} \\\\".format(fancy[method_name], avgY[0], stdY[0])
            s = "{}  &  ${:0.3f}\\pm {:0.3f}$ ".format(fancy[method_name], avgY[0], stdY[0])
            
            text_table.append(s)

            np.savetxt("valid_mean_{}.txt".format(method_name), avgY)   # X is an array
            np.savetxt("valid_stdev_{}.txt".format(method_name), stdY)   # X is an array
            
            #print stdY.shape, avgY.shape, x.shape
            #print avgY
            #print stdY

            plot_RNKK_curve(x, avgY, fancy[method_name], len(avgY), 0.5, None, ix) # stdY for None
    

        # plot formal theoretical line
        y = [1.0 / k for k in range(2, 12)]
        plot_RNKK_curve(range(2, 12), y, fancy["theoretical"], len(y), 0.5, None, 6)

        write2file("cmc-verif-deid.txt", text_table)

        f.set_size_inches(7.5, 6.0)
        f.savefig("cmc-verif-deid.pdf", bbox_inches='tight', dpi=50)

    if doReidentify:

        # temporary
        #the_font_size = 14
        #matplotlib.rcParams.update({'font.size': the_font_size})


        methods = ['inception', 'awet_poem'] # 

        naive = ['blur', 'pixelize']
        formal = ['k_same_M', 'k_same_net', 'k_same_pixel']
        fancy = {"blur":"Blurring", "pixelize":"Pixelization", "k_same_M":"k-Same-M", 
                "k_same_net":"k-Same-Net", "k_same_pixel":"k-Same-Pixel", "theoretical":"Theoretical 1/k"}
        # "$\it{k}$-Same-Net"

        rank1 = {}
        

        # deidenetified reidentification
        for ix, method_name in enumerate(methods):
            f = plt.figure()

            print method_name,

            for i in range(num_folds):
                print " fold: ", i+1,

                if "awet_" in method_name:
                    gallery_filename = "fold_gallery_{}-{}.csv".format(i+1, method_name)
                    gallery_filename_aam = "fold_gallery_aam_{}-{}.csv".format(i+1, method_name)
                else:
                    gallery_filename = "fold_gallery_{}.csv".format(i+1)
                    gallery_filename_aam = "fold_gallery_aam_{}.csv".format(i+1)

                gallery_path = os.path.join(method_name, gallery_filename)
                gallery_feats = genfromtxt(gallery_path, delimiter=',', skip_header=("awet_" in method_name))
                
                gallery_path_aam = os.path.join(method_name, gallery_filename_aam)
                gallery_feats_aam = genfromtxt(gallery_path, delimiter=',', skip_header=("awet_" in method_name))

                gallery_idents = gallery_feats[:, 0]
                gallery_descriptors = gallery_feats[:, 1:]

                gallery_descriptors_aam = gallery_feats_aam[:, 1:]

                for deid in naive:

                    if "awet_" in method_name:
                        probe_filename = "fold_probes_{}-{}.csv".format(i+1, method_name)
                    else:
                        probe_filename = "fold_probes_{}.csv".format(i+1)

                    probe_path = os.path.join(method_name, "{}".format(deid), probe_filename)
                    probe_feats = genfromtxt(probe_path, delimiter=',', skip_header=("awet_" in method_name))
                    
                    probe_idents = probe_feats[:, 0]
                    probe_descriptors = probe_feats[:, 1:]

                    dist_matrix = gen_dist_mtx(gallery_idents, gallery_descriptors, probe_idents, probe_descriptors)

                    _, iy = generate_CMC_curve(dist_matrix, probe_idents, gallery_idents)

                    # TODO: naive
                    # save only rank-1 recognition
                    if deid not in rank1:
                        rank1[(deid)] = [iy[0]]
                    else: 
                        rank1[(deid)].append(iy[0])


                for deid in formal:
                    
                    for k in range(2, 11):
                        print " k=", k
                    
                        if "awet_" in method_name:
                            probe_filename = "fold_probes_{}-{}.csv".format(i+1, method_name)
                        else:
                            probe_filename = "fold_probes_{}.csv".format(i+1)


                        # TODO: k_same_M needs separate gallery

                        probe_path = os.path.join(method_name, "{}_k={}".format(deid, k), probe_filename)
                        probe_feats = genfromtxt(probe_path, delimiter=',', skip_header=("awet_" in method_name))
                        
                        probe_idents = probe_feats[:, 0]
                        probe_descriptors = probe_feats[:, 1:]

                        if deid == "k_same_M":
                            print deid
                            gallery_descripto = gallery_descriptors_aam
                        else:
                            gallery_descripto = gallery_descriptors

                        dist_matrix = gen_dist_mtx(gallery_idents, gallery_descripto, probe_idents, probe_descriptors)

                        _, iy = generate_CMC_curve(dist_matrix, probe_idents, gallery_idents)

                        # save only rank-1 recognition
                        if (k, deid) not in rank1:
                            rank1[(k, deid)] = [iy[0]]
                        else: 
                            rank1[(k, deid)].append(iy[0])
            


            for ix, deid in enumerate(naive):
                
                avgY = np.array(rank1[(deid)]).mean(axis=0)
                stdY = np.array(rank1[(deid)]).std(axis=0)

                plot_BAR_chart(range(2, 11), [avgY]*len(range(2, 11)), fancy[deid], len(range(2, 12)), 0.5, [stdY]*len(range(2, 11)), ix+3)

            
            for ix, deid in enumerate(formal):
                y = []
                stdevY = []

                for k in range(2, 11):
                    avgY = np.array(rank1[(k, deid)]).mean(axis=0)
                    stdY = np.array(rank1[(k, deid)]).std(axis=0)

                    y.append(avgY)
                    stdevY.append(stdY)
                    # TODO: plot std

                y = np.array(y)
                stdevY = np.array(stdevY)

                print len(y)
                print len(stdevY)
                #write2file("reident_mean_{}.txt".format(deid), y)
                #write2file("reident_stdev_{}.txt".format(deid), stdevY)
                np.savetxt("reident_mean_{}.txt".format(method_name), y)   # X is an array
                np.savetxt("reident_stdev_{}.txt".format(method_name), stdevY)   # X is an array
            
                #plot_RNKK_curve(range(2, 11), y, method_name, len(range(2, 12)), 0.6, stdevY, ix)
                plot_BAR_chart(range(2, 11), y, fancy[deid], len(range(2, 12)), 0.5, stdevY, ix)

            # plot formal theoretical line
            y = [1.0 / k for k in range(2, 12)]
            plot_RNKK_curve(range(2, 12), y, fancy["theoretical"], len(y), 0.5, None, 6, xlim_ext=0.5)

            f.set_size_inches(18.5, 10.5)
            f.savefig("cmc-verif-deid-{}.pdf".format(method_name), bbox_inches='tight', dpi=100)

    plt.show()
