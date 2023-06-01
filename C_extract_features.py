import os
from os.path import join
import numpy as np

from scipy.misc import imread, imresize, imsave

from A_pick_samples import loadFold, idsFromFold, loadImages, loadPackedImages, reset_directory

# deep neural nets
from keras import backend as K
K.set_image_dim_ordering('th')
from networks_def import AlexNet, squeezenet, InceptionV3, vgg_face


def extractFeatures(images, model):
    for img in images:
        feat = model.predict(img)
    return images

if __name__ == "__main__":
    #K.set_image_dim_ordering('th')

    src_path = "./DB/xm2vts/crop/"
    out_path = "./feats/"

    src_path_aam = "./folds_aam/"

    fold_path = "./deid/"

    naive = ['blur', 'pixelize'] # original?
    formal = ['k_same_M', 'k_same_net', 'k_same_pixel']

    num_folds = 5    

    m1 = squeezenet(50, output="denseFeatures",
                    simple_bypass=True, fire11_1024=True)
    m1.load_weights("weights/luksface-weights.h5")
    m2 = AlexNet(N_classes=1000, r=1e-4, p_dropout=0.5, borders="same",
                 inshape=(3, 224, 224), include_softmax=False)
    m2.load_weights("weights/alexnet_weights.h5", by_name=True)
    m3 = InceptionV3(include_top=False)
    m3.load_weights("weights/googlenet_weights.h5")
    m4 = vgg_face(output_layer="fc6")
    m4.load_weights("weights/vgg_face_weights.h5", by_name=True)

    models = [m1, m2, m4, m3]  # m3,
    labels = ['squeezenet', 'alexnet', 'vggface', 'inception']  # 'inception',
    img_sizes = [224, 224, 224, 299]

    # prepare dir stucture
    '''
    for model, label, img_size in zip(models, labels, img_sizes):
        for i in range(num_folds):
            for deid in naive:
                reset_directory(join(out_path, label, "{}".format(deid)))
            for k in range(2, 11):
                for deid in formal:
                    reset_directory(join(out_path, label, "{}_k={}".format(deid, k)))
    '''        

    for model, label, img_size in zip(models, labels, img_sizes):


        print "Extracting with: ", label

        #reset_directory(os.path.join(out_path, label))

        # for i in range(num_folds):
            
        #     gallery = loadFold("fold_gallery_%d.txt" % (i+1))
        #     probes = loadFold("fold_probes_%d.txt" % (i+1))

        #     galleryIDs = np.array([idsFromFold(gallery)]).transpose()
        #     probeIDs = np.array([idsFromFold(probes)]).transpose()

        #     gallery_images = loadImages(path, gallery, newdim=img_size)
        #     probe_images = loadImages(path, probes, newdim=img_size)

        #     # TODO: is this ok!
        #     gallery_images = np.swapaxes(gallery_images, 1, 3).swapaxes(2,3)
        #     probe_images = np.swapaxes(probe_images, 1, 3).swapaxes(2,3)

        #     gallery_feats = model.predict(gallery_images)
        #     probe_feats = model.predict(probe_images)

        #     # add IDs 
        #     #print gallery_feats.shape, galleryIDs.shape
        #     gallery_feats = np.concatenate((galleryIDs, gallery_feats), 1) #gallery_feats[:, 0]
        #     probe_feats = np.concatenate((probeIDs, probe_feats), 1) #gallery_feats[:, 0]
        #     #print gallery_feats[:, 1:].shape, probe_feats[:, 0]

        #     filename = 'fold_gallery_{}-{}.csv'.format(i+1, label)
        #     np.savetxt(os.path.join(out_path, label, filename), gallery_feats, delimiter=',')

        #     filename = 'fold_probes_{}-{}.csv'.format(i+1, label)
        #     np.savetxt(os.path.join(out_path, label, filename), probe_feats, delimiter=',')


        for i in range(num_folds):
            '''
            gallery = loadFold("fold_gallery_%d.txt" % (i+1))
            galleryIDs = np.array([idsFromFold(gallery)]).transpose()
            gallery_images = loadImages(src_path, gallery, newdim=img_size)
            gallery_images = np.swapaxes(gallery_images, 1, 3).swapaxes(2,3)
            gallery_feats = model.predict(gallery_images)
            gallery_feats = np.concatenate((galleryIDs, gallery_feats), 1) #gallery_feats[:, 0]
            filename = 'fold_gallery_{}.csv'.format(i+1)
            np.savetxt(join(out_path, label, filename), gallery_feats, delimiter=',')
            '''
            # aam
            gallery = loadFold("fold_gallery_%d.txt" % (i+1))
            galleryIDs = np.array([idsFromFold(gallery)]).transpose()
            gallery_images = loadImages(join(src_path_aam, "fold_gallery_%d" %(i+1)), gallery, newdim=img_size)
            gallery_images = np.swapaxes(gallery_images, 1, 3).swapaxes(2,3)
            gallery_feats = model.predict(gallery_images)
            gallery_feats = np.concatenate((galleryIDs, gallery_feats), 1) #gallery_feats[:, 0]
            filename = 'fold_gallery_aam_{}.csv'.format(i+1)
            np.savetxt(join(out_path, label, filename), gallery_feats, delimiter=',')

            '''
            probes = loadFold("fold_probes_%d.txt" % (i+1))
            probeIDs = np.array([idsFromFold(probes)]).transpose()
            probe_images = loadImages(src_path, probes, newdim=img_size)
            probe_images = np.swapaxes(probe_images, 1, 3).swapaxes(2,3)
            probe_feats = model.predict(probe_images)
            probe_feats = np.concatenate((probeIDs, probe_feats), 1)
            filename = 'fold_probes_{}.csv'.format(i+1)
            np.savetxt(join(out_path, label, filename), probe_feats, delimiter=',')
            '''
            # predict for naive methods
            '''
            for deid in naive:
                n_path = os.path.join(fold_path, "{}".format(deid), "fold_probes_%d" % (i+1))
                probe_images = loadImages(n_path, probes, newdim=img_size)

                probe_images = np.swapaxes(probe_images, 1, 3).swapaxes(2,3)
                probe_feats = model.predict(probe_images)
                probe_feats = np.concatenate((probeIDs, probe_feats), 1) #gallery_feats[:, 0]
                    
                filename = 'fold_probes_{}.csv'.format(i+1)
                
                np.savetxt(join(out_path, label, "{}".format(deid), filename), probe_feats, delimiter=',')

            for k in range(2, 11):
                
                print "Extracting with k=%d: " %(k), label

                # predict for all formal methods
                for deid in formal:# ['k_same_M', 'k_same_net', 'k_same_pixel']:

                    k_path = join(fold_path, "{}_k={}".format(deid, k), "fold_probes_%d" % (i+1))
                    probe_images = loadImages(k_path, probes, newdim=img_size)

                    # TODO: is this ok!
                    probe_images = np.swapaxes(probe_images, 1, 3).swapaxes(2,3)
                    probe_feats = model.predict(probe_images)

                    # add IDs 
                    #print gallery_feats.shape, galleryIDs.shape
                    probe_feats = np.concatenate((probeIDs, probe_feats), 1) #gallery_feats[:, 0]
                    #print gallery_feats[:, 1:].shape, probe_feats[:, 0]

                    filename = 'fold_probes_{}.csv'.format(i+1)
                    
                    np.savetxt(join(out_path, label, "{}_k={}".format(deid, k), filename), probe_feats, delimiter=',')
            '''
            