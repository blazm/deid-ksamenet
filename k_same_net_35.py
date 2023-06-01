

from Generator import Emotion

def make_gen():
    
    from keras import backend as K
    K.set_image_dim_ordering('tf')
    from Generator import Generator

    #do_emotions = False

    deconv_layer = 6 # 5 or 6
    model_name = 'FaceGen.RaFD.model.d{}.adam'.format(deconv_layer)
#    model_path = '../de-id/generator/output/{}.h5'.format(model_name)
    model_path = './models/{}.h5'.format(model_name) # locally stored models

    gen = Generator(model_path, deconv_layer=deconv_layer)
    return gen

def k_same_net(gen, clustered_probes=None, clustered_images=None, k=2, emotion = 'neutral'):

    #proxy_path = "../de-id/DB/rafd2-frontal/"
    #proxys = read_files(proxy_path)
    #proxys = filterProxy(proxys)

    import random
    ids = [random.randint(0, 56)]*k

    image = gen.generate(ids, emotion)

    #del gen

    return image

if __name__ == '__main__':
    import os
    import cv2

    dataset_path = '../stylegan2-pytorch/datasets/celeba-test_aligned'
    dataset_path = '../stylegan2-pytorch/datasets/rafd-frontal_aligned'
    #dataset_path = '../stylegan2-pytorch/datasets/xm2vts_aligned'
    dataset_save = '../stylegan2-pytorch/datasets/celeba-test_k_same_net'
    dataset_save = '../stylegan2-pytorch/datasets/rafd_k-Same-Net_deidentified_emo'
    dataset_save = '/home/blaz/github/insightface/deploy/rafd_baselines/rafd_k-Same-Net'
    #dataset_save = '../stylegan2-pytorch/datasets/xm2vts_k-same-net'
    dataset_filetype = 'jpg'
    dataset_newtype = 'jpg'

    img_names = [i for i in os.listdir(dataset_path) if dataset_filetype in i] # change ppm into jpg
    img_names.sort()
    img_paths = [os.path.join(dataset_path, i) for i in img_names]
    save_paths = [os.path.join(dataset_save, i.replace(dataset_filetype, dataset_newtype)) for i in img_names]

    def ensure_dir(d):
        #dd = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
    ensure_dir(dataset_save)

    # TODO: go over all files
    gen = make_gen()
    
    for img_name, img_path in zip(img_names, img_paths):
        #img_a_path = os.path.join(path_to_original_images, str(name_a)+'.jpg')
        #img_b_path = os.path.join(img_b_dir, '{:05d}.png'.format(name_b))        
        #img_b_orig_path = os.path.join(path_to_original_images, str(name_b)+'.jpg')
        #img_a_path = os.path.join(path_to_original_images, name_a)
        #img_b_path = os.path.join(img_b_dir, name_b)        
        
        if os.path.exists(os.path.join(dataset_save, img_name)):
        #if not os.path.exists(img_a_path) or not os.path.exists(img_b_path): # if any of the pipelines failed to detect faces
            print("File already exists, skipping: ", img_name )
            continue

        #img_a = io.imread(img_a_path)
        #img = cv2.imread(img_path)
        print("Processing: ", img_name)
        print(img_name)
        emo = img_name.split('_')[-2]
        #emo = 'neutral'
        deid_img = k_same_net(gen, k=5, emotion=emo)
        deid_img = cv2.cvtColor(deid_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(dataset_save, img_name), deid_img)
