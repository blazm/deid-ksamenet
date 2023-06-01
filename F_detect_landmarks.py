import sys
import dlib
import scipy.misc
#from skimage import io
import cv2
import numpy as np

def shape_to_nparray(shape):
    """
    Reshapes Shape from dlib predictor to numpy array
    Args:
        shape (dlib Shape object): input shape points

    Returns: numpy array consisting of shape points

    """
    np_arr = []
    print shape
    for i in range(0,  shape.num_parts):
        np_arr.append((shape.part(i).x,  shape.part(i).y))
    return np.array(np_arr)
    


if __name__ == "__main__":

    img = scipy.misc.imread('./folds/fold_gallery_1/005_1_1.ppm')

    detector = dlib.get_frontal_face_detector()
    predictor_path = "./landmarks/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    shape = None
        
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print(d.__class__)
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

        
        # detection can be a rectangle: src_detection = dlib.rectangle(ix, iy, ix+iw, iy+ih) 

       # prev_shape = shape
        shape = predictor(img, d)
        print("Shape: LEN: {} Part 0: {}, Part 1: {} ...".format(shape.num_parts, shape.part(0), shape.part(1)))
        # Draw the face landmarks on the screen.
    #    win.add_overlay(shape)

        pts_src = shape_to_nparray(shape)

    
    shape_to_nparray(shape)

    # TODO: homography estimation
    # SRC: http://www.learnopencv.com/homography-examples-using-opencv-python-c/
    '''
    pts_src and pts_dst are numpy arrays of points
    in source and destination images. We need at least 
    4 corresponding points. 
    '''

   # pts_dst = shape_to_nparray(prev_shape)

#    print(pts_src)

#    h, status = cv2.findHomography(pts_src, pts_dst)
     
#    print("H: {}".format(h)) # homography matrix
