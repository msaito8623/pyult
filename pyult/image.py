import numpy as np
import cv2

def vec_to_imgs (vector, number_of_vectors, pixel_per_vector):
    """
    Convert an ultrasound vector into a series of images.
    """
    frame_size = number_of_vectors * pixel_per_vector
    number_of_frames = vector.size // int(frame_size)
    imgs = vector.reshape(number_of_frames, number_of_vectors, pixel_per_vector)
    imgs = np.rot90(imgs, axes=(1,2))
    return imgs

def crop (imgs, crop_points):
    """
    Crop every frame in a series of images according to crop_points.
    """
    def __crop2d (img, crop_points):
        ylen, xlen = img.shape
        xmin, xmax, ymin, ymax = crop_points
        xmin = 0 if xmin is None else xmin
        ymin = 0 if ymin is None else ymin
        xmax = xlen-1 if xmax is None else xmax
        ymax = ylen-1 if ymax is None else ymax
        img = img[ymin:(ymax+1), xmin:(xmax+1)]
        return img
    imgs = [ __crop2d(i, crop_points) for i in imgs ]
    imgs = np.array(imgs)
    return imgs

def add_crop_line (imgs, crop_points):
    """
    Highlight cropping points of the four sides by red lines.
    """
    def __crop2d (img, crop_points):
        ylen, xlen = img.shape
        xmin, xmax, ymin, ymax = crop_points
        xmin = 0 if xmin is None else xmin
        ymin = 0 if ymin is None else ymin
        xmax = xlen-1 if xmax is None else xmax
        ymax = ylen-1 if ymax is None else ymax
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img[ymin,:,:] = (0,0,255) 
        img[ymax,:,:] = (0,0,255) 
        img[:,xmin,:] = (0,0,255) 
        img[:,xmax,:] = (0,0,255) 
        return img
    imgs = [ __crop2d(i, crop_points) for i in imgs ]
    imgs = np.array(imgs)
    return imgs



