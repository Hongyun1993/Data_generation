
import numpy as np
from skimage import io,color

def sampling_matting(img,fg_lab,bg_lab,cluster_uk):
    fg_lab = np.reshape(fg_lab,[1,1,3])
    fg_color = color.lab2rgb(fg_lab)
    fg_color = np.reshape(fg_color,[1,3])
    bg_lab = np.reshape(bg_lab,[1,1,3])
    bg_color = color.lab2rgb(bg_lab)
    bg_color = np.reshape(bg_color,[1,3])

    uk_pixels = np.array(cluster_uk.pixels)
    len_uk = np.shape(uk_pixels)[0]
    pixels_img = img[uk_pixels[:,0],uk_pixels[:,1],:]

    fg_color = np.tile(fg_color,[len_uk,1])
    bg_color = np.tile(bg_color,[len_uk,1])
    alpha = (pixels_img - bg_color)/(fg_color - bg_color)
    alpha = alpha[:,0]
    return alpha
