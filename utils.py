import cv2
import numpy as np


def cal_mask_error(mask,mask_pred):
    '''
    input:
    mask:[row,col,n1], mask_pred:[row,col,n2],index:[row,col,n1]
    output:
    error:[n1]
    '''
    error = mask.astype(np.int) - mask_align.astype(np.int)
    ierror = np.sum(error**2)
    #print(error)
    ratio = np.sum(mask)
    #print(ratio)
    error = error/ratio
    #print(error)
    return error
