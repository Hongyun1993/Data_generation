import cv2
import sys
import numpy as np
sys.path.append('./CF')
from fastMatting import fastMatting

def cal_mask_error(mask,mask_pred):
    '''
    input:
    mask:[row,col,n1], mask_pred:[row,col,n2],index:[row,col,n1]
    output:
    error:[n1]
    '''
    error = mask.astype(np.int) - mask_pred.astype(np.int)
    error = np.sum(error**2)
    #print(error)
    ratio = np.sum(mask)
    #print(ratio)
    error = error/ratio
    #print(error)
    return error

def mask_generation(img, scribble):
    # img:[row,col,3], scribble:[row,col,1]
    col,row = scribble.shape
    kernel = np.ones((int(np.ceil(row/100)),int(np.ceil(col/100))), np.uint8)
    erosion = cv2.erode(scribble,kernel,iterations = 3)
    dilate = cv2.dilate(scribble,kernel,iterations = 3)
    mask = np.zeros((img.shape[:2]),np.uint8)
    mask[dilate == 1] = 2
    mask[erosion == 1] = 1
    bgdModel=np.zeros((1,65),np.float64)
    fgdModel=np.zeros((1,65),np.float64)
    mask2, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask3 = np.where((mask2==2)|(mask2==0),0,1).astype('uint8')
    return mask3

def trimap_generation(img,mask,dilate_ratio):
    #在原mask的基础上腐蚀1倍，i扩张2倍，也就是unknow region在物体内部占1/3,外部占2/3.
    #input: img:[row,col,3], mask:[row,col], dilate_ratio:float
    #output: trimap:[row,col],float,前景值为255,背景值为0,不确定区域为128
    dilate_ratio = np.tanh(dilate_ratio)
    row,col = np.shape(img)[:2]
    row_k,col_k = int(dilate_ratio*row/10), int(dilate_ratio*col/10)
    kernel = np.ones((row_k,col_k))
    mask_erode = cv2.erode(mask,kernel,1)
    mask_erode = cv2.erode(mask_erode,kernel,1)
    mask_dilate = cv2.dilate(mask,kernel,1)
    mask_dilate = cv2.dilate(mask_dilate,kernel,1)
    mask_dilate = cv2.dilate(mask_dilate,kernel,1)
    trimap = np.zeros((row,col))
    trimap[mask_dilate == 1] = 128
    trimap[mask_erode == 1] = 255
    return trimap

def alpha_generation(img,trimap):
    #input: img:[row,col,3],trimap:[row,col]
    #output: alpha:[row,col],float,0~1
    alpha = fastMatting(img,trimap)
    alpha[alpha<0] = 0
    alpha[alpha>1] = 1
    return alpha

def comp_img(img,mask,background):
    # img:[row,col,3], mask:[row,col], background:[row,col,3]
    result = img*mask[:,:,np.newaxis] + background*(1-mask[:,:,np.newaxis])
    return result
