import cv2
import numpy as np

def cal_bbox_error(bbox, bbox_pred):
    '''
    input:
    bbox: [n1,4], bbox_pred:[n2,4]
    output:
    error_min_index:[n1]
    '''
    print(bbox)
    print(bbox_pred)
    lens_bbox = np.shape(bbox)[0]
    lens_bbox_pred = np.shape(bbox_pred)[0]
    bbox = np.reshape(bbox,(lens_bbox,1,4))
    bbox_pred = np.reshape(bbox_pred,(1,lens_bbox_pred,4))

    bbox_mul = np.tile(bbox,(1,lens_bbox_pred,1))
    bbox_pred_mul = np.tile(bbox_pred,(lens_bbox,1,1))
    #print(np.shape(bbox_mul))
    #print(np.shape(bbox_pred_mul))
    error = bbox_mul - bbox_pred_mul
    error = np.sum(error**2, axis = 2)
    error_min = np.min(error,axis = 1)
    error_min_index = np.argmin(error,axis = 1)

    return error_min_index

def cal_mask_error(mask,mask_pred,index):
    '''
    input:
    mask:[row,col,n1], mask_pred:[row,col,n2],index:[row,col,n1]
    output:
    error:[n1]
    '''
    mask_align = mask_pred[:,:,index]
    error = mask.astype(np.int) - mask_align.astype(np.int)
    error = np.sum(error**2, axis = (0,1))
    #print(error)
    ratio = np.sum(mask,axis = (0,1))
    #print(ratio)
    error = error/ratio
    #print(error)
    return error

def choose_img_patch(img,bbox,error):
    '''
    function:
    This function clipping the img_patch which has the smallest error from
    img, and return the img_patch and the error vector which has deleted the
    img_patch's error value.

    input:
    img:[row,col,3],bbox:[n1,4],error:[n1]
    output:
    img_patch:[row_patch,col_patch,3], error:[n1 - 1]
    '''
    index = np.argmin(error)
    bbox_i = bbox[index,:]
    img_patch = img[bbox_i[0]:bbox_i[2],bbox_i[1]:bbox_i[3],:]
    np.delete(error,index)
    return img_patch,error


if __name__ == '__main__':
    bbox = [[1,2,3,4],[5,6,7,8]]
    bbox_pred = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    bbox = np.array(bbox)
    bbox_pred = np.array(bbox_pred)
    cal_bbox_error(bbox,bbox_pred)
