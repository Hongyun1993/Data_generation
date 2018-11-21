import cv2
import numpy as np

def VThin(image,array):
    h,w = np.shape(image)[:2]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i,j-1]+image[i,j]+image[i,j+1] if 0<j<w-1 else 1
                if image[i,j] == 0  and M != 0:
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def HThin(image,array):
    h,w = np.shape(image)
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1,j]+image[i,j]+image[i+1,j] if 0<i<h-1 else 1
                if image[i,j] == 0 and M != 0:
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def Xihua(image,array,num=10):
    iXihua = image.copy()
    for i in range(num):
        VThin(iXihua,array)
        HThin(iXihua,array)
    return iXihua

def Two(image):
    w,h = np.shape(image)[:2]
    size = (w,h)
    iTwo = np.zeros(size).astype(np.uint8)
    for i in range(w):
        for j in range(h):
            iTwo[i,j] = 0 if image[i,j] < 200 else 255
    return iTwo


array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

def mask_thin(mask):
    '''
    input:
    mask:[row,col], value_interval:[0,255]
    output:
    iThin:[row,col], value_interval:[0,255]
    '''
    array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
             1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]
    iTwo = Two(image)
    iThin = Xihua(iTwo,array)
    return iThin

if __name__ == '__main__':
    image = cv2.imread('./mask_davis/bmx-bumps/00000.png',cv2.IMREAD_GRAYSCALE)
    #image = (255 - image).astype(np.uint8)
    iThin = mask_thin(image)
    cv2.imshow('image',image)
    cv2.imshow('iThin',iThin)
    cv2.waitKey(0)
