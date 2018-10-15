import cv2
import numpy as np

def mask_generation(img, scribble):
    # img:[row,col,3], scribble:[row,col,1]
    col,row = scribble.shape
    kernel = np.ones((int(np.ceil(row/100)),int(np.ceil(col/100))), np.uint8)
    erosion = cv2.erode(scribble,kernel,iterations = 3) 
    mask=2*np.ones((img.shape[:2]),np.uint8)
    mask[erosion == 1] = 1
    bgdModel=np.zeros((1,65),np.float64)
    fgdModel=np.zeros((1,65),np.float64)
    mask2, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask3 = np.where((mask2==2)|(mask2==0),0,1).astype('uint8')
    return mask3

def comp_img(img,mask,background):
    # img:[row,col,3], mask:[row,col,1], background:[row,col,3]
    result = img*mask[:,:,np.newaxis] + background*(1-mask[:,:,np.newaxis])
    return result

def main():
    image_name = '30.jpg'
    background_name = '10.jpeg'
    img = cv2.imread('./images/' + image_name)
    background = cv2.imread('./background/' + background_name)
    scribbles = np.load('./masks/result.npy').astype('uint8')
    scribble = scribbles[:,:,0]
    col,row = scribble.shape
    background = cv2.resize(background,(row,col),interpolation=cv2.INTER_AREA)
    mask = mask_generation(img, scribble)
    comp = comp_img(img,mask,background)
    cv2.namedWindow('origin_image',0)
    cv2.resizeWindow('origin_image', int(row/2), int(col/2))
    cv2.imshow('origin_image',img)
    cv2.namedWindow('matting_image',0)
    cv2.resizeWindow('matting_image', int(row/2), int(col/2))
    cv2.imshow('matting_image',comp)
    cv2.imwrite('./results/' + image_name.split('.')[0] + '_' + background_name.split('.')[0] + '.jpeg',comp)

if __name__ == '__main__':
    main()
    
    
    
    
    
    