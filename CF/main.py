import os
import cv2
import numpy as np
from fastMatting import fastMatting

def comp_img(fg,alpha,bg):
    '''
    input: fg:[row,col,3], alpha:[row,col], bg[row,col,3]
    output: comp:[row,col,3]
    '''
    comp = fg*alpha[:,:,np.newaxis] + bg*(1-alpha[:,:,np.newaxis])
    return comp


loadPath = './data'
savePath = './data/pred_alpha'
fgPath = os.path.join(loadPath,'input_training_lowres')
triPath = os.path.join(loadPath,'trimap_training_lowres')
#bgPath = os.path.join(loadPath,'backgrounds')
img_name = 'GT14.png'
fg = cv2.imread(os.path.join(fgPath,img_name))
tri = cv2.imread(os.path.join(triPath,img_name),cv2.IMREAD_GRAYSCALE)
row,col = np.shape(tri)
fg = cv2.resize(fg,(int(0.5*col),int(0.5*row)),interpolation=cv2.INTER_AREA)
tri = cv2.resize(tri,(int(0.5*col),int(0.5*row)),interpolation=cv2.INTER_AREA)
#bg = #cv2.imread(os.path.join(bgPath,'1.jpg'))
alpha = fastMatting(fg,tri)
cv2.imwrite(os.path.join(savePath,img_name),alpha*255)
#comp = comp_img(fg,alpha,bg)
#cv2.destroyAllWindows()
