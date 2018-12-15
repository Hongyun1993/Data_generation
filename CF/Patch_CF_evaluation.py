#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:29:57 2018

@author: lhy
"""
import os
import sys
import cv2
import time
import numpy as np
from SILC import *
from sampling_matting import *
sys.path.append('../')
from utils import *
from skimage import io,color,transform
from Fitness import *

begin = time.time()
# load dataset-----------------------------------------------------------------
img_name = 'GT05.png'
img = io.imread('./data/input_training_lowres/'+img_name)
trimap = io.imread('./data/trimap_training_lowres/'+img_name)
mask = np.zeros(np.shape(trimap))
mask[trimap==255] = 255
mask = mask.astype(np.uint8)
# SuperPixels Generation
r_ratio = 1
img = transform.rescale(img, r_ratio)
# print('the max img:',np.max(img))
mask = transform.rescale(mask, r_ratio)
# print('the max mask:',np.max(mask))
# Refined Mask Generation by GrabCut
mask_grab = mask_generation((img*255).astype(np.uint8), mask.astype(np.uint8))
trimap = trimap_generation(img,mask_grab,0.1)
alpha_cf = alpha_generation(img,trimap)

alpha_cf[alpha_cf>1] = 1
alpha_cf[alpha_cf<0] = 0
alpha_patch = alpha_cf.copy()
clusters_uk,fitness = fitness(img,mask_grab)

insight = np.zeros(np.shape(img))
len_uk = len(fitness)
for i in range(len_uk):
    cluster = clusters_uk[i]
    uk_pixels = np.array(cluster.pixels)
    if fitness[i] == 1:
        insight[uk_pixels[:,0],uk_pixels[:,1],1] = 255
    else:
        insight[uk_pixels[:,0],uk_pixels[:,1],0] = 255
        alpha_patch[uk_pixels[:,0],uk_pixels[:,1]] = \
                mask_grab[uk_pixels[:,0],uk_pixels[:,1]]


new_img = img*255 + 0.2*insight
new_img[new_img>255] = 255

io.imsave('mask.png',(mask_grab*255).astype(np.uint8))
io.imsave('alpha_cf.png',(alpha_cf*255).astype(np.uint8))
io.imsave('output.png',new_img.astype(np.uint8))
io.imsave('alpha_patch.png',(alpha_patch*255).astype(np.uint8))
end = time.time()
print('time:',end-begin)
