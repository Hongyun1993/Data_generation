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
sys.path.append('../')
from utils import *
from skimage import io,color,transform
from Fitness import *

begin = time.time()
# load dataset-----------------------------------------------------------------
img_name = '2.jpg'
# img = io.imread('./data/input_training_lowres/'+img_name)
# trimap = io.imread('./data/trimap_training_lowres/'+img_name)
img = io.imread('./data/matting_samples/images/'+img_name)
mask = io.imread('./data/matting_samples/mask/'+img_name)
mask = (255*(mask[:,:,0]>200)).astype(np.uint8)
#mask = np.zeros(np.shape(trimap))
#mask[trimap==255] = 255
#mask = mask.astype(np.uint8)
# SuperPixels Generation
r_ratio = 0.5
img = transform.rescale(img, r_ratio)
# print('the max img:',np.max(img))
mask = transform.rescale(mask, r_ratio)
# print('the max mask:',np.max(mask))
# Refined Mask Generation by GrabCut
#mask_grab = mask_generation((img*255).astype(np.uint8), mask.astype(np.uint8))
mask_grab = mask.copy()
trimap = trimap_generation(img,mask_grab,0.3)
alpha_cf = alpha_generation(img,trimap,trimap)

alpha_cf[alpha_cf>1] = 1
alpha_cf[alpha_cf<0] = 0

trimap_patch = trimap.copy()
clusters_uk,fitness = fitness(img,mask_grab)
insight = np.zeros(np.shape(img))
insight_tmp_r = np.zeros(np.shape(trimap))
insight_tmp_g = np.zeros(np.shape(trimap))
len_uk = len(fitness)
for i in range(len_uk):
    cluster = clusters_uk[i]
    uk_pixels = np.array(cluster.pixels)
    tmp1 = np.zeros(np.shape(trimap))
    tmp1[uk_pixels[:,0],uk_pixels[:,1]] = 1
    kernel = np.ones((3,3))
    tmp2 = cv2.dilate(tmp1,kernel,1)
    tmp = tmp2 - tmp1
    if fitness[i] == 1:
        insight_tmp_g[tmp>0] = 255
    else:
        insight_tmp_r[tmp>0] = 255
        trimap_patch[uk_pixels[:,0],uk_pixels[:,1]] = \
            mask_grab[uk_pixels[:,0],uk_pixels[:,1]]*255

insight[:,:,0] = insight_tmp_r
insight[:,:,1] = insight_tmp_g

alpha_patch = alpha_generation(img,trimap,trimap_patch)
alpha_patch[alpha_patch>1] = 1
alpha_patch[alpha_patch<0] = 0

new_img = img*255
index = np.array(np.where(insight_tmp_r>0))
new_img[index[0,:],index[1,:],:] = insight[index[0,:],index[1,:],:]
index = np.array(np.where(insight_tmp_g>0))
new_img[index[0,:],index[1,:],:] = insight[index[0,:],index[1,:],:]

background = np.zeros(np.shape(img))
row,col = np.shape(alpha_cf)[:2]
alpha_cf2 = np.reshape(alpha_cf,[row,col,1])
alpha_cf2 = np.tile(alpha_cf2,[1,1,3])
alpha_patch2 = np.reshape(alpha_patch,[row,col,1])
alpha_patch2 = np.tile(alpha_patch2,[1,1,3])
comp_cf = alpha_cf2*img + (1-alpha_cf2)*background
comp_patch = alpha_patch2*img + (1-alpha_patch2)*background

io.imsave('mask.png',(mask_grab*255).astype(np.uint8))
io.imsave('alpha_cf.png',(alpha_cf*255).astype(np.uint8))
io.imsave('output.png',new_img.astype(np.uint8))
io.imsave('alpha_patch.png',(alpha_patch*255).astype(np.uint8))
io.imsave('comp_cf.png',(comp_cf*255).astype(np.uint8))
io.imsave('comp_patch.png',(comp_patch*255).astype(np.uint8))
end = time.time()
print('time:',end-begin)
