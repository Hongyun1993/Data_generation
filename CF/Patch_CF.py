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

ROOT_DIR = os.path.abspath("../Mask_RCNN")
#Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append('../Mask_RCNN/samples/coco')
import coco
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

COCO_DIR = "../../coco/images"  # TODO: enter value here

begin = time.time()
# load dataset-----------------------------------------------------------------
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
# Must call before using the dataset
dataset.prepare()
image_ids = dataset.image_ids
image_id = image_ids[8]
img = dataset.load_image(image_id)
mask,class_ids = dataset.load_mask(image_id)
mask = np.sum(mask.astype(np.uint8),axis = 2)
mask[mask>=1] = 255
mask = mask.astype(np.uint8)
# img_path = './images'
# img_name = '5.jpeg'
# img = cv2.imread(os.path.join(img_path,img_name))
#------------------------------------------------------------------------
# SuperPixels Generation
r_ratio = 1
img = transform.rescale(img, r_ratio)
# print('the max img:',np.max(img))
mask = transform.rescale(mask, r_ratio)
# print('the max mask:',np.max(mask))
p = SLICProcessor(img,1000,20)
p.iterates()
# Refined Mask Generation by GrabCut
mask_grab = mask_generation(img, mask)
io.imsave('mask.png',(mask*255).astype(np.uint8))
# Find the super pixels which can be matted
clusters = p.clusters
num_c = len(clusters)
clusters_fg = []
clusters_bg = []
clusters_uk = []
row_col_fg = []
row_col_bg = []
row_col_uk = []

alpha = np.zeros(np.shape(mask))
insight = np.zeros(np.shape(img))
for i in range(num_c):
    cluster = clusters[i]
    pixels = np.array(cluster.pixels)
    if np.shape(pixels)[0] == 0:
         continue
    print('pixels_shape:',np.shape(pixels))
    sum_mask = np.sum(mask[pixels[:,0],pixels[:,1]])
    print('sum_mask:',sum_mask)
    len_region = np.shape(pixels)[0]
    row_col = [cluster.row, cluster.col]
    if sum_mask/len_region < 0.05:
#        clusters[i].type = 'bg'
        clusters_bg.append(cluster)
        row_col_bg.append(row_col)
    elif sum_mask/len_region > 0.95:
#        clusters[i].type = 'fg'
        clusters_fg.append(cluster)
        row_col_fg.append(row_col)
        fg_pixels = np.array(cluster.pixels)
        alpha[fg_pixels[:,0], fg_pixels[:,1]] = 1
    else:
#        clusters[i].type = 'uk'
        clusters_uk.append(cluster)
        row_col_uk.append(row_col)

row_col_fg = np.array(row_col_fg)
row_col_bg = np.array(row_col_bg)
row_col_uk = np.array(row_col_uk)
len_fg = np.shape(row_col_fg)[0]
len_bg = np.shape(row_col_bg)[0]
len_uk = np.shape(row_col_uk)[0]
print('len_fg:',len_fg)
print('len_bg:',len_bg)
print('len_uk:',len_uk)
if len_fg !=0 and len_bg !=0 and len_uk !=0:
    row_col_fg = np.reshape(row_col_fg, [1,len_fg,2])
    row_col_bg = np.reshape(row_col_bg, [1,len_bg,2])
    row_col_uk = np.reshape(row_col_uk, [len_uk,1,2])

    row_col_fg = np.tile(row_col_fg,[len_uk,1,1])
    row_col_bg = np.tile(row_col_bg,[len_uk,1,1])
    row_col_uk_f = np.tile(row_col_uk,[1,len_fg,1])
    row_col_uk_b = np.tile(row_col_uk,[1,len_bg,1])

    error_u_f = np.sum((row_col_fg - row_col_uk_f)**2, axis = 2)
    error_u_b = np.sum((row_col_bg - row_col_uk_b)**2, axis = 2)

    # store the closed superpixel fg and bg index
    index_u_f = np.argmin(error_u_f, axis = 1)
    index_u_b = np.argmin(error_u_b, axis = 1)

    fitness_uk = np.zeros((len_uk))
    threshold = 30
    for i in range(len_uk):
        cluster_uk = clusters_uk[i]
        uk_pixels = np.array(cluster_uk.pixels)
        cluster_fg = clusters_fg[index_u_f[i]]
        fg_lab = np.array([cluster_fg.l, cluster_fg.a, cluster_fg.b])
        cluster_bg = clusters_bg[index_u_b[i]]
        bg_lab = np.array([cluster_bg.l, cluster_bg.a, cluster_bg.b])
        lab_error = np.sqrt(np.sum((fg_lab - bg_lab)**2))
        print('lab_error:', lab_error)
        if lab_error < threshold:
            alpha[uk_pixels[:,0], uk_pixels[:,1]]= \
            mask_grab[uk_pixels[:,0], uk_pixels[:,1]]
            insight[uk_pixels[:,0],uk_pixels[:,1],0] = 255
        else:
            fitness_uk[i] = 1
            insight[uk_pixels[:,0],uk_pixels[:,1],1] = 255
            # Region Matting
            alpha_patch = sampling_matting(img,fg_lab,bg_lab,cluster_uk)
            alpha[uk_pixels[:,0], uk_pixels[:,1]]= alpha_patch

new_img = img*255 + 0.2*insight
new_img[new_img>255] = 255
io.imsave('output.png',new_img.astype(np.uint8))
end = time.time()
print('time:',end-begin)
