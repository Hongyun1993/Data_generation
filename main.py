#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:27:27 2018

@author: hongyun
"""
import cv2
import os
import sys
import random
import math
import numpy as np
ROOT_DIR = os.path.abspath("./Mask_RCNN/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco
sys.path.append('./CF')
from fastMatting import fastMatting
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
IMAGE_DIR = './images'
BACKGROUND_DIR = './background'

if not os.path.exists('./masks'):
    os.mkdir('./masks')
if not os.path.exists('./trimaps'):
    os.mkdir('./trimaps')
if not os.path.exists('./alphas'):
    os.mkdir('./alphas')
if not os.path.exists('./results'):
    os.mkdir('./results')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

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

def trimap_generation(img,mask,dilate_ratio):
    #在原mask的基础上腐蚀1倍，扩张2倍，也就是unknow region在物体内部占1/3,外部占2/3.
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

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


visual = False
if_write = True
image_file_names = next(os.walk(IMAGE_DIR))[2]
background_file_names = next(os.walk(BACKGROUND_DIR))[2]

for i in range(30):
    print('-'*20+str(i)+'-'*20)
    image_name = random.choice(image_file_names)
    print(image_name)
    background_name = random.choice(background_file_names)
    img = cv2.imread(os.path.join(IMAGE_DIR, image_name))
    background = cv2.imread(os.path.join(BACKGROUND_DIR, background_name))
    print(np.shape(background))
    if len(np.shape(background)) == 0:
        continue
    results = model.detect([img], verbose=1)
    r = results[0]
    if np.size(r['class_ids']) == 0:
        continue
    scribbles = r['masks']
    scribble = scribbles[:,:,0].astype('uint8')
    col,row = scribble.shape
    if np.mean(scribble) < 0.1:
        continue
    background = cv2.resize(background,(row,col),interpolation=cv2.INTER_AREA)
    mask = mask_generation(img, scribble)
    trimap = trimap_generation(img,mask,0.2)
    alpha = alpha_generation(img,trimap)
    comp = comp_img(img,alpha,background)

    if visual == True:
        cv2.namedWindow('origin_image',0)
        cv2.resizeWindow('origin_image', int(row/2), int(col/2))
        cv2.imshow('origin_image',img)
        cv2.namedWindow('matting_image',0)
        cv2.resizeWindow('matting_image', int(row/2), int(col/2))
        cv2.imshow('matting_image',comp)

    if if_write == True:
        cv2.imwrite('./results/' + image_name.split('.')[0] + '_' + background_name.split('.')[0] + '.jpeg',comp)

        cv2.imwrite('./masks/' + image_name.split('.')[0] + '.jpeg',mask*255)
        cv2.imwrite('./trimaps/' + image_name.split('.')[0] + '.jpeg',trimap)
        cv2.imwrite('./alphas/' + image_name.split('.')[0] + '.jpeg',alpha*255)
