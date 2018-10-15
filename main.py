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
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
IMAGE_DIR = './images'
BACKGROUND_DIR = './background'

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

def comp_img(img,mask,background):
    # img:[row,col,3], mask:[row,col,1], background:[row,col,3]
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

for i in range(100):
    print('-'*20+str(i)+'-'*20)
    image_name = random.choice(image_file_names)
    print(image_name)
    background_name = random.choice(background_file_names)
    img = cv2.imread(os.path.join(IMAGE_DIR, image_name))
    background = cv2.imread(os.path.join(BACKGROUND_DIR, background_name))
    
    results = model.detect([img], verbose=1)
    r = results[0]
    if np.size(r['class_ids']) == 0:
        continue
    scribbles = r['masks']
    scribble = scribbles[:,:,0].astype('uint8')
    col,row = scribble.shape
    background = cv2.resize(background,(row,col),interpolation=cv2.INTER_AREA)
    mask = mask_generation(img, scribble)
    comp = comp_img(img,mask,background)
    
    if visual == True:
        cv2.namedWindow('origin_image',0)
        cv2.resizeWindow('origin_image', int(row/2), int(col/2))
        cv2.imshow('origin_image',img)
        cv2.namedWindow('matting_image',0)
        cv2.resizeWindow('matting_image', int(row/2), int(col/2))
        cv2.imshow('matting_image',comp)
        
    if if_write == True:
        cv2.imwrite('./results/' + image_name.split('.')[0] + '_' + background_name.split('.')[0] + '.jpeg',comp)
    
    
    







