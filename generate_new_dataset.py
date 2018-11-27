#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:35:31 2018

@author: lhy
"""

import os
import cv2
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from utils import *
sys.path.append('./Mask_RCNN/samples/coco')
from pycocotools.coco import COCO

## Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")

#Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

sys.path.append('./CF')
from fastMatting import fastMatting

BACKGROUND_DIR = './background'

# MS COCO Dataset
import coco
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

COCO_DIR = "../coco/images"  # TODO: enter value here

new_image_dir = '../coco/new_images'
if not os.path.exists(new_image_dir):
    os.mkdir(new_image_dir)

new_ann_dir = '../coco/new_annotations'
if not os.path.exists(new_ann_dir):
    os.mkdir(new_ann_dir)

# load dataset-----------------------------------------------------------------
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
# Must call before using the dataset
dataset.prepare()

# ann part---------------------------------------------------------------------
ann_path = '../coco/annotations/instances_train2014.json'
new_ann_path = '../coco/new_annotations/instances_train2014.json'
with open(ann_path,'r') as load_f:
    load_dict = json.load(load_f)
coco_detail = COCO(ann_path)
image_ids = coco_detail.imgs.keys()
load_dict_new = load_dict.copy()
# annotations = load_dict['annotations'] #add new image to original dataset
annotations = []  # creat new dataset
ann_id_num = len(annotations)
class_ids = sorted(coco_detail.getCatIds())
#------------------------------------------------------------------------------

# Load model-------------------------------------------------------------------
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
#------------------------------------------------------------------------------

background_file_names = next(os.walk(BACKGROUND_DIR))[2]

ii = 0

new_index_num = 2000000

lens_image = len(image_ids)
for image_id in image_ids:
    ii += 1
    print('-'*20+str(lens_image - ii)+'-'*20)

    # generate new image ------------------------------------------------------
    image = dataset.load_image(image_id)
    background_name = random.choice(background_file_names)
    background = cv2.imread(os.path.join(BACKGROUND_DIR, background_name))
    col,row = np.shape(image)[:2]
    background = cv2.resize(background,(row,col),interpolation=cv2.INTER_AREA)
    mask, class_ids = dataset.load_mask(image_id)

    # Compute Bounding box-----------------------------------------------------
#    bbox = utils.extract_bboxes(mask)
#    if np.shape(bbox)[0]>3:
#        continue
#
    results = model.detect([image], verbose=1)
    r = results[0]
    mask_pred = r['masks']
    class_ids_pred = r['class_ids']
    bbox_pred = utils.extract_bboxes(mask_pred)
    if len(bbox_pred) == 0:
        continue
#    if is_show == True:
#        visualize.display_instances(image, bbox_pred, mask_pred, class_ids_pred, dataset.class_names)
    #print('pred mask shape:',np.shape(mask_pred))

    mask = np.sum(mask.astype(np.uint8),axis = 2)
    mask[mask>1] = 1
    mask = mask.astype(np.uint8)
    if np.sum(mask)/(row*col)<0.1:
        continue
    mask_pred = np.sum(mask_pred.astype(np.uint8),axis = 2)
    mask_pred[mask_pred>1] = 1
    mask_pred = mask_pred.astype(np.uint8)
    mask_error = cal_mask_error(mask, mask_pred)

    if mask_error > 0.2:
        continue

    # append new annotation----------------------------------------------------
    ann=coco_detail.loadAnns(coco_detail.getAnnIds(imgIds=[image_id], catIds=class_ids, iscrowd=None))
    lens_ann = len(ann)
    new_image_id = image_id + new_index_num
    for i in range(lens_ann):
        ann[i]['image_id'] = new_image_id
    annotations.extend(ann)

    if ii%100 == 0:
        load_dict_new['annotations'] = annotations
        with open(new_ann_path,"w") as f:
            json.dump(load_dict_new,f)
            print("保存文件完成...")


    img_patch_name = str(mask_error) + '__' + str(image_id) + '.png'

    new_image = image.copy()
    new_image[:,:,0] = image[:,:,2]
    new_image[:,:,2] = image[:,:,0]
    mask_grab = mask_generation(new_image, mask)
    trimap = trimap_generation(new_image,mask_grab,0.05)
    alpha = alpha_generation(new_image,trimap)
    comp = comp_img(new_image,alpha,background)

    if_write = True
    if if_write == True:
        cv2.imwrite(os.path.join(new_image_dir,img_patch_name),comp)
