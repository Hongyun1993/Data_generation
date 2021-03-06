#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:35:31 2018

@author: lhy
"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

BACKGROUND_DIR = '../爬取文件'

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

new_path = '../coco/new_images_origin_mask'
if not os.path.exists(new_path):
    os.mkdir(new_path)
new_image_dir = new_path + '/train2014'
if not os.path.exists(new_image_dir):
    os.mkdir(new_image_dir)
new_ann_dir = new_path + '/annotations'
if not os.path.exists(new_ann_dir):
    os.mkdir(new_ann_dir)

# load dataset-----------------------------------------------------------------
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
# Must call before using the dataset
dataset.prepare()

# ann part---------------------------------------------------------------------
ann_path = '../coco/images/annotations/instances_train2014.json'
new_ann_path = '../coco/new_images_origin_mask/annotations/instances_train2014.json'
with open(ann_path,'r') as load_f:
    load_dict = json.load(load_f)
load_dict_new = load_dict.copy()
# annotations = load_dict['annotations'] #add new image to original dataset
annotations = []  # creat new dataset

coco_detail = COCO(ann_path)
img_details = coco_detail.imgs
#img_details_new = load_dict['images']
img_details_new = []
ann_id_num = len(annotations)
load_dict_new['images'] = []
load_dict_new['annotations'] = []
#------------------------------------------------------------------------------

# Load model-------------------------------------------------------------------
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
#------------------------------------------------------------------------------

background_file_names = next(os.walk(BACKGROUND_DIR))[1]

ii = 0

new_index_num = 10**11
image_ids = dataset.image_ids
lens_image = len(image_ids)
for o in range(lens_image):
    image_id = image_ids[o]
    print('-'*20+str(lens_image - o)+'-'*20)
    # generate new image ------------------------------------------------------
    image = dataset.load_image(image_id)
    print('image_shape:',np.shape(image))
    while 1:
        background_file = random.choice(background_file_names)
        bg_path = os.path.join(BACKGROUND_DIR,background_file)
        background_names = next(os.walk(bg_path))[-1]
        background_name = random.choice(background_names)
        background = cv2.imread(os.path.join(bg_path, background_name),cv2.IMREAD_COLOR)
        print('background shape:',np.shape(background))
        if len(np.shape(background))>0:
            r,c,l = np.shape(background)
            if r>0 and c>0 and l>0:
                break
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
    if np.sum(mask)/(row*col)<0.1 or np.sum(mask)/(row*col)>0.9:
        continue
    mask_pred = np.sum(mask_pred.astype(np.uint8),axis = 2)
    mask_pred[mask_pred>1] = 1
    mask_pred = mask_pred.astype(np.uint8)
    mask_error = cal_mask_error(mask, mask_pred)

    if mask_error > 0.2:
        continue

    # append new annotation----------------------------------------------------
    ann= dataset.image_info[image_id]["annotations"]
    img_id = dataset.image_info[image_id]['id']
    lens_ann = len(ann)
    new_image_id = int(img_id + new_index_num)
    for i in range(lens_ann):
        ann[i]['image_id'] = new_image_id
    annotations.extend(ann)

    img_patch_name = dataset.image_info[image_id]['path'].split('/')[-1]
    temp_names = list(img_patch_name)
    temp_names[15] = '1'
    img_patch_name = ''.join(temp_names)

    img_detail = img_details[img_id]
    img_detail['id'] = new_image_id
    img_detail['file_name'] = img_patch_name
    img_details_new.append(img_detail)

    ii += 1
    if ii%100 == 1:
        load_dict_new['annotations'].extend(annotations)
        load_dict_new['images'].extend(img_details_new)
        annotations = []
        img_details_new = []
        with open(new_ann_path,"w") as f:
            json.dump(load_dict_new,f)
            print("保存文件完成...")

    new_image = image.copy()
    new_image[:,:,0] = image[:,:,2]
    new_image[:,:,2] = image[:,:,0]
    comp = comp_img(new_image, mask, background)

    if_write = True
    if if_write == True:
        cv2.imwrite(os.path.join(new_image_dir,img_patch_name),comp)
