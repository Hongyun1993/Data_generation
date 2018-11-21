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

if not os.path.exists('./masks'):
    os.mkdir('./masks')
if not os.path.exists('./masks_grab'):
    os.mkdir('./masks_grab')
if not os.path.exists('./trimaps'):
    os.mkdir('./trimaps')
if not os.path.exists('./alphas'):
    os.mkdir('./alphas')
if not os.path.exists('./results'):
    os.mkdir('./results')

BACKGROUND_DIR = './background'


# MS COCO Dataset
sys.path.append('./Mask_RCNN/samples/coco/')
import coco
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

COCO_DIR = "../coco/images"  # TODO: enter value here

if not os.path.exists('./image_all'):
   os.mkdir('./image_all')



# Load dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

#Load model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

n_num = 80000
background_file_names = next(os.walk(BACKGROUND_DIR))[2]
for o in range(n_num):
    print('-'*20 + str(n_num - o) + '-'*20)
    # Load random image and mask.
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    background_name = random.choice(background_file_names)
    background = cv2.imread(os.path.join(BACKGROUND_DIR, background_name))
    col,row = np.shape(image)[:2]
    background = cv2.resize(background,(row,col),interpolation=cv2.INTER_AREA)
    mask, class_ids = dataset.load_mask(image_id)
    #print('real mask shape:',np.shape(mask))
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)
    if np.shape(bbox)[0]>3:
        continue
    is_show = False
    # Display image and additional stats
    if is_show == True:
        print("image_id ", image_id, dataset.image_reference(image_id))
        log("image", image)
        log("mask", mask)
        log("class_ids", class_ids)
        log("bbox", bbox)
        # Display image and instances
        visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
    results = model.detect([image], verbose=1)
    r = results[0]
    mask_pred = r['masks']
    class_ids_pred = r['class_ids']
    bbox_pred = utils.extract_bboxes(mask_pred)
    if len(bbox_pred) == 0:
        continue
    if is_show == True:
        visualize.display_instances(image, bbox_pred, mask_pred, class_ids_pred, dataset.class_names)
    #print('pred mask shape:',np.shape(mask_pred))

    mask = np.sum(mask.astype(np.uint8),axis = 2)
    mask[mask>1] = 1
    mask = mask.astype(np.uint8)
    if np.sum(mask)/(row*col)<0.1:
        continue
    mask_pred = np.sum(mask_pred.astype(np.uint8),axis = 2)
    mask_pred[mask_pred>1] = 1
    mask_pred = mask_pred.astype(np.uint8)
    mask_error = cal_mask_error(mask, mask_pred,index)
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
        cv2.imwrite(os.path.join('./image_all', img_patch_name),new_image)
        cv2.imwrite('./results/' + img_patch_name + '_' + background_name.split('.')[0] + '.jpeg',comp)
        cv2.imwrite('./masks/' + img_patch_name,mask*255)
        cv2.imwrite('./masks_grab/' + img_patch_name,mask_grab*255)
        cv2.imwrite('./trimaps/' + img_patch_name,trimap)
        cv2.imwrite('./alphas/' + img_patch_name,alpha*255)
print('All Done!')
