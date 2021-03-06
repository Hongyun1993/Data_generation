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

if not os.path.exists('./img_patch'):
   os.mkdir('./img_patch')



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

#n_num = 80000
#for o in range(n_num):
#    print('-'*20 + str(n_num - o) + '-'*20)
    # Load random image and mask.
#    image_id = random.choice(dataset.image_ids)
#    image = dataset.load_image(image_id)
#    mask, class_ids = dataset.load_mask(image_id)
#    #print('real mask shape:',np.shape(mask))
#    # Compute Bounding box
#    bbox = utils.extract_bboxes(mask)
#    if np.shape(bbox)[0]>5:
#        continue
#    is_show = True
#    # Display image and additional stats
#    if is_show == True:
#        print("image_id ", image_id, dataset.image_reference(image_id))
#        log("image", image)
#        log("mask", mask)
#        log("class_ids", class_ids)
#        log("bbox", bbox)
#        # Display image and instances
#        visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#    results = model.detect([image], verbose=1)
#    r = results[0]
#    mask_pred = r['masks']
#    class_ids_pred = r['class_ids']
#    bbox_pred = utils.extract_bboxes(mask_pred)
#    if len(bbox_pred) == 0:
#        continue
#    if is_show == True:
#        visualize.display_instances(image, bbox_pred, mask_pred, class_ids_pred, dataset.class_names)
#    #print('pred mask shape:',np.shape(mask_pred))
#
#    index = cal_bbox_error(bbox, bbox_pred)
#    mask_error = cal_mask_error(mask, mask_pred,index)
#    print(mask_error)
#    min_error = np.min(mask_error)
#    index = np.argmin(mask_error)
#    bbox_i = bbox[index,:]
#    ratio = np.abs((bbox_i[2] - bbox_i[0]))*np.abs((bbox_i[3] - bbox_i[1]))
#    row,col = np.shape(image)[:2]
#    if ratio/(row*col) < 0.1:
#        continue
#    img_patch, mask_error = choose_img_patch(image, bbox, mask_error)
#    img_patch_name = str(min_error) + '__' + str(image_id) + '.png'
#    new_img_patch = img_patch.copy()
#    new_img_patch[:,:,0] = img_patch[:,:,2]
#    new_img_patch[:,:,2] = img_patch[:,:,0]
#    cv2.imwrite(os.path.join('./img_patch', img_patch_name),new_img_patch)
#print('All Done!')

image_id = dataset.image_ids[7553]
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
#print('real mask shape:',np.shape(mask))
# Compute Bounding box
bbox = utils.extract_bboxes(mask)
is_show = True
# Display image and additional stats
if is_show == True:
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
