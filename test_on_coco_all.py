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
for o in range(n_num):
    print('-'*20 + str(n_num - o) + '-'*20)
    # Load random image and mask.
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
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

    index = cal_bbox_error(bbox, bbox_pred)
    mask_error = cal_mask_error(mask, mask_pred,index)
    mean_error = np.mean(mask_error)
    img_patch_name = str(mean_error) + '__' + str(image_id) + '.png'
    new_image = image.copy()
    new_image[:,:,0] = image[:,:,2]
    new_image[:,:,2] = image[:,:,0]
    cv2.imwrite(os.path.join('./image_all', img_patch_name),new_image)
print('All Done!')
