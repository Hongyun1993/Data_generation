#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:54:18 2018

@author: hongyun
"""

import sys
import json
sys.path.append('./Mask_RCNN/samples/coco')
from pycocotools.coco import COCO

ann_path = '../coco/new_images/annotations/instances_train2014.json'
with open(ann_path,'r') as load_f:
    load_dict = json.load(load_f)

coco_detail = COCO(ann_path)
image_ids = coco_detail.imgs