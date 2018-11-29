#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:41:35 2018

@author: lhy
"""
import sys
import json

ann_path = '../coco/image/annotations/instances_train2014.json'
new_ann_path = '../coco/new_image/annotations/instances_train2014.json'

with open(ann_path,'r') as load_f:
    load_dict = json.load(load_f)
    with open(new_ann_path,'r') as new_load_f:
        new_load_dict = json.load(new_load_f)
    load_dict['images'].append(new_load_dict['images'])
    load_dict['annotations'].extend(new_load_dict['annotations'])

with open(ann_path,"w") as f:
    json.dump(load_dict, f)
    print('merge done!')
