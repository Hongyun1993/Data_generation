#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:29:57 2018

@author: lhy
"""
import sys
import numpy as np
from SILC import *
sys.path.append('../')
from utils import *

def fitness(img,mask_grab):

    p = SLICProcessor(img,2500,20)
    p.iterates()
    #Param
    ratio = 0.05
    threshold = 40
    row,col = np.shape(mask_grab)
    rad  = ratio*np.sqrt(row**2+col**2)
    # Find the super pixels which can be matted
    clusters = p.clusters
    num_c = len(clusters)
    clusters_fg = []
    clusters_bg = []
    clusters_uk = []
    row_col_fg = []
    row_col_bg = []
    row_col_uk = []

    for i in range(num_c):
        cluster = clusters[i]
        pixels = np.array(cluster.pixels)
        if np.shape(pixels)[0] == 0:
             continue

        sum_mask = np.sum(mask_grab[pixels[:,0],pixels[:,1]])
        len_region = np.shape(pixels)[0]
        row_col = [cluster.row, cluster.col]
        if sum_mask/len_region == 0:
    #        clusters[i].type = 'bg'
            clusters_bg.append(cluster)
            row_col_bg.append(row_col)
        elif sum_mask/len_region == 1:
    #        clusters[i].type = 'fg'
            clusters_fg.append(cluster)
            row_col_fg.append(row_col)
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

        error_u_f = np.sqrt(np.sum((row_col_fg - row_col_uk_f)**2, axis = 2))
        error_u_b = np.sqrt(np.sum((row_col_bg - row_col_uk_b)**2, axis = 2))

        # store the closed superpixel fg and bg index
        index_u_f = np.array(np.where(error_u_f<rad))
        index_u_b = np.array(np.where(error_u_b<rad))

        fitness_uk = np.zeros((len_uk))
        for i in range(len_uk):
            fg_index = np.squeeze(np.array(np.where(index_u_f[0,:]==i)))
            len_f = len(fg_index)
            fg_lab = np.array([0.,0.,0.])
            for ii in range(len_f):
                index = index_u_f[1,fg_index[ii]]
                cluster_fg = clusters_fg[index]
                fg_lab += np.array([cluster_fg.l, cluster_fg.a, cluster_fg.b])
            fg_lab = fg_lab/len_f

            bg_index = np.squeeze(np.array(np.where(index_u_b[0,:]==i)))
            len_b = len(bg_index)
            bg_lab = np.array([0.,0.,0.])
            for ii in range(len_b):
                index = index_u_b[1,bg_index[ii]]
                cluster_bg = clusters_bg[index]
                bg_lab += np.array([cluster_bg.l, cluster_bg.a, cluster_bg.b])
            bg_lab = bg_lab/len_b

            lab_error = np.sqrt(np.sum((fg_lab - bg_lab)**2))
            print('lab_error:', lab_error)
            if lab_error >= threshold:
                fitness_uk[i] = 1
        return clusters_uk, fitness_uk
    else:
        print('have no sufficient fg, bg and uk')
        return None
