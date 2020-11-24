#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:54:31 2020

@author: ryan
"""

import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
os.chdir(os.path.dirname(__file__))

class DataGen(keras.utils.Sequence):
    def __init__(self, ids,path, batch_size=8, image_size=128,form="tif",shuffle=True,color="rgb"):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle=shuffle
        self.on_epoch_end()
        self.form=form
        self.color=color
        
    def __load__(self, id):
        ## Path
        image_path1 = self.path+"/image/"+id+"."+self.form
        mask_path1 = self.path+"/mask/"+id+"_mask."+self.form
        image_path2 = self.path+"/image_aug/"+id+"."+self.form
        mask_path2 = self.path+"/mask_aug/"+id+"_mask."+self.form
        ## Reading Image and Mask
        if self.color=="rgb":
            image = cv2.imread(image_path1, 1)
            if type(image) is not np.ndarray:
                image = cv2.imread(image_path2, 1)
            if type(image) is not np.ndarray:
                raise ValueError([image_path2,image_path1])
            image = cv2.resize(image,(self.image_size, self.image_size))
        if self.color=="gray":
            image = cv2.imread(image_path1, 0)
            if type(image) is not np.ndarray:
                image = cv2.imread(image_path2, 0)
            if type(image) is not np.ndarray:
                raise ValueError([image_path2,image_path1])
            image = cv2.resize(image,(self.image_size, self.image_size))
            image=np.expand_dims(image,axis=-1)
        mask=cv2.imread(mask_path1,0)
        if type(mask) is not np.ndarray:
            mask=cv2.imread(mask_path2,0)
        if type(mask) is np.ndarray:
            mask=cv2.resize(mask, (self.image_size, self.image_size))
            mask=np.expand_dims(mask, axis=-1)
            mask = mask/255.0
        else:
            mask=image
            mask=mask>0
        ## Normalizaing 
        image = image/255.0
        
        
        return image, mask
    
    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.ids):
            files_batch = self.indexs[index*self.batch_size:]
        else:
            files_batch = self.indexs[index*self.batch_size:(index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for index in files_batch:
            _img,_mask=self.__load__(index)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    
    def on_epoch_end(self):
        self.indexs = self.ids
        if self.shuffle == True:
            np.random.shuffle(self.indexs)
    
    def __len__(self):
        return int(np.floor(len(self.ids)/float(self.batch_size)))