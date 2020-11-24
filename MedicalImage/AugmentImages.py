#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:37:39 2020

@author: ryan
"""

'''
I could have use ImageGenerator from keras directory, but flow from directory always mess up with the relations between image
and mask, and use flow is just as verbose, so I just augmented image to another folder and train it on a customized generator.
'''
import os
from datagen_unet import *
from Unet import *
import cv2
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(__file__))

data_gen_args = dict(rotation_range=20,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.01,
                    zoom_range=0.01,
                    horizontal_flip=True,
                    fill_mode='nearest')
'''
Get x,y from the generator here
'''
image_size = 256
train_path = "Preprocessed_Data/Train"
batch_size = 4
def get_img_name(file):
    return file.split("/")[-1][:-4]
train_ids=glob("Preprocessed_Data/Train/image/*.jpg")
train_ids=list(map(get_img_name,train_ids))
train_gen = DataGen(train_ids,batch_size=batch_size,path=train_path,form="jpg",image_size=image_size)
for i in range(len(train_gen)):
    if(i==0):
        x=train_gen.__getitem__(0)[0]
        y=train_gen.__getitem__(0)[1]
    else:
        x=np.append(x,train_gen.__getitem__(i)[1],axis=0)  
        y=np.append(y,train_gen.__getitem__(i)[1],axis=0)  
train_aug = trainGenerator_flow(1,x,y,data_gen_args,image_color_mode = "rgb",target_size = (256,256),
                           save_to_dir1 ="Preprocessed_Data/Aug/image_aug",save_to_dir2="Preprocessed_Data/Aug/mask_Aug",seed=2)

i=0
while i<400:
    next(train_aug)
    i=i+1