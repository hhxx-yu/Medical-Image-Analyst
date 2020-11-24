#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:28:37 2020

@author: ryan
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from outline import *
os.chdir(os.path.dirname(__file__))
model = load_model("final_model_tumor.h5")
from outline import *
valid_ids=sorted(glob("brain_tumor_preprocessed_data/Valid/image/*.tif"))
test_ids=sorted(glob("brain_tumor_preprocessed_data/Test/image/*.tif"))
def get_img_name(file):
    return file.split("/")[-1][:-4]
valid_ids=list(map(get_img_name,valid_ids))
test_ids=list(map(get_img_name,test_ids))
valid_gen = DataGen(valid_ids,path='brain_tumor_preprocessed_data',image_size=image_size,shuffle=False)
test_gen=DataGen(test_ids,path='brain_tumor_preprocessed_data',image_size=image_size,shuffle=False)
#save the predicted image of valid set, yellow line is the true mask, green line is the predicted mask
for i in range(len(valid_gen)):
    x,y=valid_gen.__getitem__(i)
    result = model.predict(x)
    for j in range(x.shape[0]):
        image = outline(x[j],y[j,:,:,0],color=[1,1,0])
        image = outline(image, result[j,:,:, 0], color=[0, 1, 0])
        plt.imsave("output/valid/"+valid_gen.ids[8*i+j]+"pred"+".png",image)
#save the predicted image of test set, yellow line is the true mask, green line is the predicted mask
for i in range(len(test_gen)):
    x,y=test_gen.__getitem__(i)
    result = model.predict(x)
    for j in range(x.shape[0]):
        image = outline(x[j],y[j,:,:,0],color=[1,1,0])
        image = outline(image, result[j,:,:, 0], color=[0, 1, 0])
        plt.imsave("output/test/"+test_gen.ids[8*i+j]+"pred"+".png",image)