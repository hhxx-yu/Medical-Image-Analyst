#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:00:42 2020

@author: ryan
"""

from glob import glob
os.chdir(os.path.dirname(__file__))
files=glob("*.jpeg")
import os 
os.chdir(os.getcwd() + os.sep + os.pardir)
import numpy as np
from shutil import copyfile
files=glob("*.jpg")
for i,file in enumerate(files):
    if i%10==0:
        copyfile(file,os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/Test/image',file))
        copyfile(os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/full/mask',file[:-4]+"_mask.jpg")
                 ,os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/Test/mask',file[:-4]+"_mask.jpg"))
    elif i%10==1:
        copyfile(file,os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/Valid/image',file))
        copyfile(os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/full/mask',file[:-4]+"_mask.jpg")
                 ,os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/Valid/mask',file[:-4]+"_mask.jpg"))
    else:
        copyfile(file,os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/Train/image',file))
        copyfile(os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/full/mask',file[:-4]+"_mask.jpg")
                 ,os.path.join(os.path.dirname(__file__),'brain_tumor_preprocessed_data/Train/mask',file[:-4]+"_mask.jpg"))
img_files=sorted(glob("*.jpg"))
mask_files=sorted(glob("*.jpg"))

