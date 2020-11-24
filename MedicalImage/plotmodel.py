#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:05:18 2020

@author: ryan
"""
from tensorflow.keras.models import load_model
model_aug=load_model("final_model_416.h5")
from tensorflow.keras.utils import plot_model
plot_model(model_aug, to_file='model_combined.png')