#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:47:05 2020

@author: ryan
"""
import numpy as np
def outline(image, mask, color,threshold=0.5):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image